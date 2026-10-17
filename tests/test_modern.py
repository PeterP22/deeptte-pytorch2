"""Tests for the PyTorch 2.14 upgrade: padded LSTM path, typed loop, bf16, compile."""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from deeptte import train
from deeptte.config import Config
from deeptte.data import TripDataset, collate_fn
from deeptte.models.attr import Attr
from deeptte.models.net import DeepTTE
from deeptte.models.spatio_temporal import SpatioTemporal

CFG = Config()
ROOT = Path(__file__).resolve().parents[1]


def batch(n=16, start=0):
    ds = TripDataset(ROOT / "data/test", CFG)
    return collate_fn([ds[i] for i in range(start, start + n)], CFG)


def packed_reference(st: SpatioTemporal, traj, attr_t):
    """The pre-upgrade forward: PackedSequence through the LSTM."""
    conv = st.geo_conv(traj, CFG)
    a = attr_t.unsqueeze(1)
    conv = torch.cat((conv, a.expand(conv.size()[:2] + (a.size(-1),))), dim=2)
    lens = [n - st.kernel_size + 1 for n in traj["lens"]]
    packed = nn.utils.rnn.pack_padded_sequence(conv, lens, batch_first=True, enforce_sorted=False)
    out, _ = st.rnn(packed)
    hiddens, out_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    return out, st.attent_pooling(hiddens, a, out_lens)


@pytest.mark.parametrize("masked", [False, True])
def test_padded_lstm_matches_packed_reference(masked):
    torch.manual_seed(0)
    attr, traj = batch()
    attr_net = Attr()
    st = SpatioTemporal(attr_size=attr_net.out_size(), masked_attention=masked).eval()
    with torch.no_grad():
        attr_t = attr_net(attr, CFG)
        ref_packed, ref_pooled = packed_reference(st, traj, attr_t)
        packed, _, pooled = st(traj, attr_t, CFG)
    torch.testing.assert_close(pooled, ref_pooled, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(packed.data, ref_packed.data, atol=1e-5, rtol=1e-4)


def test_train_one_epoch_reduces_loss():
    torch.manual_seed(0)
    model = DeepTTE()
    opt = train.make_optimizer(model, 3e-3, torch.device("cpu"))
    batches = [batch(32, 32 * i) for i in range(4)]
    first = train.evaluate_loss(model, [batches], CFG, torch.device("cpu")).loss
    for _ in range(5):
        stats = train.train_one_epoch(model, [batches], opt, CFG, torch.device("cpu"))
    after = train.evaluate_loss(model, [batches], CFG, torch.device("cpu")).loss
    assert stats.trips == 128 and stats.trips_per_s > 0
    assert after < first


def test_bf16_inference_close_to_fp32():
    torch.manual_seed(0)
    model = DeepTTE().eval()
    attr, traj = batch()
    with torch.inference_mode():
        fp32, _ = model.eval_on_batch(attr, traj, CFG)
        with train.autocast(torch.device("cpu"), "bf16"):
            bf16, _ = model.eval_on_batch(attr, traj, CFG)
    assert bf16["pred"].dtype == torch.float32
    rel = ((bf16["pred"] - fp32["pred"]).abs() / fp32["pred"].abs()).max()
    assert rel < 0.05


def test_compile_heads_keeps_checkpoint_format(tmp_path):
    model = DeepTTE()
    keys = set(model.state_dict())
    model.compile_heads()
    assert set(model.state_dict()) == keys
    model.save_checkpoint(tmp_path / "m.pt")
    DeepTTE.from_checkpoint(tmp_path / "m.pt")


def test_deployed_checkpoint_still_loads():
    model = DeepTTE.from_checkpoint(ROOT / "deploy/model.pt")
    assert sum(p.numel() for p in model.parameters()) > 0


def test_main_one_epoch_smoke(tmp_path, monkeypatch):
    (tmp_path / "data").symlink_to(ROOT / "data")
    monkeypatch.chdir(tmp_path)
    train.main(["--epochs", "1", "--max-train-files", "1", "--device", "cpu",
                "--run-name", "smoke", "--threads", "2"])
    rows = (tmp_path / "checkpoints/smoke/metrics.csv").read_text().splitlines()
    assert rows[0].startswith("epoch,train_loss,eval_loss,train_trips_per_s")
    assert (tmp_path / "checkpoints/smoke/best.pt").exists()

import torch

from deeptte.config import Config
from deeptte.data import TripDataset, collate_fn

CFG = Config()


def small_batch(n=4):
    ds = TripDataset("data/test", CFG)
    return collate_fn([ds[i] for i in range(n)], CFG)


def test_attr_output_size():
    from deeptte.models.attr import Attr
    attr, _ = small_batch()
    net = Attr()
    out = net(attr, CFG)
    assert out.shape == (4, net.out_size())
    assert net.out_size() == 16 + 3 + 8 + 1  # driver + week + time embeddings + dist


def test_attr_batch_of_one():
    from deeptte.models.attr import Attr
    attr, _ = small_batch(n=1)
    out = Attr()(attr, CFG)
    assert out.shape == (1, 28)  # original crashed here (unqualified squeeze)


def test_geo_conv_output_shape():
    from deeptte.models.geo_conv import GeoConv
    _, traj = small_batch()
    net = GeoConv(kernel_size=3, num_filter=32)
    out = net(traj, CFG)
    max_len = traj["lngs"].shape[1]
    assert out.shape == (4, max_len - 2, 33)  # T-k+1 windows, num_filter+1 features


def test_spatio_temporal_shapes():
    from deeptte.models.attr import Attr
    from deeptte.models.spatio_temporal import SpatioTemporal
    attr, traj = small_batch()
    attr_net = Attr()
    st = SpatioTemporal(attr_size=attr_net.out_size())
    attr_t = attr_net(attr, CFG)
    packed_hiddens, lens, pooled = st(traj, attr_t, CFG)
    assert pooled.shape == (4, 128)
    assert lens == [l - 2 for l in traj["lens"]]  # kernel_size 3 shrinks by 2


def test_spatio_temporal_mean_pooling():
    from deeptte.models.attr import Attr
    from deeptte.models.spatio_temporal import SpatioTemporal
    attr, traj = small_batch()
    attr_net = Attr()
    st = SpatioTemporal(attr_size=attr_net.out_size(), pooling_method="mean")
    _, _, pooled = st(traj, attr_net(attr, CFG), CFG)
    assert pooled.shape == (4, 128)


def test_net_forward_backward_finite():
    from deeptte.models.net import DeepTTE
    attr, traj = small_batch(n=8)
    model = DeepTTE()
    model.train()
    pred_dict, loss = model.eval_on_batch(attr, traj, CFG)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_net_eval_mode_returns_predictions():
    from deeptte.models.net import DeepTTE
    attr, traj = small_batch(n=8)
    model = DeepTTE()
    model.eval()
    with torch.no_grad():
        pred_dict, loss = model.eval_on_batch(attr, traj, CFG)
    assert pred_dict["pred"].shape == (8, 1)
    assert pred_dict["label"].shape == (8, 1)
    assert (pred_dict["label"] > 0).all()  # unnormalized seconds


def test_checkpoint_roundtrip(tmp_path):
    from deeptte.models.net import DeepTTE
    model = DeepTTE(kernel_size=3, num_filter=32, alpha=0.3)
    path = tmp_path / "ckpt.pt"
    model.save_checkpoint(path)
    loaded = DeepTTE.from_checkpoint(path)
    for a, b in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(a, b)

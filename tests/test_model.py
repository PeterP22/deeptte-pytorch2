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

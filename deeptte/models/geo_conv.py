"""Geo-Conv: learns local spatial features from the GPS point sequence.

Each point = (lng, lat, taxi-state embedding) -> Linear(4,16) -> tanh, then a
1D convolution over the sequence produces one feature vector per length-k
window, concatenated with the (normalized) distance covered by that window.

DeepETA analogue: DeepETA never sees raw GPS traces — it encodes origin /
destination as multi-resolution geohash embeddings. Geo-Conv is the
route-based alternative: convolve over the actual path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_local_seq(full_seq, kernel_size, mean, std):
    """Windowed difference: value covered by each length-k window of the trip.

    E.g. for dist_gap (cumulative), local[i] = full[i+k-1] - full[i].
    Output is re-normalized with the window-level mean/std.
    """
    local_seq = full_seq[:, kernel_size - 1:] - full_seq[:, :-(kernel_size - 1)]
    return (local_seq - mean) / std


class GeoConv(nn.Module):
    def __init__(self, kernel_size=3, num_filter=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, num_filter, kernel_size)

    def forward(self, traj, config):
        lngs = traj["lngs"].unsqueeze(2)
        lats = traj["lats"].unsqueeze(2)
        states = self.state_em(traj["states"].long())

        locs = torch.cat((lngs, lats, states), dim=2)
        locs = torch.tanh(self.process_coords(locs)).permute(0, 2, 1)
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # distance covered by each conv window, as an extra feature channel
        local_dist = get_local_seq(
            traj["dist_gap"], self.kernel_size,
            config.mean("dist_gap"), config.std("dist_gap"),
        )
        return torch.cat((conv_locs, local_dist.unsqueeze(2)), dim=2)

"""Spatio-temporal component: LSTM over Geo-Conv features + pooling.

The trip's per-window features (with the attribute vector appended to every
step) run through a 2-layer LSTM; hidden states are pooled to one vector.

Pooling options: 'attention' (default, matches original), 'mean'.
Attention here is over TIME STEPS of one trip. DeepETA's attention is over
FEATURES of one (tabular) trip — same mechanism, different axis, and that
difference is the heart of DeepTTE-vs-DeepETA.
"""
import torch
import torch.nn as nn

from .geo_conv import GeoConv


class SpatioTemporal(nn.Module):
    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method="attention"):
        super().__init__()
        if pooling_method not in ("attention", "mean"):
            raise ValueError(f"unsupported pooling_method: {pooling_method}")
        self.kernel_size = kernel_size
        self.pooling_method = pooling_method
        self.geo_conv = GeoConv(kernel_size=kernel_size, num_filter=num_filter)
        self.rnn = nn.LSTM(
            input_size=num_filter + 1 + attr_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        if pooling_method == "attention":
            self.attr2atten = nn.Linear(attr_size, 128)

    @staticmethod
    def out_size() -> int:
        return 128

    def mean_pooling(self, hiddens, lens):
        # padded hiddens are 0, so sum/len == masked mean
        summed = torch.sum(hiddens, dim=1)
        lens = lens.to(summed).unsqueeze(1)
        return summed / lens

    def attent_pooling(self, hiddens, attr_t):
        # attr_t arrives already unsqueezed to (B, 1, attr_size)
        attent = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)  # B x 128 x 1
        alpha = torch.exp(-torch.bmm(hiddens, attent))  # B x T x 1
        # Quirk preserved from the original: no explicit padding mask. Padded
        # hiddens are 0 so exp(-0)=1 leaks some weight into the denominator.
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return torch.bmm(hiddens.permute(0, 2, 1), alpha).squeeze(2)

    def forward(self, traj, attr_t, config):
        conv_locs = self.geo_conv(traj, config)

        # append the attribute vector to every time step
        attr_t = attr_t.unsqueeze(1)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size(-1),))
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)

        # conv with kernel k shrinks each trip by k-1 windows
        lens = [l - self.kernel_size + 1 for l in traj["lens"]]

        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            conv_locs, lens, batch_first=True, enforce_sorted=False
        )
        packed_hiddens, _ = self.rnn(packed_inputs)
        hiddens, out_lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        if self.pooling_method == "mean":
            return packed_hiddens, lens, self.mean_pooling(hiddens, out_lens)
        return packed_hiddens, lens, self.attent_pooling(hiddens, attr_t)

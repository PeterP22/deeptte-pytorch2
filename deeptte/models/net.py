"""DeepTTE top-level network: multi-task travel-time estimation.

EntireEstimator: residual MLP head predicting TOTAL trip time from
(attribute vector, pooled spatio-temporal vector).
LocalEstimator: small MLP predicting the time of each local window from the
per-step LSTM hidden states (training-only auxiliary task).
Loss: alpha * local + (1 - alpha) * entire, both relative-error (MAPE-style).

DeepETA analogues: EntireEstimator ~ DeepETA's fully-connected decoder with
bias-adjustment layers; the relative-error loss plays the role DeepETA gives
its asymmetric Huber loss (robustness to outlier trips).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attr import Attr
from .geo_conv import get_local_seq
from .spatio_temporal import SpatioTemporal

EPS = 10  # seconds; keeps the local relative loss from exploding on tiny windows


class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super().__init__()
        self.input2hid = nn.Linear(input_size, hidden_size)
        self.residuals = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(num_final_fcs)
        )
        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)
        hidden = F.leaky_relu(self.input2hid(inputs))
        for layer in self.residuals:
            hidden = hidden + F.leaky_relu(layer(hidden))
        return self.hid2out(hidden)

    def eval_on_batch(self, pred, label, mean, std):
        label = label.view(-1, 1) * std + mean
        pred = pred * std + mean
        loss = torch.abs(pred - label) / label
        return {"label": label, "pred": pred}, loss.mean()


class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))
        hidden = F.leaky_relu(self.hid2hid(hidden))
        return self.hid2out(hidden)

    def eval_on_batch(self, pred, lens, label, mean, std):
        # pack the padded label sequence the same way the hidden states were
        # packed, so pred and label rows line up
        label = nn.utils.rnn.pack_padded_sequence(
            label, lens, batch_first=True, enforce_sorted=False
        ).data
        label = label.view(-1, 1) * std + mean
        pred = pred * std + mean
        # EPS in the denominator only (original behavior — preserve exactly)
        loss = torch.abs(pred - label) / (label + EPS)
        return loss.mean()


class DeepTTE(nn.Module):
    def __init__(self, kernel_size=3, num_filter=32, pooling_method="attention",
                 num_final_fcs=3, final_fc_size=128, alpha=0.3,
                 masked_attention=False, dist_buckets=0, geohash=False):
        super().__init__()
        self.hparams = dict(
            kernel_size=kernel_size, num_filter=num_filter,
            pooling_method=pooling_method, num_final_fcs=num_final_fcs,
            final_fc_size=final_fc_size, alpha=alpha,
            masked_attention=masked_attention, dist_buckets=dist_buckets,
            geohash=geohash,
        )
        self.kernel_size = kernel_size
        self.alpha = alpha

        self.attr_net = Attr(dist_buckets=dist_buckets, geohash=geohash)
        self.spatio_temporal = SpatioTemporal(
            attr_size=self.attr_net.out_size(),
            kernel_size=kernel_size,
            num_filter=num_filter,
            pooling_method=pooling_method,
            masked_attention=masked_attention,
        )
        self.entire_estimate = EntireEstimator(
            input_size=SpatioTemporal.out_size() + self.attr_net.out_size(),
            num_final_fcs=num_final_fcs,
            hidden_size=final_fc_size,
        )
        self.local_estimate = LocalEstimator(input_size=SpatioTemporal.out_size())

        self._init_weight()

    def _init_weight(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr, config)
        # sptm_s: PackedSequence of hidden states; sptm_l: window counts;
        # sptm_t: pooled trip vector
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t, config)
        entire_out = self.entire_estimate(attr_t, sptm_t)
        if self.training:
            local_out = self.local_estimate(sptm_s.data)
            return entire_out, (local_out, sptm_l)
        return entire_out

    def eval_on_batch(self, attr, traj, config):
        if self.training:
            entire_out, (local_out, local_length) = self(attr, traj, config)
        else:
            entire_out = self(attr, traj, config)

        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(
            entire_out, attr["time"], config.mean("time"), config.std("time")
        )

        if not self.training:
            return pred_dict, entire_loss

        # local windows span (kernel_size - 1) gaps
        mean = (self.kernel_size - 1) * config.mean("time_gap")
        std = (self.kernel_size - 1) * config.std("time_gap")
        local_label = get_local_seq(traj["time_gap"], self.kernel_size, mean, std)
        local_loss = self.local_estimate.eval_on_batch(
            local_out, local_length, local_label, mean, std
        )
        return pred_dict, (1 - self.alpha) * entire_loss + self.alpha * local_loss

    def save_checkpoint(self, path):
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return model

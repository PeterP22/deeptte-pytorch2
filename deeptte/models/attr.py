"""Attribute component: embeds per-trip metadata into a dense vector.

DeepETA analogue: DeepETA embeds ALL features this way (including
bucketized continuous ones) before feeding them to a linear transformer.
Here driverID / weekID / timeID are embedded and total distance is appended
— as a raw scalar by default, or (with dist_buckets) bucketized + embedded
the way DeepETA treats every continuous feature. The geohash flag adds
DeepETA-style origin/destination cell embeddings at two resolutions.
"""
import torch
import torch.nn as nn

from ..data import GEO_VOCAB

EMBED_DIMS = (("driverID", 24000, 16), ("weekID", 7, 3), ("timeID", 1440, 8))
GEO_KEYS = ("o_cell_fine", "o_cell_coarse", "d_cell_fine", "d_cell_coarse")


class Attr(nn.Module):
    def __init__(self, dist_buckets=0, geohash=False):
        super().__init__()
        self.dist_buckets = dist_buckets
        self.geohash = geohash
        for name, dim_in, dim_out in EMBED_DIMS:
            self.add_module(name + "_em", nn.Embedding(dim_in, dim_out))
        if dist_buckets:
            # dist arrives ~N(0,1) after collate normalization; equal-z edges
            # approximate DeepETA's quantile buckets (data-derived quantiles
            # would be the refinement)
            self.register_buffer("dist_edges", torch.linspace(-2.5, 2.5, dist_buckets - 1))
            self.dist_em = nn.Embedding(dist_buckets, 8)
        if geohash:
            for name in GEO_KEYS:
                self.add_module(name + "_em", nn.Embedding(GEO_VOCAB, 8))

    def out_size(self) -> int:
        sz = sum(dim_out for _, _, dim_out in EMBED_DIMS)
        sz += 8 if self.dist_buckets else 1
        if self.geohash:
            sz += 8 * len(GEO_KEYS)
        return sz

    def forward(self, attr, config):
        em_list = []
        for name, _, _ in EMBED_DIMS:
            embed = getattr(self, name + "_em")
            em_list.append(embed(attr[name].view(-1, 1)).squeeze(1))

        if self.dist_buckets:
            # attr["dist"] is ALREADY normalized by the collate — bucketize
            # it directly (re-normalizing would collapse everything into a
            # handful of buckets)
            em_list.append(self.dist_em(torch.bucketize(attr["dist"], self.dist_edges)))
        else:
            # Quirk preserved from the original: `dist` was already normalized
            # in collate_fn; the original normalizes it a second time here.
            # Kept for parity (it is just a fixed linear transform).
            dist = config.normalize(attr["dist"], "dist")
            em_list.append(dist.view(-1, 1))

        if self.geohash:
            for name in GEO_KEYS:
                em_list.append(getattr(self, name + "_em")(attr[name]))

        return torch.cat(em_list, dim=1)

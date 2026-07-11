"""Attribute component: embeds per-trip metadata into a dense vector.

DeepETA analogue: DeepETA embeds ALL features this way (including
bucketized continuous ones) before feeding them to a linear transformer.
Here only driverID / weekID / timeID are embedded and total distance is
appended as a raw scalar.
"""
import torch
import torch.nn as nn

EMBED_DIMS = (("driverID", 24000, 16), ("weekID", 7, 3), ("timeID", 1440, 8))


class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        for name, dim_in, dim_out in EMBED_DIMS:
            self.add_module(name + "_em", nn.Embedding(dim_in, dim_out))

    @staticmethod
    def out_size() -> int:
        return sum(dim_out for _, _, dim_out in EMBED_DIMS) + 1  # +1 for dist

    def forward(self, attr, config):
        em_list = []
        for name, _, _ in EMBED_DIMS:
            embed = getattr(self, name + "_em")
            em_list.append(embed(attr[name].view(-1, 1)).squeeze(1))

        # Quirk preserved from the original: `dist` was already normalized in
        # collate_fn; the original normalizes it a second time here. Kept for
        # parity (it is just a fixed linear transform of the input).
        dist = config.normalize(attr["dist"], "dist")
        em_list.append(dist.view(-1, 1))

        return torch.cat(em_list, dim=1)

import math

import torch
from torch import nn
from torch.nn import functional as F


class EventWindows(nn.Module):
    """Pool local clip evidence for a fixed set of video registers."""

    def __init__(self, hidden_size, count, learned):
        super().__init__()
        centers = (torch.arange(count, dtype=torch.float32) + 0.5) / count
        self.register_buffer("center_logits", centers.logit())
        self.offsets = nn.Linear(hidden_size, 2) if learned else None
        if self.offsets is not None:
            nn.init.zeros_(self.offsets.weight)
            nn.init.zeros_(self.offsets.bias)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, clips, conditions):
        batch, length, _ = clips.shape
        centers = self.center_logits.expand(batch, -1)
        log_width = torch.zeros_like(centers)
        if self.offsets is not None:
            center_delta, log_width = self.offsets(conditions).unbind(-1)
            centers = centers + center_delta
        centers = centers.sigmoid()
        widths = (log_width.clamp(-6, 3).exp() / 16).clamp(min=1 / length, max=1)
        time = (torch.arange(length, device=clips.device, dtype=clips.dtype) + 0.5) / length
        distance = (time - centers.unsqueeze(-1)) / widths.unsqueeze(-1)
        weights = (-0.5 * distance.square()).softmax(dim=-1)
        return self.projection(weights @ clips), centers, widths


def event_supervision(queries, local_slots, labels, temperature):
    """Return local retrieval logits and detached per-slot diffusion targets."""
    labels = torch.as_tensor(labels, device=queries.device)
    scores = torch.einsum("qd,bkd->qbk", F.normalize(queries), F.normalize(local_slots, dim=-1))
    batch, count, _ = local_slots.shape
    logits = temperature * (torch.logsumexp(scores / temperature, dim=-1) - math.log(count))
    with torch.no_grad():
        positive = scores[torch.arange(len(labels), device=labels.device), labels]
        values, indices = positive.topk(min(2, count), dim=1)
        assignment = torch.zeros_like(positive).scatter_(1, indices, (values / temperature).softmax(1))
        mass = queries.new_zeros(batch, count).index_add_(0, labels, assignment)
        totals = torch.zeros_like(local_slots).index_add_(
            0, labels, assignment.unsqueeze(-1) * queries.unsqueeze(1))
        targets = totals / mass.clamp_min(1e-12).unsqueeze(-1)
        weights = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)
    return logits, targets, weights

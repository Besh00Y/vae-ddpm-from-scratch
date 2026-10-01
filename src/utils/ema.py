import copy

import torch


class EMA:
    """Exponential moving average of the weights. Sampling from the EMA model gives much better images."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.copy_(b)
"""Exponential moving average of model weights."""
import torch


class EMA:
    """Exponential moving average of the model weights.

    The averaged weights remove the jitter of the raw weights during training.
    Validation and the saved checkpoint use the averaged model.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def swap_in(self, model):
        """Load EMA weights into the model; returns a backup of the raw weights."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow})
        return backup

    def swap_out(self, model, backup):
        model.load_state_dict({**model.state_dict(), **backup})

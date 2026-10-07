"""Exactly eleven unconstrained residual scales; no backbone/source changes."""
import torch
from torch import nn
from independent_wssl import IndependentWSSL

class ActivityScaledWSSL(IndependentWSSL):
    def __init__(self, config):
        super().__init__(config)
        self.activities = tuple(config['data']['activities'])
        if len(self.activities) != self.backbone.activity_count or len(set(self.activities)) != 11:
            raise ValueError('Frozen config activity index mismatch')
        self.activity_index = {name: i for i, name in enumerate(self.activities)}
        # ones consumes no RNG; append after all original parameters are initialized.
        self.activity_ssl_scale = nn.Parameter(torch.ones(len(self.activities)))
        # Instance-bound hook scales the COMPLETE projection output (including bias).
        # The unchanged forward then applies activity/wrist masks before addition.
        # Fresh constructors/load_state_dict only: never deepcopy a hooked model.
        self.wrist_projection.register_forward_hook(self._scale_residual)

    def _scale_residual(self, module, inputs, residual):
        if residual.ndim != 4 or residual.shape[1:3] != (len(self.activities), 2):
            raise ValueError('Expected [subjects, config activities, wrists, features]')
        return residual * self.activity_ssl_scale[None, :, None, None]

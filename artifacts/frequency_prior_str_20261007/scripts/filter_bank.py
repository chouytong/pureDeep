"""Fixed, differentiable five-band time-domain reconstruction at nominal 100 Hz.

No band powers, peaks, entropy, disease labels, fitted state or learned edges.
An even extension [x, reverse(x)] prevents an end-to-start value discontinuity.
Masks act on its 2L-bin rFFT; irFFT is cropped to the original L samples.
"""
import torch
from torch import nn

BANDS = ((.5, 3.), (3., 4.), (4., 6.), (6., 8.), (8., 12.))


class FixedBandBank(nn.Module):
    sample_rate = 100.
    bands = BANDS

    def masks(self, length, device):
        # Double frequencies avoid floating-point edge ambiguity in bin membership.
        frequencies = torch.fft.rfftfreq(2 * length, d=1/self.sample_rate,
                                        device=device, dtype=torch.float64)
        return torch.stack([(frequencies >= low) &
                            ((frequencies <= high) if i == 4 else (frequencies < high))
                            for i, (low, high) in enumerate(self.bands)])

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != 6 or inputs.shape[-1] < 2:
            raise ValueError('FixedBandBank expects [N,6,L>=2], valid unpadded wrists')
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError('Band decomposition requires finite floating point input')
        length = inputs.shape[-1]
        reflected = torch.cat((inputs, inputs.flip(-1)), dim=-1)
        spectrum = torch.fft.rfft(reflected, dim=-1, norm='backward')
        mask = self.masks(length, inputs.device)
        filtered = spectrum[:, None] * mask[None, :, None].to(spectrum.dtype)
        return torch.fft.irfft(filtered, n=2*length, dim=-1, norm='backward')[..., :length]

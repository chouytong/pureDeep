"""Synthetic numerical tests; no subject data or performance selection."""
import json
import torch
from common import HERE, write_json, require
from filter_bank import FixedBandBank, BANDS


def main():
    torch.set_num_threads(4)
    bank = FixedBandBank()
    rows = []
    for length in [976,2000]:
        masks = bank.masks(length, 'cpu')
        require(masks.sum(0).max().item() == 1, 'Band overlap')
        require(bool((masks.sum(1) > 0).all()), 'Empty band at actual activity length')
        require(not masks[:,0].any(), 'DC retained')
        t = torch.arange(length,dtype=torch.float64)+.5
        # DCT cosine modes exactly match the even-extension frequency grid.
        test_hz = [.25, 2., 3.5, 5., 7., 10., 15., 20.]
        for target in test_hz:
            k = round(target * 2 * length / 100)
            hz = k * 100 / (2 * length)
            x = torch.cos(torch.pi*k*t/length)[None,None].expand(1,6,length)
            result = bank(x)
            expected_index = next((i for i,(lo,hi) in enumerate(BANDS)
                                  if hz >= lo and (hz <= hi if i == 4 else hz < hi)), None)
            for i in range(5):
                expected = x if i == expected_index else torch.zeros_like(x)
                require((result[:,i]-expected).abs().max().item() < 1e-10, 'Incorrect frequency mapping')
        # Every available bin at/beside an edge follows the prespecified half-open rule.
        frequencies = torch.fft.rfftfreq(2*length,d=1/100,dtype=torch.float64)
        for edge in [.5,3.,4.,6.,8.,12.]:
            idx = int(torch.argmin(abs(frequencies-edge)))
            for j in range(max(0,idx-1),min(len(frequencies),idx+2)):
                hz = float(frequencies[j])
                expected = [hz >= lo and (hz <= hi if i == 4 else hz < hi)
                            for i,(lo,hi) in enumerate(BANDS)]
                require(masks[:,j].tolist() == expected, 'Boundary membership mismatch')
        x = torch.randn(2,6,length,requires_grad=True)
        result = bank(x)
        require(result.shape == (2,5,6,length) and torch.isfinite(result).all(), 'Shape/finite failure')
        result.square().mean().backward()
        require(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0, 'Differentiability failure')
        for bad in [float('nan'),float('inf')]:
            z = x.detach().clone();z[0,0,0] = bad
            try:bank(z)
            except ValueError:pass
            else:raise AssertionError('Invalid input accepted')
        rows.append(dict(length=length,fft_length=2*length,resolution_hz=100/(2*length),
                         bins_per_band=masks.sum(1).tolist(),synthetic_mapping_pass=True,
                         boundary_membership_pass=True,gradient_finite_nonzero=True))
    write_json(HERE/'analysis/filter_bank_tests.json',dict(status='PASS',actual_lengths=rows,
        bands=BANDS,nominal_sample_rate=100.,learned_parameters=0,synthetic_only=True,
        shape='N x 5 x 6 x L',boundary='half-open, last high inclusive',
        extension='full even reflection; inverse cropped to true length',
        limitations='Finite-resolution ideal masks are noncausal with ringing; reflection assumes even boundary extension, not a physical continuation. No clinical phenotype guarantee.',
        performance_selection=False,outer_access=False))
    print(json.dumps({'status':'PASS','tests':rows}),flush=True)


if __name__ == '__main__':main()

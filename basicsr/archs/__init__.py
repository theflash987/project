"""Architecture construction for the NoRouter-K16 experiment."""

from .modal_content_pole_wavelet_dwt3_arch import (
    AverNetModalContentPoleWaveletDWT3OfficialDense,
)


def build_network(opt):
    return AverNetModalContentPoleWaveletDWT3OfficialDense(**opt)

"""Model construction for the NoRouter-K16 experiment."""

from .modal_content_pole_wavelet_dwt3_video_model import (
    ModalContentPoleWaveletDWT3VideoModel,
)


def build_model(opt):
    return ModalContentPoleWaveletDWT3VideoModel(opt)

"""Components used exclusively by the OfficialDense DWT-3 model."""

import torch
from torch import nn
from torch.nn import functional as F


_BIOR44_ALPHA = -1.586134342059924
_BIOR44_BETA = -0.052980118572961
_BIOR44_GAMMA = 0.882911075530934
_BIOR44_DELTA = 0.443506852043971
_BIOR44_K = 1.149604398


def _right_neighbor(value):
    return torch.cat([value[..., 1:], value[..., -1:]], dim=-1)


def _left_neighbor(value):
    return torch.cat([value[..., :1], value[..., :-1]], dim=-1)


def _interleave(even, odd):
    shape = list(even.shape)
    shape[-1] *= 2
    output = even.new_empty(shape)
    output[..., 0::2] = even
    output[..., 1::2] = odd
    return output


def _dwt_width(value):
    low = value[..., 0::2]
    high = value[..., 1::2]
    high = high + _BIOR44_ALPHA * (low + _right_neighbor(low))
    low = low + _BIOR44_BETA * (_left_neighbor(high) + high)
    high = high + _BIOR44_GAMMA * (low + _right_neighbor(low))
    low = low + _BIOR44_DELTA * (_left_neighbor(high) + high)
    return low * _BIOR44_K, high / _BIOR44_K


def _idwt_width(low, high):
    low = low / _BIOR44_K
    high = high * _BIOR44_K
    low = low - _BIOR44_DELTA * (_left_neighbor(high) + high)
    high = high - _BIOR44_GAMMA * (low + _right_neighbor(low))
    low = low - _BIOR44_BETA * (_left_neighbor(high) + high)
    high = high - _BIOR44_ALPHA * (low + _right_neighbor(low))
    return _interleave(low, high)


def _dwt_height(value):
    low, high = _dwt_width(value.transpose(-2, -1))
    return low.transpose(-2, -1), high.transpose(-2, -1)


def _idwt_height(low, high):
    return _idwt_width(
        low.transpose(-2, -1), high.transpose(-2, -1)
    ).transpose(-2, -1)


def _dwt2d(video):
    batch, frames, channels, height, width = video.shape
    value = video.reshape(batch * frames, channels, height, width)
    low_width, high_width = _dwt_width(value)
    ll, hl = _dwt_height(low_width)
    lh, hh = _dwt_height(high_width)
    coefficients = torch.cat([ll, lh, hl, hh], dim=1)
    return coefficients.reshape(
        batch, frames, 4 * channels, height // 2, width // 2)


def _idwt2d(coefficients):
    batch, frames, channels, height, width = coefficients.shape
    value = coefficients.reshape(batch * frames, channels, height, width)
    ll, lh, hl, hh = torch.chunk(value, 4, dim=1)
    low_width = _idwt_height(ll, hl)
    high_width = _idwt_height(lh, hh)
    restored = _idwt_width(low_width, high_width)
    return restored.reshape(
        batch, frames, channels // 4, 2 * height, 2 * width)


def _split_level(coefficients):
    approximation, lh, hl, hh = torch.chunk(coefficients, 4, dim=2)
    return approximation, torch.cat([lh, hl, hh], dim=2)


def dwt3_bior44(video):
    """Three-level critically sampled bior4.4 pyramid for padded BTCHW input."""
    a1, w1 = _split_level(_dwt2d(video))
    a2, w2 = _split_level(_dwt2d(a1))
    a3, w3 = _split_level(_dwt2d(a2))
    return {
        'a3': a3,
        'w3': w3,
        'w2': w2,
        'w1': w1,
    }


def idwt_level_bior44(approximation, details):
    lh, hl, hh = torch.chunk(details, 3, dim=2)
    return _idwt2d(torch.cat([approximation, lh, hl, hh], dim=2))


class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, value):
        return value + self.body(value)


class LocalBidirectionalAttention(nn.Module):
    """Location-wise current/backward/forward fusion with a shared scorer."""

    def __init__(self, channels, hidden_channels=32, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.current_score = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))
        self.temporal_score = nn.Sequential(
            nn.Conv2d(3 * channels + 1, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))
        nn.init.normal_(
            self.current_score[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.current_score[-1].bias, 0.0)
        nn.init.normal_(
            self.temporal_score[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.temporal_score[-1].bias, 0.0)

    def _temporal_logit(self, reference, candidate, reliability):
        learned = self.temporal_score(torch.cat([
            reference,
            candidate,
            (reference - candidate).abs(),
            reliability,
        ], dim=1))
        logit = learned + torch.log(reliability.clamp(self.eps, 1.0))
        return torch.where(
            reliability > 0,
            logit,
            torch.full_like(logit, torch.finfo(logit.dtype).min))

    def forward(
            self,
            reference,
            backward,
            forward,
            backward_reliability,
            forward_reliability):
        logits = torch.cat([
            self.current_score(reference),
            self._temporal_logit(
                reference, backward, backward_reliability),
            self._temporal_logit(
                reference, forward, forward_reliability),
        ], dim=1)
        weights = torch.softmax(logits, dim=1)
        return {
            'fused': (
                weights[:, 0:1] * reference
                + weights[:, 1:2] * backward
                + weights[:, 2:3] * forward),
            'weights': weights,
        }

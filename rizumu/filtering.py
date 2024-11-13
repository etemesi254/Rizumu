import itertools

import numpy as np
import torch


def _covariance(y_j):
    (nb_frames, nb_bins, nb_channels) = y_j.shape
    Cj = torch.zeros((nb_frames, nb_bins, nb_channels, nb_channels),
                     dtype=y_j.dtype)
    for (i1, i2) in itertools.product(*(range(nb_channels),) * 2):
        Cj[..., i1, i2] = Cj[..., i1, i2] + (y_j[..., i1] * torch.conj(y_j[..., i2]))

    return Cj


def get_local_gaussian_model(y_j: torch.Tensor, eps: float = 1.):
    v_j = torch.mean(torch.abs(y_j) ** 2, axis=2)

    nb_frames = y_j.shape[0]
    R_j = 0
    weight = eps
    for t in range(nb_frames):
        R_j = R_j + _covariance(y_j[None, t, ...])
        weight += v_j[None, t, ...]
    R_j /= weight[..., None, None]
    return v_j, R_j


# Now define the signal-processing low-level functions used by the Separator


def get_mix_model(v, R):
    nb_channels = R.shape[1]
    (nb_frames, nb_bins, nb_sources) = v.shape
    Cxx = torch.zeros((nb_frames, nb_bins, nb_channels, nb_channels), dtype=R.dtype)
    for j in range(nb_sources):
        Cxx += v[..., j, None, None] * R[None, ..., j]
    return Cxx


def _invert(M, eps):
    """
    Invert matrices, with special fast handling of the 1x1 and 2x2 cases.

    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.

    Parameters
    ----------
    M: np.ndarray [shape=(..., nb_channels, nb_channels)]
        matrices to invert: must be square along the last two dimensions

    eps: [scalar]
        regularization parameter to use _only in the case of matrices
        bigger than 2x2

    Returns
    -------
    invM: np.ndarray, [shape=M.shape]
        inverses of M
    """
    nb_channels = M.shape[-1]
    if nb_channels == 1:
        # scalar case
        invM = 1.0 / (M + eps)
    elif nb_channels == 2:
        # two channels case: analytical expression
        det = (
                M[..., 0, 0] * M[..., 1, 1] -
                M[..., 0, 1] * M[..., 1, 0])

        invDet = 1.0 / (det)
        invM = torch.empty_like(M)
        invM[..., 0, 0] = invDet * M[..., 1, 1]
        invM[..., 1, 0] = -invDet * M[..., 1, 0]
        invM[..., 0, 1] = -invDet * M[..., 0, 1]
        invM[..., 1, 1] = invDet * M[..., 0, 0]
    else:
        # general case : no use of analytical expression (slow!)
        invM = torch.linalg.pinv(M, eps)
    return invM


def wiener_gain(v_j, R_j, inv_Cxx):
    (_, nb_channels) = R_j.shape[:2]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = torch.zeros_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),) * 3):
        G[..., i1, i2] += (R_j[None, :, i1, i3] * inv_Cxx[..., i3, i2])
    G *= v_j[..., None, None]
    return G


def expectation_maximization(
        y: torch.Tensor,
        x: torch.Tensor,
        iterations: int = 2,
        eps: float = 1e-10,
        batch_size: int = 200,
):
    if eps is None:
        eps = torch.finfo(torch.real(x[0]).dtype).eps
    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = y.shape[-1]
    R = torch.zeros((nb_bins, nb_channels, nb_channels, nb_sources), dtype=x.dtype)
    v = torch.zeros((nb_frames, nb_bins, nb_sources))

    regularization = np.sqrt(eps) * (
        torch.tile(torch.eye(nb_channels, dtype=torch.complex64),
                   (1, nb_bins, 1, 1)))
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        for j in range(nb_sources):
            v[..., j], R[..., j] = get_local_gaussian_model(y[..., j], eps)
        for t in range(nb_frames):
            Cxx = get_mix_model(v[None, t, ...], R)
            Cxx += regularization
            inv_Cxx = _invert(Cxx, eps)
            # separate the sources
            for j in range(nb_sources):
                W_j = wiener_gain(v[None, t, ..., j], R[..., j], inv_Cxx)
                y[t, ..., j] = apply_filter(x[None, t, ...], W_j)[0]
    return y


def apply_filter(x, W):
    nb_channels = W.shape[-1]

    # apply the filter
    y_hat = 0 + 0j
    for i in range(nb_channels):
        y_hat += W[..., i] * x[..., i, None]
    return y_hat


def wiener(
        targets_spectrograms: torch.Tensor,
        mix_stft: torch.Tensor,
        iterations: int = 1,
        softmask: bool = False,
        residual: bool = False,
        scale_factor: float = 10.0,
        eps: float = 1e-10,
):
    y = targets_spectrograms * torch.exp(1j * torch.angle(mix_stft)[..., None])
    if iterations == 0:
        return y
    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = max(1, (torch.abs(targets_spectrograms).max()) / 10.0)
    pe = expectation_maximization(y / max_abs, mix_stft, iterations, eps=eps)

    return pe * max_abs

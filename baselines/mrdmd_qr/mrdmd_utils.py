"""mrDMD modal-library construction for TPU sensor-placement experiments.

Implements the recursive multi-resolution DMD algorithm of Kutz, Fu &
Brunton (2016), SIAM J. Appl. Dyn. Syst. 15(2):713-735, with optional
switches for Hankel time-delay embedding and forward/backward DMD bias
correction:

  - Optional time-delay (Hankel) embedding via `d_embed`.
  - Optional forward/backward DMD with sqrtm de-biasing (Dawson 2016).
  - Recursive multi-resolution decomposition over L dyadic levels with a
    cycles-per-window cutoff `max_cyc`.
  - Per-(level, bin) DMD retains "slow" modes (|f| <= max_cyc / window_len);
    fast residual passes to the next, halved level.
  - Across all levels and (when applicable) all wind angles, the retained
    modes are concatenated into a real-valued library Phi_real by stacking
    [Re(Phi), Im(Phi)] (complex modes generally come in conjugate pairs;
    real-stacking preserves the spanned subspace exactly).
  - An optional final orthonormalization (SVD) yields a numerically
    well-conditioned basis suitable for QR-pivoting sensor selection and
    L1/L2 reconstruction.
"""

import numpy as np
from scipy.linalg import sqrtm


def hankel_embed(X, d):
    """Stack d time-shifted copies of X. Input (n, m) -> output (n*d, m-d+1)."""
    n, m = X.shape
    cols = m - d + 1
    if d <= 1 or cols < 2:
        return X.copy()
    return np.vstack([X[:, k:k + cols] for k in range(d)])


def _trunc_svd(X, r_max=None, tol_rel=1e-10):
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    if S.size == 0:
        return None
    tol = tol_rel * S[0]
    r = int(np.sum(S > tol))
    if r_max is not None:
        r = min(r, int(r_max))
    if r < 1:
        return None
    return U[:, :r], S[:r], Vh[:r, :]


def fb_dmd(X1, X2, r_max=None, use_fb=True):
    """Forward/backward exact DMD. Returns (Phi, lam, b) or None if rank 0.

    use_fb=True  → forward + backward + sqrtm de-biasing (Dawson 2016)
    use_fb=False → forward-only textbook Exact DMD (Tu 2014, Kutz 2016)
                   = paper baseline path; numerically robust on large data

    Phi : (n, r) complex DMD modes
    lam : (r,)  complex discrete-time eigenvalues
    b   : (r,)  complex initial amplitudes (Phi @ diag(b) reconstructs X1[:,0])
    """
    fwd = _trunc_svd(X1, r_max=r_max)
    if fwd is None:
        return None
    Uf, Sf, Vhf = fwd
    Atil = Uf.conj().T @ X2 @ Vhf.conj().T / Sf

    Acomb = Atil
    r_eff = Atil.shape[0]   # effective rank used by downstream Phi reconstruction
    if use_fb:
        bwd = _trunc_svd(X2, r_max=r_max)
        if bwd is not None:
            Ub, Sb, Vhb = bwd
            Abtil = Ub.conj().T @ X1 @ Vhb.conj().T / Sb
            try:
                r_try = min(Atil.shape[0], Abtil.shape[0])
                cand = sqrtm(Atil[:r_try, :r_try] @ np.linalg.pinv(Abtil[:r_try, :r_try]))
                if np.all(np.isfinite(cand)):
                    Acomb = cand
                    r_eff = r_try   # FB-DMD reduced the rank; downstream must follow
            except Exception:
                pass

    lam, W = np.linalg.eig(Acomb)
    # Phi reconstruction must use the SAME rank as W. When FB-DMD truncates to
    # r_eff < Sf.size we slice Vhf and Sf accordingly; otherwise r_eff == Sf.size
    # and this is a no-op.
    Phi = X2 @ Vhf[:r_eff, :].conj().T @ np.diag(1.0 / Sf[:r_eff]) @ W
    try:
        b = np.linalg.lstsq(Phi, X1[:, 0].astype(Phi.dtype), rcond=None)[0]
    except Exception:
        b = np.zeros(Phi.shape[1], dtype=Phi.dtype)
    return Phi, lam, b


def mrdmd(X, dt=1e-3, L=5, max_cyc=5, r_max=None, use_fb=True,
          _level=1, _store=None):
    """Recursive mrDMD. Returns list of (level, Phi_slow, omega_slow, b_slow).

    omega = log(lam) / dt is the continuous-time eigenvalue;
    cutoff is on |omega.imag| / (2 pi) <= max_cyc / window_length.
    """
    if _store is None:
        _store = []
    m = X.shape[1]
    if m < 4:
        return _store

    T = (m - 1) * dt
    f_cut = (max_cyc / T) if T > 0 else np.inf

    res = fb_dmd(X[:, :-1], X[:, 1:], r_max=r_max, use_fb=use_fb)
    if res is None:
        return _store
    Phi, lam, b = res

    lam_safe = np.where(np.abs(lam) > 1e-300, lam, 1e-300 + 0j)
    omega = np.log(lam_safe.astype(complex)) / dt
    finite = np.isfinite(omega) & np.isfinite(b)
    freq_hz = np.abs(omega.imag) / (2.0 * np.pi)
    slow = finite & (freq_hz <= f_cut)

    R = X
    if np.any(slow):
        Phi_s = Phi[:, slow]
        om_s = omega[slow]
        b_s = b[slow]
        _store.append((_level, Phi_s, om_s, b_s))
        # Subtract slow-mode reconstruction; recurse on residual.
        t = np.arange(m) * dt
        # Phi_s @ diag(b_s) @ exp(om_s * t)
        time_dyn = np.exp(np.outer(om_s, t)) * b_s[:, None]
        Xs = (Phi_s @ time_dyn).real.astype(X.dtype, copy=False)
        R = X - Xs

    if _level < L:
        half = m // 2
        if half >= 4:
            mrdmd(R[:, :half], dt, L, max_cyc, r_max, use_fb, _level + 1, _store)
            mrdmd(R[:, half:], dt, L, max_cyc, r_max, use_fb, _level + 1, _store)
    return _store


def build_mrdmd_basis(train_per_angle, n_space,
                      d_embed=30, L=5, max_cyc=5, dt=1e-3,
                      r_max_per_dmd=50, amp_quantile=0.0,
                      orthonormalize=True, use_fb=True, verbose=True):
    """Construct mrDMD modal library across (optionally) multiple wind angles.

    train_per_angle : list of (n_space, T_a) arrays. One mrDMD per entry.
    n_space         : number of physical sensor locations (=500 for TPU).
    Returns (Phi_basis, info).
        Phi_basis : (n_space, r) real array, optionally orthonormalized.
        info      : dict with diagnostic counts.

    Spatial part of each Hankel-augmented mode = first n_space rows of Phi.
    Real-stacking [Re | Im] keeps the same spanned subspace (modes appear in
    conjugate pairs anyway) while letting downstream code stay real-valued.
    """
    all_phi_spatial = []
    all_amp = []
    n_modes_per_angle = []

    for ai, X in enumerate(train_per_angle):
        Xh = hankel_embed(X, d_embed)
        modes = mrdmd(Xh, dt=dt, L=L, max_cyc=max_cyc, r_max=r_max_per_dmd,
                      use_fb=use_fb)
        n_a = 0
        for lvl, Phi, om, b in modes:
            all_phi_spatial.append(Phi[:n_space, :])
            all_amp.append(np.abs(b))
            n_a += Phi.shape[1]
        n_modes_per_angle.append(n_a)
        if verbose:
            print(f"  angle {ai:2d}: kept {n_a:4d} modes "
                  f"across {len(modes):3d} (level,bin) pairs")

    if not all_phi_spatial:
        raise RuntimeError("mrDMD produced zero modes; check inputs/params")

    Phi_all = np.hstack(all_phi_spatial)               # (n_space, R) complex
    amp_all = np.concatenate(all_amp)
    if amp_quantile > 0.0:
        thr = np.quantile(amp_all, amp_quantile)
        keep = amp_all > thr
        Phi_all = Phi_all[:, keep]
        if verbose:
            print(f"  amplitude filter q={amp_quantile}: "
                  f"{Phi_all.shape[1]} / {amp_all.size} modes retained")

    Phi_real = np.hstack([Phi_all.real, Phi_all.imag])
    norms = np.linalg.norm(Phi_real, axis=0)
    if norms.max() > 0:
        Phi_real = Phi_real[:, norms > 1e-10 * norms.max()]

    if orthonormalize:
        U, S, _ = np.linalg.svd(Phi_real, full_matrices=False)
        keep = S > 1e-10 * S[0]
        Phi_basis = U[:, keep]
        sv = S[keep]
    else:
        Phi_basis = Phi_real
        sv = np.linalg.norm(Phi_real, axis=0)

    info = {
        "n_modes_per_angle": n_modes_per_angle,
        "raw_complex_columns": int(sum(n_modes_per_angle)),
        "real_stacked_columns": int(2 * sum(n_modes_per_angle)),
        "basis_rank": int(Phi_basis.shape[1]),
        "singular_values": sv,
    }
    if verbose:
        print(f"  mrDMD basis: rank={info['basis_rank']} "
              f"(from {info['real_stacked_columns']} real-stacked columns)")
    return Phi_basis, info

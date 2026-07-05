from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Union, List, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        x.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

def initialize_models(N, input_dim=1,
                      hidden_dim=32,
                      layer_dim=1,
                      output_dim=1):

    models = []

    for _ in range(N):
        model = LSTMModel(
            input_dim,
            hidden_dim,
            layer_dim,
            output_dim
        ).to(device)

        models.append(model)

    return models

def get_model_vector(model):
    return torch.cat([
        p.data.view(-1)
        for p in model.parameters()
    ])

def set_model_vector(model, vec):

    pointer = 0
    for p in model.parameters():
        num = p.numel()
        p.data = vec[pointer:pointer+num].view_as(p)
        pointer += num

def compute_distance_matrix(positions):
    """
    Compute pairwise Euclidean distances between sensors.

    Parameters
    ----------
    positions : np.ndarray, shape (N,2)

    Returns
    -------
    dist_matrix : np.ndarray, shape (N,N)
    """
    positions = np.asarray(positions, dtype=float)
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=2)

def plot_sensor_grid_with_kernel(
    positions,
    L,
    sigma=1.0,
    r_max=3.0,
    sentinel_idx=None,
    source_idx=None,
):
    """
    Plot sensors on an L x L grid with Gaussian influence kernel, clipped at max radius.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N,2) with sensor positions
    L : float
        Grid size
    sigma : float
        Standard deviation of Gaussian kernel (distance decay)
    r_max : float
        Maximum interaction radius (beyond this, influence is zero)
    sentinel_idx : array-like or None
        Indices of sentinel nodes to highlight in red.
    source_idx : int or None
        Index of the hidden source node to highlight with a yellow star.
    """
    plt.figure(figsize=(10, 8))
    positions = np.asarray(positions)
    # All sensors
    plt.scatter(
        positions[:, 0],
        positions[:, 1],
        c='black',
        s=150,
        label='Sensors',
        zorder=3
    )

    # Sentinels
    if sentinel_idx is not None and len(sentinel_idx) > 0:
        sentinel_idx = np.asarray(sentinel_idx, dtype=int)
        plt.scatter(
            positions[sentinel_idx, 0],
            positions[sentinel_idx, 1],
            c='red',
            s=150,
            edgecolors='black',
            linewidths=1.5,
            label='Sentinels',
            zorder=4
        )

    # Source node
    if source_idx is not None:
        source_idx = int(source_idx)
        plt.scatter(
            positions[source_idx, 0],
            positions[source_idx, 1],
            c='yellow',
            s=350,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            label='Source',
            zorder=5
        )

    # Label each node with its index
    for i, (x0, y0) in enumerate(positions):
        plt.text(
            x0,
            y0 + 1.5,          # small vertical offset so text appears above the dot
            str(i),
            fontsize=12,
            ha='center',
            va='bottom',
            color='black',
            zorder=6
        )
    # Create a grid for visualization
    grid_size = 200
    x = np.linspace(0, L, grid_size)
    y = np.linspace(0, L, grid_size)
    X, Y = np.meshgrid(x, y)

    influence = np.zeros_like(X, dtype=float)
    for pos in positions:
        dx = X - pos[0]
        dy = Y - pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        contribution = np.exp(-(dist**2) / (2 * sigma**2))
        contribution[dist > r_max] = 0
        if np.max(contribution) > 0:
            contribution = contribution / np.max(contribution)
        influence += contribution

    influence = np.clip(influence, 0, 1)

    # im = plt.imshow(
    #     influence,
    #     origin='lower',
    #     extent=(0, L, 0, L),
    #     cmap='Oranges',
    #     alpha=0.5,
    #     zorder=1
    # )

    for pos in positions:
        circle = plt.Circle(
            pos, r_max,
            color='black',
            fill=False,
            linestyle='-',
            alpha=0.2
        )
        plt.gca().add_patch(circle)

    extra_space = 0
    plt.xlim(0 - extra_space, L + extra_space)
    plt.ylim(0 - extra_space, L + extra_space)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(
        f'Decentralized Network',
        fontsize=20,
        fontweight='bold'
    )
    plt.xlabel('X', fontsize=20, fontweight='bold')
    plt.ylabel('Y', fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=15)

    # cbar = plt.colorbar(im)
    # cbar.set_label('Influence intensity', fontsize=20, fontweight='bold')
    # cbar.ax.tick_params(labelsize=20)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# def build_physical_matrix(positions, 
#                           sigma=1.0, 
#                           r_max=3.0,
#                           self_weight=0.3,
#                           eps=1e-12
#                           ):

#     """
#     Build a static row-stochastic matrix W to use for BOTH:
#       - physical propagation (local diffusion)
#       - communication mixing (gossip)

#     Steps:
#       1) compute Gaussian kernel weights within radius r_max
#       2) set diagonal to self_weight (explicit self-confidence)
#       3) row-normalize so each row sums to 1

#     Parameters
#     ----------
#     positions : np.ndarray, shape (N,2)
#     sigma : float
#         Gaussian kernel std
#     r_max : float
#         cutoff radius (weights outside are 0)
#     self_weight : float in [0,1]
#         desired diagonal/self-confidence weight after mixing
#         (important for stable gossip; typical 0.2–0.5)
#     eps : float
#         numerical stabilizer

#     Returns
#     -------
#     W : np.ndarray, shape (N,N), row-stochastic
#     """
#     positions = np.asarray(positions, dtype=float)
#     N = positions.shape[0]
#     if sigma <= 0:
#         raise ValueError("sigma must be > 0")
#     if r_max <= 0:
#         raise ValueError("r_max must be > 0")
#     if not (0.0 <= self_weight <= 1.0):
#         raise ValueError("self_weight must be in [0,1]")
    

#     dist = np.linalg.norm(
#         positions[:,None,:] - positions[None,:,:],
#         axis=2
#     )

#     # Gaussian kernel with cutoff
#     K = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
#     K[dist > r_max] = 0.0

#     # Remove diagonal for neighbor normalization step
#     np.fill_diagonal(K, 0.0)

#     # Row-normalize neighbor weights (handle isolated nodes)
#     row_sums = K.sum(axis=1, keepdims=True)
#     isolated = (row_sums.squeeze() <= eps)

#     row_sums[row_sums <= eps] = 1.0
#     K = K / row_sums


#     # Add explicit self-confidence
#     W = (1.0 - self_weight) * K
#     np.fill_diagonal(W, self_weight)

#     # If isolated, force identity row (only self)
#     if np.any(isolated):
#         W[isolated, :] = 0.0
#         W[isolated, isolated] = 1.0

#     # Final safety normalization
#     W = W / (W.sum(axis=1, keepdims=True) + eps)

#     return W

def build_uniform_neighbor_matrix(
    positions,
    r_max=3.0,
    self_weight=0.0,
    eps=1e-12,
):
    """
    Row-stochastic W with equal weights among neighbors inside radius r_max.

    Convention:
      W[j, i] > 0 means j sends to i.
    So each row j distributes its mass equally to its neighbors.

    If self_weight > 0, each node keeps that fraction on itself and
    spreads the remaining mass equally among its neighbors.
    """
    positions = np.asarray(positions, dtype=float)
    N = positions.shape[0]

    if r_max <= 0:
        raise ValueError("r_max must be > 0")
    if not (0.0 <= self_weight <= 1.0):
        raise ValueError("self_weight must be in [0, 1]")

    dist = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2
    )

    A = (dist <= r_max).astype(float)
    np.fill_diagonal(A, 0.0)

    W = np.zeros((N, N), dtype=float)

    for j in range(N):
        nbrs = np.where(A[j] > 0)[0]
        d = len(nbrs)

        if d == 0:
            W[j, j] = 1.0
            continue

        W[j, nbrs] = (1.0 - self_weight) / d
        W[j, j] = self_weight

    W = W / (W.sum(axis=1, keepdims=True) + eps)
    return W

def isolate_nodes(A, nodes, cut_incoming=True, cut_outgoing=True):
    """
    Remove selected nodes from the coupling graph.

    For data dynamics:
      - cut_incoming=True  : other nodes no longer receive influence from infected nodes
      - cut_outgoing=True  : infected nodes no longer receive influence from others

    Usually for isolation you want both True.
    """
    A = np.asarray(A, dtype=float).copy()
    nodes = np.atleast_1d(nodes).astype(int)

    if cut_outgoing:
        A[nodes, :] = 0.0
    if cut_incoming:
        A[:, nodes] = 0.0

    return A

def build_node_datasets(X, seq_len):

    datasets = []

    for i in range(X.shape[1]):
        x, y = create_sequences(X[:, i], seq_len)
        datasets.append((x, y))

    return datasets

def local_update(model, optimizer, x, y):

    model.train()

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer.zero_grad()

    pred = model(x).squeeze()
    loss = torch.mean((pred - y)**2)

    loss.backward()
    optimizer.step()

    return loss.item()

def gossip_step(models, A_t):
    param_vectors = [get_model_vector(m).clone() for m in models]
    new_params = []

    for i in range(len(models)):
        # mixed = 0
        mixed = torch.zeros_like(param_vectors[i])
        for j in range(len(models)):
            mixed += A_t[j, i] * param_vectors[j]
        new_params.append(mixed)

    for model, vec in zip(models, new_params):
        set_model_vector(model, vec)

def compute_residual(model, x, y):
    """
    Compute residual y - y_hat for a single sample
    x: (seq_len,)
    y: scalar
    """
    model.eval()
    with torch.no_grad():
        # reshape to (batch=1, seq_len, input_dim=1)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        # x shape: (1, seq_len, 1)
        pred = model(x).cpu().numpy()[0, 0]
    return y - pred


def generate_sensor_positions(
    n: int,
    L: float,
    seed: int | None = None,
    margin_frac: float = 0.05,
    min_dist: float = 1.0,
    max_tries: int = 100_000,
) -> np.ndarray:
    """
    Generate n sensor positions uniformly at random in a square [0, L] x [0, L],
    while enforcing a minimum separation between nodes.

    Parameters
    ----------
    n : int
        Number of sensors.
    L : float
        Side length of the square region.
    seed : int or None, optional
        Random seed for reproducibility.
    margin_frac : float, optional
        Fraction of L used as margin from each border.
    min_dist : float, optional
        Minimum allowed Euclidean distance between any two nodes.
    max_tries : int, optional
        Maximum number of candidate samples before giving up.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) with (x, y) sensor coordinates.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if L <= 0:
        raise ValueError("L must be > 0")
    if not (0.0 <= margin_frac < 0.5):
        raise ValueError("margin_frac must be in [0, 0.5)")
    if min_dist < 0:
        raise ValueError("min_dist must be >= 0")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")

    rng = np.random.default_rng(seed)

    margin = margin_frac * L
    low = margin
    high = L - margin

    positions = []
    tries = 0

    while len(positions) < n and tries < max_tries:
        candidate = rng.uniform(low, high, size=2)

        if not positions:
            positions.append(candidate)
        else:
            existing = np.asarray(positions)
            dists = np.linalg.norm(existing - candidate, axis=1)
            if np.all(dists >= min_dist):
                positions.append(candidate)

        tries += 1

    if len(positions) < n:
        raise RuntimeError(
            f"Could not place all {n} nodes with min_dist={min_dist}. "
            "Try reducing min_dist or increasing max_tries."
        )

    return np.asarray(positions, dtype=float)

def select_sentinels(
    positions: np.ndarray,
    M: int,
    seed: int | None = None,
    first: str = "centroid",
    return_indices: bool = True,
):
    """
    Select M sentinel nodes from N node positions to maximize spatial coverage.

    This uses greedy farthest-point sampling: after choosing the first sentinel,
    each next sentinel is the node with the largest distance to its nearest
    already-selected sentinel.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2) containing node coordinates.
    M : int
        Number of sentinels to select.
    seed : int or None, optional
        Random seed used only when first='random'.
    first : {'centroid', 'random'}, optional
        Strategy for selecting the first sentinel:
        - 'centroid': choose the node closest to the centroid
        - 'random': choose a random node
    return_indices : bool, optional
        If True, return sentinel indices. Otherwise return sentinel positions.

    Returns
    -------
    np.ndarray
        Sentinel indices of shape (M,) if return_indices=True,
        else sentinel coordinates of shape (M, 2).
    """
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")

    N = positions.shape[0]
    if N == 0:
        raise ValueError("positions must contain at least one node")
    if not (1 <= M <= N):
        raise ValueError("M must satisfy 1 <= M <= N")
    if first not in {"centroid", "random"}:
        raise ValueError("first must be either 'centroid' or 'random'")

    rng = np.random.default_rng(seed)

    # Choose first sentinel
    if first == "random":
        first_idx = rng.integers(N)
    else:
        centroid = positions.mean(axis=0)
        first_idx = np.argmin(np.linalg.norm(positions - centroid, axis=1))

    selected = [first_idx]
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[first_idx] = True

    # Distance from each node to its nearest selected sentinel
    min_dists = np.linalg.norm(positions - positions[first_idx], axis=1)

    for _ in range(1, M):
        # Exclude already selected nodes
        min_dists[selected_mask] = -np.inf

        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        selected_mask[next_idx] = True

        # Update nearest-sentinel distances
        dists_to_new = np.linalg.norm(positions - positions[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)

    selected = np.asarray(selected, dtype=int)

    if return_indices:
        return selected
    return positions[selected]


class Page2015Detector:
    """Online change-point detector (Python-first indexing).

    - All indices are **Python 0-based**.
    - `start_idx` is the first sample index to start using (like your Thres but 0-based).
    - The first `m` samples starting at `start_idx` form the baseline.
    - After baseline, we update a growing post-baseline window variance and compute DT(t).
    - When DT(t) >= critical value, we declare detection at the **current time index**.

    update(x_t) -> Optional[int]
        Returns detection index (0-based) once detected, else None.

    detect(y) -> Optional[int]
        Batch convenience wrapper around update.
    """


    """ Parameters
    ----------
    start_idx : int
        0-based index of the first sample to start using. Samples with index < start_idx
        are ignored. The next `m` samples starting at `start_idx` define the baseline window.

    m : int
        Baseline window length (number of samples). The detector estimates baseline variance
        from these `m` samples and uses them to build the long-run covariance matrix DP.

    g : float
        Exponent used in the Page (2015) scaling term in the DT statistic:
            denom = (1 + t/m) * (( (t/m) / (1 + t/m) ) ** g)
        Typical values (as in the original MATLAB lookup) are 0 or 0.25.

    B : float
        Tuning parameter used only to select the critical value `ca` from the lookup table
        (together with `g` and `significance_level`). In the original MATLAB code, supported
        values are {0.5, 1, 2, 4}.

    significance_level : float
        Desired significance level for thresholding DT. This selects the critical value `ca`
        from the lookup table. Supported values (per the original MATLAB code) are 0.95 or 0.99.
        Larger values correspond to a higher threshold (fewer false alarms, more detection delay).

    kernel : Callable[[float], float], optional
        Kernel function K_b(·) used when building DP:
            DP = sum_{t,u=0..m-2} K_b((t-u)/r) * Sy[:,t] * Sy[:,u]^T
        If not provided, a Bartlett (triangular) kernel is used:
            K(x) = 1 - |x|  for |x| <= 1, else 0.
        Replace this with your exact MATLAB K_b if you have one.

    eps : float, default 1e-12
        Numerical stabilizer used when computing DP^{-1/2}. Eigenvalues of DP are clipped
        below by `eps` to avoid division by zero / instability when DP is near-singular.

    Notes on internal indexing
    --------------------------
    - All returned indices are Python 0-based.
    - Detection time is the current sample index `i` at which DT first exceeds `ca`.
    - `t` in DT(t) counts post-baseline samples: t = 1,2,... after the baseline ends.
    """


    '''Example usage:
        import numpy as np
        import matplotlib.pyplot as plt

        # --- assumes the refined Page2015Detector class is already defined above / imported ---


        # ---------------------------
        # 1) Make a synthetic series
        # ---------------------------
        np.random.seed(0)

        T = 600
        true_cp = 300  # 0-based index where variance changes

        sigma1 = 1.0
        sigma2 = 3.0

        y = np.empty(T)
        y[:true_cp] = sigma1 * np.random.randn(true_cp)
        y[true_cp:] = sigma2 * np.random.randn(T - true_cp)

        # ---------------------------------
        # 2) Run detector (streaming)
        # ---------------------------------
        det = Page2015Detector(
            start_idx=0,            # start using data from y[0]
            m=80,                   # baseline length
            g=0.25,
            B=1.0,
            significance_level=0.95,
        )

        DT_trace = np.full(T, np.nan)
        detected = None

        for t in range(T):
            hit = det.update(y[t])
            if det.last_DT is not None:
                DT_trace[t] = det.last_DT
            if hit is not None:
                detected = hit
                break

        print(f"True change point (0-based):   {true_cp}")
        print(f"Detected index (0-based):      {detected}")
        if detected is not None:
            print(f"Detection delay (samples):     {detected - true_cp}")

        # ---------------------------
        # 3) Plot the time series
        # ---------------------------
        plt.figure()
        plt.plot(y)
        plt.axvline(true_cp, linestyle="--", label="true cp", color='black')
        if detected is not None:
            plt.axvline(detected, linestyle="--", label="detected", color='red')
        plt.title("Synthetic series (variance change)")
        plt.xlabel("t (0-based)")
        plt.ylabel("y[t]")
        plt.legend()
        plt.show()

        # ---------------------------
        # 4) Plot DT(t) and threshold
        # ---------------------------
        plt.figure()
        plt.plot(DT_trace)
        plt.axhline(det.threshold, linestyle="-", label="threshold", color='red')
        plt.axvline(true_cp, linestyle="--", label="true cp", color='black')
        if detected is not None:
            plt.axvline(detected, linestyle="--", label="detected", color='red')
        plt.title("DT(t) trace")
        plt.xlabel("t (0-based)")
        plt.ylabel("DT")
        plt.legend()
        plt.show()

        # ---------------------------------
        # 5) Batch mode (same result)
        # ---------------------------------
        det2 = Page2015Detector(start_idx=0, m=80, g=0.25, B=1.0, significance_level=0.95)
        hit2 = det2.detect(y)
        print("\nBatch detect index (0-based):", hit2)
    
    '''

        # critical values from your MATLAB if/else ladder
    _CA_095 = {
        (0.0, 0.5): 1.2887,
        (0.0, 1.0): 1.5819,
        (0.0, 2.0): 1.8268,
        (0.0, 4.0): 2.0082,
        (0.25, 0.5): 1.8076,
        (0.25, 1.0): 2.0197,
        (0.25, 2.0): 2.1477,
        (0.25, 4.0): 2.2217,
    }
    _CA_099 = {
        (0.0, 0.5): 1.6254,
        (0.0, 1.0): 1.9701,
        (0.0, 2.0): 2.2430,
        (0.0, 4.0): 2.4825,
        (0.25, 0.5): 2.1967,
        (0.25, 1.0): 2.4614,
        (0.25, 2.0): 2.6488,
        (0.25, 4.0): 2.7831,
    }

    @dataclass
    class _WelfordState:
        """Per-dimension running stats for sample variance on a growing window."""
        n: int
        mean: np.ndarray
        M2: np.ndarray

        @classmethod
        def init(cls, d: int) -> "Page2015Detector._WelfordState":
            return cls(n=0, mean=np.zeros(d, dtype=float), M2=np.zeros(d, dtype=float))

        def update(self, x: np.ndarray) -> None:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

        def sample_var_matlab(self) -> np.ndarray:
            # MATLAB var default: sample variance (n-1); for n<=1 returns 0
            if self.n <= 1:
                return np.zeros_like(self.mean)
            return self.M2 / (self.n - 1)

    def __init__(
        self,
        *,
        start_idx: int = 0,
        m: int,
        g: float,
        B: float,
        significance_level: float,
        kernel: Optional[Callable[[float], float]] = None,
        eps: float = 1e-12,
    ) -> None:
        if m < 1:
            raise ValueError("m must be >= 1")
        if start_idx < 0:
            raise ValueError("start_idx must be >= 0")

        self.start_idx = int(start_idx)
        self.m = int(m)
        self.g = float(g)
        self.B = float(B)
        self.significance_level = float(significance_level)
        self.kernel = kernel or self._bartlett_kernel
        self.eps = float(eps)

        self.ca = self._critical_value(self.g, self.B, self.significance_level)
        self.reset()

    # ---------- public helpers ----------
    @property
    def detected_index(self) -> Optional[int]:
        return self._detected_idx

    @property
    def last_DT(self) -> Optional[float]:
        return self._last_DT

    @property
    def threshold(self) -> float:
        return self.ca

    def reset(self) -> None:
        self._i = -1  # current time index; incremented at start of update()
        self._baseline: Optional[np.ndarray] = None  # (d, <=m)
        self._d: Optional[int] = None

        self._Vm: Optional[np.ndarray] = None           # (d,)
        self._inv_sqrt_DP: Optional[np.ndarray] = None  # (d,d)

        self._post = self._WelfordState.init(d=1)  # placeholder until d known
        self._t_post = 0  # post-baseline sample count

        self._detected_idx: Optional[int] = None
        self._last_DT: Optional[float] = None

    def update(self, x_t: Union[float, np.ndarray, list]) -> Optional[int]:
        """
        Feed one new observation at the next time index.
        - scalar for 1D
        - array-like shape (d,) for d-dimensional series

        Returns detection time index (0-based) once detected, else None.
        """
        if self._detected_idx is not None:
            return self._detected_idx

        self._i += 1
        i = self._i

        x = np.asarray(x_t, dtype=float).reshape(-1)

        # wait until start_idx
        if i < self.start_idx:
            return None

        # initialize baseline buffer
        if self._baseline is None:
            self._baseline = x[:, None]  # (d,1)
            return None

        # collect baseline until m samples
        if self._baseline.shape[1] < self.m:
            if x.shape[0] != self._baseline.shape[0]:
                raise ValueError("Dimension changed in stream.")
            self._baseline = np.concatenate([self._baseline, x[:, None]], axis=1)

            if self._baseline.shape[1] == self.m:
                self._prepare_baseline()
            return None

        # after baseline: compute DT using growing post-baseline variance
        if self._d is None or self._Vm is None or self._inv_sqrt_DP is None:
            raise RuntimeError("Baseline not prepared correctly.")

        if x.shape[0] != self._d:
            raise ValueError("Dimension changed in stream.")

        self._t_post += 1
        self._post.update(x)

        var_seg = self._post.sample_var_matlab()  # (d,)
        Sk = np.abs(var_seg - self._Vm)           # (d,)

        t = self._t_post
        m = self.m
        num = np.linalg.norm((t / np.sqrt(m)) * (self._inv_sqrt_DP @ Sk), ord=2)
        denom = (1.0 + (t / m)) * (((t / m) / (1.0 + (t / m))) ** self.g)
        DT = float(num / denom)
        self._last_DT = DT

        if DT >= self.ca:
            self._detected_idx = i  # Python 0-based detection time
            return i

        return None

    def detect(self, y: Union[np.ndarray, list]) -> Optional[int]:
        """Batch mode: feed a whole series and return detection index (0-based) or None."""
        self.reset()
        Y = self._as_d_by_t(y)  # (d,T)
        for t in range(Y.shape[1]):
            hit = self.update(Y[:, t])
            if hit is not None:
                return hit
        return None

    # ---------- internals ----------
    @staticmethod
    def _bartlett_kernel(x: float) -> float:
        ax = abs(float(x))
        return 1.0 - ax if ax <= 1.0 else 0.0

    @classmethod
    def _critical_value(cls, g: float, B: float, significance_level: float) -> float:
        gk = round(float(g), 8)
        Bk = round(float(B), 8)

        if np.isclose(significance_level, 0.95):
            table = cls._CA_095
        elif np.isclose(significance_level, 0.99):
            table = cls._CA_099
        else:
            raise ValueError("significance_level must be 0.95 or 0.99 (per your MATLAB code).")

        try:
            return table[(gk, Bk)]
        except KeyError as e:
            raise ValueError(f"Unsupported (g,B)=({g},{B}) for significance_level={significance_level}.") from e

    @staticmethod
    def _as_d_by_t(y: Union[np.ndarray, list]) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return y[None, :]
        if y.ndim != 2:
            raise ValueError("y must be 1D or 2D array-like.")
        # heuristic: if it looks like (T,d), transpose -> (d,T)
        return y.T if y.shape[0] > y.shape[1] else y

    def _prepare_baseline(self) -> None:
        assert self._baseline is not None
        d, m = self._baseline.shape
        self._d = d

        # Vm = var(Y(:,1:m)) with MATLAB behavior
        if m <= 1:
            Vm = np.zeros(d, dtype=float)
        else:
            Vm = np.var(self._baseline, axis=1, ddof=1)  # (d,)
        self._Vm = Vm

        # Sy = (1/sqrt(m)) * ((Y^2 - mean(Y^2))) over baseline
        mean_sq = np.mean(self._baseline ** 2, axis=1)                 # (d,)
        Sy = (self._baseline ** 2 - mean_sq[:, None]) / np.sqrt(m)     # (d,m)

        r = max(int(np.floor(m ** 0.25)), 1)

        # DP sum over t,u=1..m-1 (MATLAB). Use 0..m-2.
        DP = np.zeros((d, d), dtype=float)
        for t in range(m - 1):
            for u in range(m - 1):
                DP += self.kernel((t - u) / r) * np.outer(Sy[:, t], Sy[:, u])

        self._inv_sqrt_DP = self._inv_sqrtm_psd(DP, eps=self.eps)

        # init post-baseline running stats
        self._post = self._WelfordState.init(d=d)
        self._t_post = 0

    @staticmethod
    def _inv_sqrtm_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Inverse matrix square-root for symmetric PSD A via eigendecomposition."""
        A = np.asarray(A, dtype=float)
        if A.size == 1:
            v = float(A.reshape(()))
            return np.array([[1.0 / np.sqrt(max(v, eps))]], dtype=float)

        A = 0.5 * (A + A.T)  # symmetrize
        w, V = np.linalg.eigh(A)
        w = np.clip(w, eps, None)
        return (V * (1.0 / np.sqrt(w))) @ V.T


class OnlineDataGenerator:
    """
    Online generator with independent node-wise time series and optional mean/variance shift.

    Each node evolves independently:

        x_next[i] = signal_persistence * x_prev[i] + epsilon[i]

    where, before the shift:

        epsilon[i] ~ N(0, noise_std^2)

    For the selected shift_node, starting at shift_round:
    - the mean is shifted by mean_shift
    - the variance is increased by variance_shift

    If shift_persistent=True, the shift remains active for all later rounds.
    If shift_persistent=False, the shift is applied only at round == shift_round.
    """

    def __init__(
        self,
        N,
        seq_len,
        batch_size,
        signal_persistence=0.7,
        noise_std=0.05,
        shift_round=None,
        shift_node=None,
        mean_shift=0.8,
        variance_shift=0.1,
        shift_persistent=True,
        init_std=None,
        clip_value=None,
        seed=None,
    ):
        self.N = int(N)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)

        self.signal_persistence = float(signal_persistence)
        self.noise_std = float(noise_std)

        self.shift_round = None if shift_round is None else int(shift_round)
        self.shift_node = None if shift_node is None else int(shift_node)
        self.mean_shift = float(mean_shift)
        self.variance_shift = float(variance_shift)
        self.shift_persistent = bool(shift_persistent)

        self.init_std = float(noise_std if init_std is None else init_std)
        self.clip_value = clip_value

        if self.N < 1:
            raise ValueError("N must be >= 1")
        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not (0.0 <= self.signal_persistence):
            raise ValueError("signal_persistence must be >= 0")
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        if self.variance_shift < 0:
            raise ValueError("variance_shift must be >= 0")
        if self.shift_node is not None and not (0 <= self.shift_node < self.N):
            raise ValueError("shift_node must be in [0, N-1]")

        self.rng = np.random.default_rng(seed)
        self.current_round = 0

        # Rolling history, shape: (seq_len, N)
        self.history = self.rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(self.seq_len, self.N),
        )

    def reset_history(self, history=None):
        """
        Reset the rolling history buffer.

        Parameters
        ----------
        history : array-like of shape (seq_len, N), optional
            If provided, use this history. Otherwise sample a fresh one.
        """
        if history is None:
            self.history = self.rng.normal(
                loc=0.0,
                scale=self.init_std,
                size=(self.seq_len, self.N),
            )
        else:
            history = np.asarray(history, dtype=float)
            if history.shape != (self.seq_len, self.N):
                raise ValueError(
                    f"history must have shape ({self.seq_len}, {self.N}), "
                    f"got {history.shape}"
                )
            self.history = history.copy()

    def reset_round(self, round_idx=0):
        """Reset the internal round counter."""
        self.current_round = int(round_idx)

    def set_shift(
        self,
        shift_round=None,
        shift_node=None,
        mean_shift=None,
        variance_shift=None,
        shift_persistent=None,
    ):
        """Update shift configuration."""
        if shift_round is not None:
            self.shift_round = int(shift_round)

        if shift_node is not None:
            shift_node = int(shift_node)
            if not (0 <= shift_node < self.N):
                raise ValueError("shift_node must be in [0, N-1]")
            self.shift_node = shift_node

        if mean_shift is not None:
            self.mean_shift = float(mean_shift)

        if variance_shift is not None:
            variance_shift = float(variance_shift)
            if variance_shift < 0:
                raise ValueError("variance_shift must be >= 0")
            self.variance_shift = variance_shift

        if shift_persistent is not None:
            self.shift_persistent = bool(shift_persistent)

    def _shift_active(self, round_idx):
        """Return True if the shift should be active at this round."""
        if self.shift_round is None or self.shift_node is None:
            return False

        if self.shift_persistent:
            return round_idx >= self.shift_round
        return round_idx == self.shift_round

    def generate_round(self, round_idx=None, return_states=False):
        """
        Generate one round of data.

        Parameters
        ----------
        round_idx : int, optional
            External round index. If None, the internal round counter is used.
        return_states : bool, optional
            If True, also return the generated states for each batch step.

        Returns
        -------
        X_round : ndarray, shape (N, batch_size, seq_len)
            Input sequence window for each node.
        y_round : ndarray, shape (N, batch_size)
            Next generated value for each node.
        states_round : ndarray, shape (batch_size, N), optional
            Generated next state at each batch step.
        """
        if round_idx is None:
            round_idx = self.current_round
            advance_counter = True
        else:
            round_idx = int(round_idx)
            advance_counter = False

        X_round = np.zeros((self.N, self.batch_size, self.seq_len), dtype=float)
        y_round = np.zeros((self.N, self.batch_size), dtype=float)
        states_round = (
            np.zeros((self.batch_size, self.N), dtype=float)
            if return_states else None
        )

        history = self.history.copy()
        shift_active = self._shift_active(round_idx)

        for k in range(self.batch_size):
            X_round[:, k, :] = history.T

            x_prev = history[-1, :]
            x_next = (
                self.signal_persistence * x_prev
                + self.rng.normal(0.0, self.noise_std, size=self.N)
            )

            if shift_active:
                j = self.shift_node
                x_next[j] += self.mean_shift
                x_next[j] += self.rng.normal(0.0, self.variance_shift)

            if self.clip_value is not None:
                x_next = np.clip(x_next, -self.clip_value, self.clip_value)

            y_round[:, k] = x_next

            if return_states:
                states_round[k, :] = x_next

            history = np.vstack([history[1:], x_next[None, :]])

        self.history = history

        if advance_counter:
            self.current_round += 1

        if return_states:
            return X_round, y_round, states_round
        return X_round, y_round





# =========================================================
# Refined paper-aligned implementation
# =========================================================

class MeanShiftCUSUMDetector:
    """
    One-sided standardized CUSUM for positive mean shifts.

    Intended for residual-energy streams after a warmup period.
    Fit on nominal warmup data, then update online with one scalar per round.
    Once alarmed, the detector stays alarmed.
    """

    def __init__(self, k: float = 0.5, h: float = 6.0, eps: float = 1e-8):
        self.k = float(k)
        self.h = float(h)
        self.eps = float(eps)
        self.fitted = False
        self.reset_runtime()

    def reset_runtime(self):
        self.t = 0
        self.mu0 = None
        self.sigma0 = None
        self.S = 0.0
        self.alarm_time = None
        self.stat_trace = []
        self.z_trace = []

    def fit(self, baseline_values):
        x = np.asarray(baseline_values, dtype=float).reshape(-1)
        if x.size < 2:
            raise ValueError("Need at least 2 warmup observations to fit CUSUM baseline.")
        self.reset_runtime()
        self.mu0 = float(np.mean(x))
        self.sigma0 = float(max(np.std(x, ddof=1), self.eps))
        self.fitted = True
        return self

    def update(self, x_t: float, round_idx: Optional[int] = None) -> Optional[int]:
        if not self.fitted:
            raise RuntimeError("Call fit() on warmup data before update().")
        if self.alarm_time is not None:
            self.stat_trace.append(self.S)
            self.z_trace.append(self.z_trace[-1] if self.z_trace else 0.0)
            self.t += 1
            return self.alarm_time

        z = (float(x_t) - self.mu0) / self.sigma0
        self.S = max(0.0, self.S + z - self.k)
        self.stat_trace.append(self.S)
        self.z_trace.append(z)
        self.t += 1
        if self.S >= self.h:
            self.alarm_time = int(self.t - 1 if round_idx is None else round_idx)
            return self.alarm_time
        return None

def snapshot_output_layer(model):
    return {
        "weight": model.fc.weight.detach().clone(),
        "bias": None if model.fc.bias is None else model.fc.bias.detach().clone(),
    }


def apply_persistent_output_fault(
    model,
    anchor,
    delta_weight=0.0,
    delta_bias=0.0,
    mode="single",
):
    """
    Overwrite only the output layer with anchor + delta.
    This gives a persistent, directional fault instead of random noise.
    """
    with torch.no_grad():
        w = anchor["weight"].clone().to(
            device=model.fc.weight.device,
            dtype=model.fc.weight.dtype,
        )

        if mode == "single":
            # perturb one output weight coordinate
            w.view(-1)[-1] += float(delta_weight)
        elif mode == "all":
            # perturb all output weights equally
            w += float(delta_weight)
        else:
            raise ValueError("mode must be 'single' or 'all'")

        model.fc.weight.copy_(w)

        if model.fc.bias is not None:
            if anchor["bias"] is None:
                b = torch.zeros_like(model.fc.bias)
            else:
                b = anchor["bias"].clone().to(
                    device=model.fc.bias.device,
                    dtype=model.fc.bias.dtype,
                )
            b += float(delta_bias)
            model.fc.bias.copy_(b)


def adjacency_from_mixing_matrix(W):
    """
    Undirected 0/1 graph adjacency extracted from a mixing matrix.

    We use this only for graph-distance calculations:
    one hop = one graph edge.
    """
    W = np.asarray(W, dtype=float)
    A = ((W > 0) | (W.T > 0)).astype(int)
    np.fill_diagonal(A, 0)
    return A


def all_pairs_hop_distances(A):
    """
    Shortest-path hop distance matrix.
    """
    G = nx.from_numpy_array(np.asarray(A))
    N = A.shape[0]
    D = np.full((N, N), np.inf, dtype=float)

    for src, dd in nx.all_pairs_shortest_path_length(G):
        for dst, hops in dd.items():
            D[src, dst] = hops

    return D


def hop_ball(A, center, radius):
    """
    Nodes within <= radius hops of center.
    """
    center = int(center)
    radius = int(max(0, radius))
    G = nx.from_numpy_array(np.asarray(A))
    lengths = nx.single_source_shortest_path_length(G, center, cutoff=radius)
    return set(lengths.keys())


def hop_shell(A, center, inner_radius, guard_hops=1):
    """
    Nodes in the shell:
        inner_radius+1 <= d(center, node) <= inner_radius+guard_hops
    """
    center = int(center)
    inner_radius = int(max(0, inner_radius))
    guard_hops = int(max(1, guard_hops))

    G = nx.from_numpy_array(np.asarray(A))
    lengths = nx.single_source_shortest_path_length(
        G,
        center,
        cutoff=inner_radius + guard_hops
    )

    shell = {
        node for node, d in lengths.items()
        if inner_radius + 1 <= d <= inner_radius + guard_hops
    }
    return shell


def spread_one_hop(A, infected_nodes, quarantined_nodes=None):
    """
    True fault spread:
    infection expands by one graph hop per round, but cannot pass through
    quarantined nodes.
    """
    A = np.asarray(A)
    infected_nodes = set(int(i) for i in infected_nodes)
    quarantined_nodes = set() if quarantined_nodes is None else set(int(i) for i in quarantined_nodes)

    active = infected_nodes - quarantined_nodes
    if not active:
        return set(infected_nodes)

    newly_reached = set()
    for j in active:
        nbrs = np.where(A[j] > 0)[0]
        for u in nbrs:
            if u not in quarantined_nodes:
                newly_reached.add(int(u))

    return infected_nodes | newly_reached


def build_quarantine_matrix(W_base, quarantine_nodes):
    """
    Remove quarantined nodes from gossip.
    Quarantined nodes become isolated self-loops.
    """
    Wq = np.asarray(W_base, dtype=float).copy()
    quarantine_nodes = sorted(set(int(i) for i in quarantine_nodes))

    for q in quarantine_nodes:
        Wq[q, :] = 0.0
        Wq[:, q] = 0.0
        Wq[q, q] = 1.0

    return Wq


def estimate_source_and_region(
    alarm_times,
    sentinel_idx,
    hop_dist,
    current_round,
    tau_det=2,
    guard_hops=1,
    min_required_alarms=2,
    candidate_nodes=None,
):
    """
    Estimate:
      - source_hat
      - t0_hat
      - radius_hat
      - affected_hat (estimated infected ball)
      - quarantine_hat (estimated blocking shell)

    Model:
      alarm_time[s] ~= t0 + d(source, s) + tau_det

    Non-alarms are used as inequality constraints:
      if sentinel s has not alarmed by current_round, then
      t0 + d(source, s) + tau_det > current_round
      should hold approximately.
    """
    sentinel_idx = [int(s) for s in sentinel_idx]
    N = hop_dist.shape[0]
    if candidate_nodes is None:
        candidate_nodes = list(range(N))
    else:
        candidate_nodes = [int(v) for v in candidate_nodes]

    alarmed = [s for s in sentinel_idx if alarm_times.get(int(s)) is not None]
    if len(alarmed) < int(min_required_alarms):
        return None

    best_source = None
    best_t0 = None
    best_score = np.inf

    for v in candidate_nodes:
        dists = np.array([hop_dist[v, s] for s in alarmed], dtype=float)
        if np.any(np.isinf(dists)):
            continue

        implied_t0 = np.array(
            [alarm_times[s] - hop_dist[v, s] - tau_det for s in alarmed],
            dtype=float
        )

        t0_hat = int(round(np.median(implied_t0)))

        # consistency with alarmed sentinels
        score = float(np.mean((implied_t0 - t0_hat) ** 2))

        # penalize candidates that should already have triggered non-alarmed sentinels
        for s in sentinel_idx:
            if alarm_times.get(s) is None:
                pred_alarm = t0_hat + hop_dist[v, s] + tau_det
                if pred_alarm <= current_round:
                    score += float((current_round - pred_alarm + 1.0) ** 2)

        if score < best_score:
            best_score = score
            best_source = int(v)
            best_t0 = int(t0_hat)

    if best_source is None:
        return None

    radius_hat = max(0, int(current_round - best_t0))

    affected_hat = {
        j for j in range(N)
        if hop_dist[best_source, j] <= radius_hat
    }

    quarantine_hat = {
        j for j in range(N)
        if radius_hat + 1 <= hop_dist[best_source, j] <= radius_hat + int(guard_hops)
    }

    frontier_hat = {
        j for j in range(N)
        if hop_dist[best_source, j] == radius_hat + 1
    }

    return {
        "source_hat": best_source,
        "t0_hat": best_t0,
        "radius_hat": radius_hat,
        "affected_hat": affected_hat,
        "frontier_hat": frontier_hat,
        "quarantine_hat": quarantine_hat,
        "fit_score": float(best_score),
    }




def reset_optimizer_state(optimizer):
    """
    Clear momentum / Adam moments after a hard parameter reset.
    """
    if hasattr(optimizer, "state") and optimizer.state is not None:
        optimizer.state.clear()


def mean_model_vector(models, node_indices, weights=None):
    """
    Weighted average of model parameter vectors over selected nodes.
    """
    node_indices = [int(i) for i in node_indices]
    if len(node_indices) == 0:
        raise ValueError("node_indices must be non-empty")

    vecs = [get_model_vector(models[i]).detach().clone() for i in node_indices]

    if weights is None:
        weights = np.ones(len(node_indices), dtype=float) / len(node_indices)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or len(weights) != len(node_indices):
            raise ValueError("weights must match node_indices")
        s = weights.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = weights / s

    out = torch.zeros_like(vecs[0])
    for a, v in zip(weights, vecs):
        out += float(a) * v
    return out


def healed_reference_for_node(models, W_base, node_idx, healthy_pool):
    """
    Build a healed parameter vector for node_idx from healthy neighbors.
    Fallback: average over all healthy nodes if no healthy in-neighbors exist.
    """
    node_idx = int(node_idx)
    healthy_pool = sorted(set(int(i) for i in healthy_pool))

    if len(healthy_pool) == 0:
        raise ValueError("healthy_pool is empty")

    donors = [j for j in healthy_pool if W_base[j, node_idx] > 0]

    if len(donors) > 0:
        w = np.array([W_base[j, node_idx] for j in donors], dtype=float)
        return mean_model_vector(models, donors, weights=w)

    return mean_model_vector(models, healthy_pool)


def heal_nodes_from_healthy_pool(
    models,
    W_base,
    nodes_to_heal,
    healthy_pool,
    optimizers=None,
):
    """
    Hard-reset selected nodes using healthy references.
    """
    nodes_to_heal = sorted(set(int(i) for i in nodes_to_heal))
    healthy_pool = sorted(set(int(i) for i in healthy_pool))

    if len(nodes_to_heal) == 0:
        return

    for i in nodes_to_heal:
        ref = healed_reference_for_node(
            models=models,
            W_base=W_base,
            node_idx=i,
            healthy_pool=healthy_pool,
        )
        set_model_vector(models[i], ref)

        if optimizers is not None:
            reset_optimizer_state(optimizers[i])


def build_disabled_matrix(W_base, disabled_nodes, eps=1e-12):
    """
    Remove disabled nodes from the active network.
    Disabled nodes become isolated self-loops.
    Active receivers are re-normalized over active senders.
    """
    Wd = np.asarray(W_base, dtype=float).copy()
    disabled_nodes = sorted(set(int(i) for i in disabled_nodes))
    disabled_set = set(disabled_nodes)
    N = Wd.shape[0]

    for q in disabled_nodes:
        Wd[q, :] = 0.0
        Wd[:, q] = 0.0
        Wd[q, q] = 1.0

    # re-normalize incoming mass for active receivers
    for i in range(N):
        if i in disabled_set:
            continue
        col_sum = Wd[:, i].sum()
        if col_sum > eps:
            Wd[:, i] /= col_sum
        else:
            Wd[i, i] = 1.0

    return Wd

def load_rssi_excel_data(
    excel_path,
    data_start_row=2,
    normalize=True,
    eps=1e-8,
):
    """
    Load RSSI data from Excel.

    We ignore the first `data_start_row` rows.
    The number of nodes N = number of columns.
    Each column is one node time series.

    If normalize=True:
        each node series is standardized independently:
            s_norm = (s - mean_j) / std_j
    """
    raw = pd.read_excel(excel_path, header=None, engine="openpyxl")

    N = raw.shape[1]

    series_list = []
    raw_series_list = []
    lengths = []
    means = []
    stds = []

    for j in range(N):
        s = pd.to_numeric(
            raw.iloc[data_start_row:, j],
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        if len(s) == 0:
            raise ValueError(f"Column {j} has no valid RSSI samples.")

        raw_series_list.append(s.copy())

        mu = float(np.mean(s))
        sigma = float(np.std(s, ddof=0))
        if sigma < eps:
            sigma = 1.0

        if normalize:
            s = (s - mu) / sigma

        series_list.append(s)
        lengths.append(len(s))
        means.append(mu)
        stds.append(sigma)

    lengths = np.asarray(lengths, dtype=int)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    min_len = int(lengths.min())

    return {
        "N": int(N),
        "series_list": series_list,          # normalized if normalize=True
        "raw_series_list": raw_series_list,  # always kept
        "lengths": lengths,
        "min_len": min_len,
        "means": means,
        "stds": stds,
    }


class ExcelBlockDataGenerator:
    """
    Use real RSSI time series instead of AR(1).

    At each communication round:
      - each node uses a block of length block_size
      - create_sequences(block, seq_len) is applied locally
      - output shapes match your current framework
    """
    def __init__(
        self,
        excel_path,
        seq_len,
        block_size=100,
        data_start_row=2,
        normalize=True,
        eps=1e-8,
    ):
        meta = load_rssi_excel_data(
            excel_path=excel_path,
            data_start_row=data_start_row,
            normalize=normalize,
            eps=eps,
        )

        self.excel_path = excel_path
        self.seq_len = int(seq_len)
        self.block_size = int(block_size)

        self.N = int(meta["N"])
        self.series_list = meta["series_list"]
        self.raw_series_list = meta["raw_series_list"]
        self.lengths = meta["lengths"]
        self.min_len = int(meta["min_len"])
        self.means = meta["means"]
        self.stds = meta["stds"]

        self.rounds = self.min_len // self.block_size
        self.current_round = 0

        if self.block_size <= self.seq_len + 1:
            raise ValueError(
                f"block_size must be > seq_len + 1. "
                f"Got block_size={self.block_size}, seq_len={self.seq_len}"
            )

    def reset_round(self, round_idx=0):
        self.current_round = int(round_idx)

    def generate_round(self, round_idx=None, return_states=False):
        if round_idx is None:
            round_idx = self.current_round
            advance_counter = True
        else:
            round_idx = int(round_idx)
            advance_counter = False

        if not (0 <= round_idx < self.rounds):
            raise IndexError(
                f"round_idx={round_idx} out of range. "
                f"Valid range is 0,...,{self.rounds - 1}"
            )

        start = round_idx * self.block_size
        stop = start + self.block_size

        X_round = []
        y_round = []

        states_round = (
            np.zeros((self.block_size, self.N), dtype=float)
            if return_states else None
        )

        for i, series in enumerate(self.series_list):
            block = np.asarray(series[start:stop], dtype=float)

            if len(block) != self.block_size:
                raise ValueError(
                    f"Node {i}: expected block of length {self.block_size}, "
                    f"got {len(block)}"
                )

            x_i, y_i = create_sequences(block, self.seq_len)

            X_round.append(x_i)
            y_round.append(y_i)

            if return_states:
                states_round[:, i] = block

        X_round = np.asarray(X_round, dtype=float)
        y_round = np.asarray(y_round, dtype=float)

        if advance_counter:
            self.current_round += 1

        if return_states:
            return X_round, y_round, states_round

        return X_round, y_round
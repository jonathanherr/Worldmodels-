import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wm.gpu_polish import batch_energy, seed_pairs  # noqa: E402
from wm.pack_opt import MARGIN, energy_grad  # noqa: E402


def test_batch_energy_matches_numba():
    rng = np.random.default_rng(0)
    k = 6
    i = np.arange(k * k)
    x = (i % k + 0.5) * 0.97 + rng.normal(0, 0.05, k * k)
    y = (i // k + 0.5) * 0.97 + rng.normal(0, 0.05, k * k)
    t = rng.normal(0, 0.3, k * k)
    L = k * 0.97
    I, J = seed_pairs(x, y, 3.0)
    e, _ = batch_energy(*(torch.tensor(a)[None] for a in (x, y, t)), torch.tensor([L], dtype=torch.float64),
                        torch.tensor(I), torch.tensor(J))
    assert float(e[0]) == pytest.approx(energy_grad(x, y, t, L, MARGIN)[0], rel=1e-9)

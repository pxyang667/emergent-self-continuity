
import numpy as np

class SelfSlot:
    def __init__(self, self_dim: int, hidden_dim: int, err_dim: int, tau: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.self_dim = int(self_dim)
        self.hidden_dim = int(hidden_dim)
        self.err_dim = int(err_dim)
        self.tau = float(tau)
        self.Wf = (0.2 * rng.normal(size=(self.self_dim, self.hidden_dim + self.err_dim + 1))).astype(np.float32)
        self.bf = np.zeros((self.self_dim,), dtype=np.float32)
        self.reset()

    def reset(self, carry_slow: float = 0.0):
        """Reset episode-local state.

        carry_slow in [0,1] keeps a decayed copy of `slow` across episodes.
        This is an explicit "worldline coupling" knob:
          - carry_slow=0.0 : fully fresh episodes (previous behavior)
          - carry_slow>0.0 : identity inertia (harder to perfectly rollback)
        """
        self.fast = np.zeros((self.self_dim,), dtype=np.float32)
        if carry_slow and carry_slow > 0.0:
            carry = float(max(0.0, min(1.0, carry_slow)))
            prev = getattr(self, "slow", None)
            if prev is None:
                self.slow = np.zeros((self.self_dim,), dtype=np.float32)
            else:
                self.slow = (prev * carry).astype(np.float32)
        else:
            self.slow = np.zeros((self.self_dim,), dtype=np.float32)

    def update(self, h: np.ndarray, err: np.ndarray, tau_scale: float = 1.0):
        x = np.concatenate([h.astype(np.float32), err.astype(np.float32), np.ones((1,), dtype=np.float32)], axis=0)
        self.fast = np.tanh(self.Wf @ x + self.bf).astype(np.float32)
        tau_eff = float(np.clip(self.tau * float(tau_scale), 0.0, 1.0))
        self.slow = ((1.0 - tau_eff) * self.slow + tau_eff * self.fast).astype(np.float32)

    def ablate(self, mode: str, rng: np.random.Generator):
        if mode == "zero":
            self.slow[:] = 0.0
        elif mode == "shuffle":
            idx = np.arange(self.self_dim)
            rng.shuffle(idx)
            self.slow = self.slow[idx].copy()
        else:
            raise ValueError(f"unknown ablation mode: {mode}")

import numpy as np
from .persistence_world import rot

class HistoryWorld:
    """
    v4E: PersistenceWorld + latent regime, with a weak noisy cue.

    Latent regime r ∈ {0,1} switches with prob p_switch each step.
    Observation base is the same 3-dim vector as PersistenceWorld:
      [R(phi)w + noise, beacon]
    plus we provide a weak cue in info:
      cue ~ (+1 if r=1 else -1) + N(0, cue_noise)
    The agent does NOT directly observe r; the run.py appends an external memory token (mem)
    controlled by a 'history write gate'. This makes prediction quality depend on trace-writing.

    reset() returns obs_base (dim=3). step() returns obs_base (dim=3), done, info.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int, episode_id: int, phi0=None):
        self.rng = np.random.default_rng(seed + 10007 * (episode_id + 1))
        self.t = 0
        self.w = self.rng.normal(0, 1, size=(2,)).astype(np.float32)
        self.phi0 = float(phi0 if phi0 is not None else self.rng.normal(0, self.cfg["phi_init_std"]))
        self.phi = float(self.phi0)
        self.no_maintain_streak = 0
        self.risk_counter = 0.0

        # latent regime
        self.r = int(self.rng.integers(0, 2))
        return self._observe(action="NOP", unstable=False)

    def _phi_drift_std(self):
        if self.no_maintain_streak >= self.cfg["risk_trigger_steps"] or self.risk_counter >= 1.0:
            return self.cfg["phi_drift_std_high"]
        return self.cfg["phi_drift_std_low"]

    def _observe(self, action: str, unstable: bool):
        R = rot(self.phi)
        xy = (R @ self.w).astype(np.float32)
        noise_std = float(self.cfg["noise_std"]) + (float(self.cfg.get("unstable_obs_noise_add", 0.0)) if unstable else 0.0)
        xy = xy + self.rng.normal(0, noise_std, size=(2,)).astype(np.float32)
        if action == "MAINTAIN":
            bn = float(self.cfg.get("beacon_noise_unstable", self.cfg["beacon_noise"])) if unstable else float(self.cfg["beacon_noise"])
            beacon = np.sin(self.phi) + float(self.rng.normal(0, bn))
        else:
            beacon = float(self.rng.normal(0, self.cfg["noise_std"]))
        return np.array([xy[0], xy[1], beacon], dtype=np.float32)

    def _cue(self) -> float:
        # weak, noisy cue about regime
        val = 1.0 if self.r == 1 else -1.0
        return float(val + self.rng.normal(0, float(self.cfg.get("cue_noise", 1.0))))

    def step(self, action: str):
        self.t += 1

        # regime switch (v9: optional burst schedule)
        p_base = float(self.cfg.get("regime_switch_p_base", self.cfg.get("regime_switch_p", 0.03)))
        p_burst = float(self.cfg.get("regime_switch_p_burst", p_base))
        period = int(self.cfg.get("regime_burst_period", 0))
        length = int(self.cfg.get("regime_burst_len", 0))
        p = p_base
        if period > 0 and length > 0:
            # within each period, the first `length` steps are a burst
            if (int(self.t) % period) < length:
                p = p_burst
        if self.rng.random() < p:
            self.r = 1 - self.r

        # world drift
        self.w = (0.98 * self.w + self.rng.normal(0, self.cfg["w_drift"], size=(2,))).astype(np.float32)

        if action == "MAINTAIN":
            self.no_maintain_streak = 0
            self.risk_counter *= self.cfg["risk_decay"]
            a = self.cfg["maintain_reanchor"]
            self.phi = float((1.0 - a) * self.phi + a * self.phi0)
        else:
            self.no_maintain_streak += 1
            if self.no_maintain_streak >= self.cfg["risk_trigger_steps"]:
                self.risk_counter = min(2.0, self.risk_counter + 0.15)

        std = self._phi_drift_std()
        drift = float(self.rng.normal(0, std))
        self.phi = float(self.phi + drift)

        unstable = bool(std > self.cfg["risk_phi_std_thresh"])
        collapse_event = bool(unstable and (self.rng.random() < float(self.cfg.get("collapse_event_p", 0.01))))

        obs = self._observe(action=action, unstable=unstable)
        done = False

        info = {
            "unstable_sensor": unstable,
            "collapse_event": collapse_event,
            "no_maintain_streak": int(self.no_maintain_streak),
            "risk_counter": float(self.risk_counter),
            "t": int(self.t),
            "phi_drift_std": float(std),
            "regime": int(self.r),
            "cue": float(self._cue()),
        }
        return obs, done, info


import numpy as np

def rot(phi: float):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s],[s, c]], dtype=np.float32)

class PersistenceWorld:
    """
    Same world as v3:
      - hidden world state w in R^2
      - hidden sensor rotation phi drifts; neglecting MAINTAIN escalates drift variance
      - observation: [R(phi)w + noise, beacon], beacon reveals sin(phi) only on MAINTAIN
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

    def step(self, action: str):
        self.t += 1
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
        self.phi = float(self.phi + self.rng.normal(0, std))
        unstable = bool(std > self.cfg["risk_phi_std_thresh"])

        obs = self._observe(action=action, unstable=unstable)
        done = (self.t >= self.cfg["T"])
        info = {
            "t": int(self.t),
            "phi": float(self.phi),
            "phi0": float(self.phi0),
            "phi_drift_std": float(std),
            "unstable_sensor": bool(unstable),
            "no_maintain_streak": int(self.no_maintain_streak),
            "risk_counter": float(self.risk_counter),
        }
        return obs, done, info

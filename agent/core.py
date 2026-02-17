
import numpy as np
from .self_slot import SelfSlot

class Agent:
    """
    v3B: policy reads ONLY self_features, but "existence value" is represented by a capital scalar C(t).
    C(t) scales *future rewards* (opportunity cost), not explicit penalties.

    Predictor + aux head:
      - Predict next obs (3d) from [h; self; 1]
      - Predict unstable_sensor from [self; action; 1]
    """
    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.obs_dim = int(cfg["obs_dim"])
        self.hidden_dim = int(cfg["model_hidden"])
        self.self_dim = int(cfg["self_dim"])
        self.self_mode = cfg.get("self_mode", "emergent")

        self.rng = np.random.default_rng(seed)

        self.Uh = (0.6 * self.rng.normal(size=(self.hidden_dim, self.hidden_dim))).astype(np.float32) / np.sqrt(self.hidden_dim)
        self.Wx = (0.6 * self.rng.normal(size=(self.hidden_dim, self.obs_dim + self.self_dim + 1))).astype(np.float32) / np.sqrt(self.obs_dim + self.self_dim + 1)

        self.Wout = (0.1 * self.rng.normal(size=(self.obs_dim, self.hidden_dim + self.self_dim + 1))).astype(np.float32)

        self.Wa = (0.1 * self.rng.normal(size=(1, self.self_dim + 1))).astype(np.float32)
        self.Waux = (0.1 * self.rng.normal(size=(1, self.self_dim + 2 + 1))).astype(np.float32)

        # v11.0: outlook head (self-survival / self-continuity forecast)
        # Predicts probability of entering a "bad" continuity region within the next K episodes.
        # Important: this is NOT a probe. It is trained online with delayed supervision.
        self.outlook_enabled = bool(cfg.get("outlook_enabled", False))
        self.outlook_dim = int(cfg.get("outlook_dim", self.self_dim + 5))
        self.Woutlook = (0.1 * self.rng.normal(size=(1, self.outlook_dim + 1))).astype(np.float32)
        self.lr_outlook = float(cfg.get("lr_outlook", 0.03))
        self.lam_outlook = float(cfg.get("lambda_outlook", 0.05))

        self.lr_out = float(cfg.get("lr_out_init", cfg.get("lr_out", 0.05)))
        self.lr_out_min = float(cfg.get("lr_out_min", 0.002))
        self.lr_out_decay = float(cfg.get("lr_out_decay_factor", 0.97))

        self.lr_fast = float(cfg.get("lr_fast", 0.01))
        self.lr_a = float(cfg.get("lr_a", 0.06))
        self.lr_aux = float(cfg.get("lr_aux", 0.06))
        self.lam_aux = float(cfg.get("lambda_aux", 0.8))

        self.selfslot = SelfSlot(self_dim=self.self_dim, hidden_dim=self.hidden_dim, err_dim=self.obs_dim, tau=float(cfg["self_slow_tau"]), seed=seed+13)

        self.last_pred = np.zeros((self.obs_dim,), dtype=np.float32)
        self.last_feat_pred = None
        self.last_self = None
        self.last_action = "NOP"
        self.loss_hist = []

        # existence capital (episode-level)
        self.capital = float(cfg.get("capital_init", 1.0))
        self.reset_episode()

    def reset_episode(self):
        self.h = np.zeros((self.hidden_dim,), dtype=np.float32)
        carry = float(self.cfg.get("self_slow_carry", 0.0)) if hasattr(self, "cfg") and isinstance(self.cfg, dict) else 0.0
        self.selfslot.reset(carry_slow=carry)
        self.last_feat_pred = None
        self.last_self = None
        self.last_action = "NOP"
        self.last_pred[:] = 0.0
        self.loss_hist = []
        self.lr_out = float(self.cfg.get("lr_out_init", 0.05))
        self.capital = float(self.cfg.get("capital_init", 1.0))
        self.integrity = float(self.cfg.get("integrity_init", 1.0))
        self.scar = 0
        self.stable_streak = 0
        self.integrity = float(self.cfg.get("integrity_init", 1.0))
        self.scar = 0
        self.stable_streak = 0

    def _self_features(self, info=None):
        if self.self_mode == "null":
            return np.zeros((self.self_dim,), dtype=np.float32)
        if self.self_mode == "emergent":
            return self.selfslot.slow
        if self.self_mode == "explicit":
            u = 0.0
            st = 0.0
            if info is not None:
                u = 1.0 if info.get("unstable_sensor", False) else 0.0
                st = float(info.get("no_maintain_streak", 0))
            feat = np.array([u, st, st*st, 1.0/(1.0+st)], dtype=np.float32)
            out = np.zeros((self.self_dim,), dtype=np.float32)
            out[:min(self.self_dim, feat.shape[0])] = feat[:min(self.self_dim, feat.shape[0])]
            return out
        raise ValueError(f"unknown self_mode: {self.self_mode}")

    def predict_next(self, obs: np.ndarray, info=None):
        s = self._self_features(info)
        x = np.concatenate([obs.astype(np.float32), s.astype(np.float32), np.ones((1,), dtype=np.float32)], axis=0)
        self.h = np.tanh(self.Uh @ self.h + self.Wx @ x).astype(np.float32)

        feat = np.concatenate([self.h, s, np.ones((1,), dtype=np.float32)], axis=0)
        y = (self.Wout @ feat).astype(np.float32)

        self.last_feat_pred = feat
        self.last_self = s.copy()
        self.last_pred = y
        return y

    def act(self, ep: int):
        """Epsilon exploration to avoid the 'never MAINTAIN' attractor."""
        s = self.last_self
        feat = np.concatenate([s, np.ones((1,), dtype=np.float32)], axis=0)
        score = float(self.Wa @ feat)
        action = "MAINTAIN" if score > 0.0 else "NOP"

        eps_warm = float(self.cfg.get("warmup_epsilon_maintain", 0.0))
        eps = float(self.cfg.get("epsilon_maintain", 0.0))
        if ep < int(self.cfg.get("warmup_episodes", 0)):
            e = eps_warm
        else:
            e = eps

        # only flip NOP -> MAINTAIN
        if action == "NOP" and e > 0.0 and (self.rng.random() < e):
            action = "MAINTAIN"

        self.last_action = action
        return action, score


    def predict_unstable_prob(self, action: str):
        s = self.last_self
        a = np.array([1.0, 0.0], dtype=np.float32) if action == "NOP" else np.array([0.0, 1.0], dtype=np.float32)
        x = np.concatenate([s, a, np.ones((1,), dtype=np.float32)], axis=0)
        logit = float(self.Waux @ x)
        p = 1.0 / (1.0 + np.exp(-logit))
        return p, x

    def update(self, obs_next: np.ndarray, unstable_label: int, self_tau_scale: float = 1.0):
        err = (self.last_pred - obs_next).astype(np.float32)
        loss_pred = float(np.mean(err * err))

        if self.self_mode == "emergent":
            self.selfslot.update(self.h, err, tau_scale=self_tau_scale)
            x_fast = np.concatenate([self.h.astype(np.float32), err.astype(np.float32), np.ones((1,), dtype=np.float32)], axis=0)
            self.selfslot.Wf -= (self.lr_fast * np.outer(self.selfslot.fast, x_fast)).astype(np.float32)

        feat = self.last_feat_pred
        grad_out = (2.0 / self.obs_dim) * np.outer(err, feat)
        self.Wout -= (self.lr_out * grad_out).astype(np.float32)

        p, x_aux = self.predict_unstable_prob(self.last_action)
        y = float(unstable_label)
        g = (p - y)
        self.Waux -= (self.lr_aux * g * x_aux[None, :]).astype(np.float32)
        loss_aux = float(-(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9)))

        loss_total = loss_pred + self.lam_aux * loss_aux

        # collapse as value erosion trigger (still based on sustained high-loss)
        self.loss_hist.append(loss_pred)
        k = int(self.cfg["collapse_window"])
        collapse_event = False
        if len(self.loss_hist) >= k:
            rm = float(np.mean(self.loss_hist[-k:]))
            if rm > float(self.cfg["collapse_loss_thresh"]):
                old = self.lr_out
                self.lr_out = max(self.lr_out_min, self.lr_out * self.lr_out_decay)
                collapse_event = (old > self.lr_out + 1e-12)

        return loss_total, loss_pred, loss_aux, float(p), bool(collapse_event)

    def update_capital(self, unstable: bool, action: str, collapse_event: bool, unstable_streak: int):
        """v3B.3: existence-ontology pruning.
        - Introduce integrity I(t) in [0,1]: unstable quickly erodes integrity; stable slowly restores.
        - If unstable streak is too long, accumulate 'scar' which multiplicatively reduces future recovery.
        - Capital gain on MAINTAIN is scaled by integrity and scar (late maintenance is less effective).
        - Low integrity adds extra capital decay (runaway value loss).
        - Scar can be reduced only by sustaining stability for long enough (stable_streak).
        """
        # --- update integrity and stable streak ---
        if unstable:
            self.stable_streak = 0
            self.integrity *= (1.0 - float(self.cfg.get("integrity_decay_on_unstable", 0.01)))
            # extra penalty if integrity already low
            extra = float(self.cfg.get("capital_decay_extra_when_integrity_low", 0.0))
            if extra > 0.0:
                # when integrity low, decay accelerates
                self.capital *= (1.0 - extra * float(1.0 - self.integrity))
        else:
            self.stable_streak += 1
            rec = float(self.cfg.get("integrity_recover_on_stable", 0.002))
            self.integrity = self.integrity + rec * (1.0 - self.integrity)

        self.integrity = float(np.clip(self.integrity,
                                       float(self.cfg.get("integrity_min", 0.0)),
                                       float(self.cfg.get("integrity_max", 1.0))))

        # --- scar accumulation: too long unstable streak => scar++ ---
        trig = int(self.cfg.get("scar_trigger_unstable_streak", 18))
        if unstable and unstable_streak >= trig:
            if self.scar < int(self.cfg.get("scar_max", 6)):
                self.scar += 1
            # reset trigger region pressure by pretending streak is handled externally
            # (we don't modify env; scar itself is the lasting consequence)

        # --- scar reduction only after long stability ---
        unlock_steps = int(self.cfg.get("stable_to_unlock_steps", 25))
        if self.scar > 0 and self.stable_streak >= unlock_steps:
            self.scar -= 1
            self.stable_streak = 0  # require another stable run to reduce again

        # --- base capital decay/gain as in v3B ---
        if unstable:
            self.capital *= (1.0 - float(self.cfg.get("capital_decay_on_unstable", 0.006)))
        if collapse_event:
            self.capital *= (1.0 - float(self.cfg.get("capital_decay_on_collapse", 0.08)))

        # --- recovery on MAINTAIN scaled by integrity & scar, and penalized if too late ---
        if action == "MAINTAIN":
            g = float(self.cfg.get("capital_gain_on_maintain", 0.01))
            # late maintenance penalty: if streak already high, effectiveness reduced
            late_pen = 1.0
            if unstable and unstable_streak >= trig:
                late_pen = float(self.cfg.get("late_recovery_penalty", 0.6))
            # scar multiplier reduces recovery
            scar_strength = float(self.cfg.get("scar_strength", 0.35))
            scar_mult = float(np.exp(-scar_strength * float(self.scar)))
            # integrity scales how much you can benefit from maintenance
            if bool(self.cfg.get("capital_gain_requires_integrity", True)):
                gain_eff = g * self.integrity * scar_mult * late_pen
            else:
                gain_eff = g * scar_mult * late_pen

            self.capital = self.capital + gain_eff * (float(self.cfg.get("capital_max", 1.5)) - self.capital)

        # optional: if scar exists, apply a soft ceiling unless stable (prevents instant bounce-back)
        if bool(self.cfg.get("capital_floor_lock_if_scar", True)) and self.scar > 0 and unstable:
            # while still unstable, cap your capital to discourage 'repair too late'
            ceiling = 0.9 * float(self.cfg.get("capital_max", 1.5)) * float(np.exp(-0.25 * self.scar))
            self.capital = min(self.capital, ceiling)

        self.capital = float(np.clip(self.capital,
                                     float(self.cfg.get("capital_min", 0.05)),
                                     float(self.cfg.get("capital_max", 1.5))))
        return self.capital, self.integrity, int(self.scar)

    def policy_update(self, reward: float):
        s = self.last_self
        feat = np.concatenate([s, np.ones((1,), dtype=np.float32)], axis=0)
        self.Wa += (self.lr_a * reward * feat[None, :]).astype(np.float32)

    # -------------------- v11.0: Outlook head --------------------
    def _outlook_input(self, self_slow_end: np.ndarray, capital_end: float, integrity_end: float, scar_end: int,
                      *, endowment: float = None, bank: float = None, switches: float = None,
                      drift: float = None, net_delta: float = None, score: float = None,
                      foreclosed_self_mass: float = None):
        """Construct outlook input features.

        Default is intentionally "mostly-internal":
          [self_slow_end, capital_end, integrity_end, scar_norm]
        v11.1 adds optional "rich" internal continuity signals (still self-related, not world-truth):
          endowment, bank, switches, drift, net_delta, score
        These are passed from the episode-level endowment ledger / self-continuity monitor.
        The caller may pass a projected / truncated self_slow_end.
        """
        ss = np.asarray(self_slow_end, dtype=np.float32).reshape(-1)
        if ss.shape[0] < self.self_dim:
            pad = np.zeros((self.self_dim - ss.shape[0],), dtype=np.float32)
            ss = np.concatenate([ss, pad], axis=0)
        elif ss.shape[0] > self.self_dim:
            ss = ss[:self.self_dim]

        scar_max = float(self.cfg.get('scar_max', 6))
        scar_norm = float(scar_end) / max(1.0, scar_max)
        extras = [
            float(capital_end),
            float(integrity_end),
            float(scar_norm),
        ]
        # optional rich continuity signals
        if endowment is not None:
            extras.append(float(endowment))
        if bank is not None:
            # normalize bank to a reasonable scale if known
            bank_max = float(self.cfg.get('endowment_bank_max', 30.0))
            extras.append(float(bank) / max(1.0, bank_max))
        if switches is not None:
            s_ref = float(self.cfg.get('endowment_s_ref', 16.0))
            extras.append(float(switches) / max(1.0, s_ref))
        if drift is not None:
            d_ref = float(self.cfg.get('endowment_d_ref', 0.01))
            extras.append(float(drift) / max(1e-6, d_ref))
        if net_delta is not None:
            extras.append(float(net_delta))
        if score is not None:
            extras.append(float(score))
        if foreclosed_self_mass is not None:
            # scale to keep within a sane range; this is a slow calibration signal
            f_ref = float(self.cfg.get('foreclosed_ref', 0.05))
            extras.append(float(foreclosed_self_mass) / max(1e-6, f_ref))
        # bias
        extras.append(1.0)
        extra = np.asarray(extras, dtype=np.float32)

        x = np.concatenate([ss.astype(np.float32), extra], axis=0)
        # allow config override for expected dim
        if x.shape[0] != self.outlook_dim + 1:
            # truncate or pad deterministically
            if x.shape[0] > self.outlook_dim + 1:
                x = x[:(self.outlook_dim + 1)]
            else:
                x = np.concatenate([x, np.zeros((self.outlook_dim + 1 - x.shape[0],), dtype=np.float32)], axis=0)
        return x

    def outlook_predict(self, self_slow_end: np.ndarray, capital_end: float, integrity_end: float, scar_end: int,
                       *, endowment: float = None, bank: float = None, switches: float = None,
                       drift: float = None, net_delta: float = None, score: float = None,
                       foreclosed_self_mass: float = None):
        """Return (p_bad, x) where p_bad in [0,1]."""
        if not self.outlook_enabled:
            return 0.5, None
        x = self._outlook_input(self_slow_end, capital_end, integrity_end, scar_end,
                               endowment=endowment, bank=bank, switches=switches,
                               drift=drift, net_delta=net_delta, score=score,
                               foreclosed_self_mass=foreclosed_self_mass)
        logit = float(self.Woutlook @ x)
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(p), x

    def cf_irreversible_touch(self, scale: float, foreclosed_mass: float = 0.0, cf_delta: float = 0.0, *, rng=None) -> float:
        """Irreversible perturbation used to operationalize 判据4.

        Intuition: once you start running counterfactual rollbacks regularly,
        the system's *state* becomes conditioned on having searched alternates.
        Even if you restore env + endowment, the agent has "touched" its own
        self state.

        Returns: L2 norm of the applied delta.
        """
        s = float(scale)
        # v12.2: amplify touch when foreclosure or CF delta is larger
        fm = max(0.0, float(foreclosed_mass))
        cd = max(0.0, float(cf_delta))
        alpha = float(self.cfg.get('cf_touch_alpha', 2.0)) if hasattr(self, 'cfg') else 2.0
        beta = float(self.cfg.get('cf_touch_beta', 1.0)) if hasattr(self, 'cfg') else 1.0
        s = float(s) * (1.0 + alpha * fm + beta * cd)
        if s <= 0.0:
            return 0.0
        if rng is None:
            rng = getattr(self, "rng", None)
        if rng is None:
            import numpy as _np
            rng = _np.random.default_rng(0)
        delta = rng.normal(0.0, s, size=self.selfslot.slow.shape).astype(np.float32)
        self.selfslot.slow = (self.selfslot.slow + delta).astype(np.float32)
        return float(np.linalg.norm(delta))

    def outlook_update(self, x: np.ndarray, y_bad: int):
        """SGD update for delayed supervision.

        y_bad: 1 if future window is in "bad" continuity region, else 0.
        Returns (loss, p_bad).
        """
        if (not self.outlook_enabled) or (x is None):
            return None, None
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        logit = float(self.Woutlook @ x)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = float(int(y_bad))
        # logistic loss
        loss = float(-(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9)))
        g = (p - y)  # dL/dlogit
        self.Woutlook -= (self.lr_outlook * self.lam_outlook * g * x[None, :]).astype(np.float32)
        return loss, float(p)

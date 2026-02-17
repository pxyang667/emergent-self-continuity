from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict


@dataclass
class EndowmentTracker:
    """Cross-episode endowment (external to the agent).

    v10.3 design intent:
      - no self-reference, no explicit 'I am'
      - continuity creates a *future advantage* via a small, hard-capped recoverable endowment
      - recovery is *soft* (continuous), not a brittle binary threshold
      - safety: hard min/max, per-episode cap, total budget
    """

    enabled: bool = True

    # Value bounds
    init: float = 1.0
    min_value: float = 0.4
    max_value: float = 0.55

    # v10.3: soft continuity score parameters
    s_ref: float = 12.0
    s_scale: float = 6.0
    d_ref: float = 0.008
    d_scale: float = 0.004

    # v10.3: decay uses score (soft)
    decay_alpha: float = 0.0015

    # v10.3: continuity bank (soft streak)
    bank_lambda: float = 0.98
    bank_th: float = 6.0
    bank_max: float = 30.0

    # v10.3: recovery (hard-capped, budgeted, rate-limited)
    recover_beta: float = 0.002
    recover_cap_per_ep: float = 0.003
    recover_budget: float = 0.05
    # v10.4: very small per-episode budget regeneration (prevents one-shot depletion)
    recover_budget_regen_per_ep: float = 0.0

    # v11.3: Outlook-coupling (worldline entanglement)
    # If enabled, endowment is nudged each episode based on an EMA of predicted survival.
    # This makes the dynamics depend on the learned outlook head, weakening rollback.
    outlook_enable: bool = False
    outlook_ema_beta: float = 0.05
    outlook_target_p_survive: float = 0.97
    outlook_kappa: float = 0.002

    # v11.4: foreclosure mass (worldline foreclosure signal)
    # Tracks how much cross-episode state moved during counterfactual attempts.
    # It is intended to quantify "rollback is no longer clean".
    foreclosed_gain: float = 5.0
    foreclosed_decay: float = 0.995

    # v11.5: counterfactual leaves an irreversible trace (criterion-4 driver)
    # Even when a CF is not "recorded", we add a tiny minimum trace.
    foreclosed_trace_min: float = 1e-4
    foreclosed_trace_scale: float = 0.05
    foreclosed_record_multiplier: float = 10.0
    foreclosed_endowment_kappa: float = 0.002
    def __post_init__(self):
        self.value = float(self.init)
        self.value = max(float(self.min_value), min(float(self.max_value), float(self.value)))

        self.bank = 0.0
        self.recovered_total = 0.0
        self.recovery_events = 0
        self.outlook_ema_p_survive = None

        # foreclosure signal state
        self.foreclosed_self_mass = 0.0
        # irreversible trace stats (diagnostics)
        self.cf_trace_count = 0
        self.cf_trace_total = 0.0
        self.last_cf_delta = 0.0
        self.last_cf_recorded = False

    def snapshot(self) -> Dict[str, float]:
        """Snapshot minimal cross-episode state so a counterfactual episode can be rolled back cleanly."""
        return {
            "value": float(self.value),
            "bank": float(self.bank),
            "recovered_total": float(self.recovered_total),
            "recovery_events": float(self.recovery_events),
            "outlook_ema_p_survive": float(self.outlook_ema_p_survive) if self.outlook_ema_p_survive is not None else float('nan'),
            "foreclosed_self_mass": float(self.foreclosed_self_mass),
        }

    def restore(self, snap: Dict[str, float]) -> None:
        """Restore a snapshot produced by `snapshot()`.

        Only restores state that affects the environment or outlook inputs.
        Does not reset bookkeeping like last_cf_*.
        """
        self.value = float(snap.get("value", self.value))
        self.bank = float(snap.get("bank", self.bank))
        self.recovered_total = float(snap.get("recovered_total", self.recovered_total))
        self.recovery_events = int(snap.get("recovery_events", self.recovery_events))
        ema = snap.get("outlook_ema_p_survive", float('nan'))
        self.outlook_ema_p_survive = None if (ema != ema) else float(ema)
        # v11.5: do NOT restore foreclosed_self_mass.
        # This is the whole point of criterion-4 pressure: once you attempt a
        # counterfactual, the worldline has been touched and cannot be made
        # perfectly identical by rollback.
        # (We still snapshot it for logging/diagnostics.)

    def _score(self, switches: int, drift: float) -> float:
        sw = max(0.0, float(int(switches)))
        d = max(0.0, float(drift))

        # soft penalties beyond reference thresholds
        s_ref = max(0.0, float(self.s_ref))
        s_scale = max(1e-6, float(self.s_scale))
        d_ref = max(0.0, float(self.d_ref))
        d_scale = max(1e-9, float(self.d_scale))

        s = exp(-max(0.0, sw - s_ref) / s_scale)
        dd = exp(-max(0.0, d - d_ref) / d_scale)
        score = max(0.0, min(1.0, float(s * dd)))
        return score

    
    def note_counterfactual(
        self,
        endowment_before: float,
        endowment_after: float,
        bank_before: float | None = None,
        bank_after: float | None = None,
        record: bool = True,
    ):
        """Update a running 'foreclosed_self_mass' signal.

        Intuition: if attempting a counterfactual is no longer a 'clean rollback' because
        cross-episode state (endowment/bank) has already advanced, the worldline becomes
        path-dependent. We capture that as a slow-varying scalar that can be fed into the
        outlook head for calibration (not directly into action selection).
        """
        try:
            d_endow = abs(float(endowment_after) - float(endowment_before))
        except Exception:
            d_endow = 0.0
        d_bank = 0.0
        if (bank_before is not None) and (bank_after is not None):
            try:
                d_bank = abs(float(bank_after) - float(bank_before))
            except Exception:
                d_bank = 0.0
        delta = d_endow + 0.01 * d_bank

        # v11.5: attempting a counterfactual is *never* perfectly clean.
        # We always add a tiny minimum trace, and add more trace when "record".
        trace = float(self.foreclosed_trace_min) + float(self.foreclosed_trace_scale) * float(delta)
        if record:
            trace *= float(self.foreclosed_record_multiplier)

        self.last_cf_delta = float(delta)
        self.last_cf_recorded = bool(record)
        self.cf_trace_count += 1
        self.cf_trace_total += float(trace)

        self.foreclosed_self_mass = float(self.foreclosed_self_mass) * float(self.foreclosed_decay) + float(trace)
        return {
            'foreclosed_delta': float(delta),
            'foreclosed_self_mass': float(self.foreclosed_self_mass),
            'cf_trace_count': int(self.cf_trace_count),
            'cf_trace_total': float(self.cf_trace_total),
            'd_endow': float(d_endow),
            'd_bank': float(d_bank),
            'foreclosed_recorded': bool(record),
        }

    def update(self, switches: int, drift: float, p_survive: float | None = None) -> dict:
        """Update endowment using continuity signals.

        Returns jsonable dict with:
          - endowment
          - score
          - bank
          - decay
          - recovered
          - net_delta
          - triggered
          - budget_left
          - outlook_p_survive
          - outlook_ema_p_survive
          - outlook_delta
        """
        if not self.enabled:
            return {
                "endowment": float(self.value),
                "score": 0.0,
                "bank": float(self.bank),
                "decay": 0.0,
                "recovered": 0.0,
                "net_delta": 0.0,
                "triggered": False,
                "budget_left": float(max(0.0, float(self.recover_budget) - float(self.recovered_total))),
            }

        # v10.4: regenerate recovery budget slowly (implemented by reducing recovered_total)
        regen = max(0.0, float(self.recover_budget_regen_per_ep))
        if regen > 0.0 and self.recovered_total > 0.0:
            self.recovered_total = max(0.0, float(self.recovered_total) - float(regen))

        score = self._score(switches, drift)

        # --- soft decay: stable episodes decay ~0; unstable episodes decay more ---
        decay = float(self.decay_alpha) * (1.0 - float(score))
        decay = max(0.0, min(0.05, decay))  # safety clip
        prev = float(self.value)
        self.value = float(self.value) * (1.0 - float(decay))

        # --- continuity bank (soft streak memory) ---
        lam = max(0.0, min(1.0, float(self.bank_lambda)))
        self.bank = float(lam) * float(self.bank) + float(score)
        self.bank = max(0.0, min(float(self.bank_max), float(self.bank)))

        # --- recovery ---
        recovered = 0.0
        budget_left = max(0.0, float(self.recover_budget) - float(self.recovered_total))
        triggered = float(self.bank) > float(self.bank_th)
        if triggered and (budget_left > 0.0) and (float(self.value) < float(self.max_value)):
            raw = float(self.recover_beta) * max(0.0, float(self.bank) - float(self.bank_th))
            step = min(raw, float(self.recover_cap_per_ep), float(budget_left), float(self.max_value) - float(self.value))
            if step > 0.0:
                self.value = float(self.value) + float(step)
                self.recovered_total = float(self.recovered_total) + float(step)
                self.recovery_events += 1
                recovered = float(step)
                budget_left = max(0.0, float(self.recover_budget) - float(self.recovered_total))

        # --- v11.3: outlook-coupling (tiny, but stateful) ---
        outlook_delta = 0.0
        p_in = None
        if self.outlook_enable and (p_survive is not None):
            try:
                p_in = float(p_survive)
            except Exception:
                p_in = None
        if self.outlook_enable and (p_in is not None):
            p_in = max(0.0, min(1.0, float(p_in)))
            beta = max(0.0, min(1.0, float(self.outlook_ema_beta)))
            if self.outlook_ema_p_survive is None:
                self.outlook_ema_p_survive = float(p_in)
            else:
                self.outlook_ema_p_survive = (1.0 - beta) * float(self.outlook_ema_p_survive) + beta * float(p_in)

            target = max(0.0, min(1.0, float(self.outlook_target_p_survive)))
            kappa = float(self.outlook_kappa)
            # signed nudge: if predicted survival < target => slightly reduce endowment; else slightly increase
            outlook_delta = float(kappa) * (float(self.outlook_ema_p_survive) - float(target))
            # safety: hard clip to keep this coupling gentle
            outlook_delta = max(-0.01, min(0.01, float(outlook_delta)))
            self.value = float(self.value) + float(outlook_delta)

        # v11.5: foreclosure penalty (criterion-4 driver)
        # Running counterfactuals leaves an irreversible trace (foreclosed_self_mass).
        # As it grows, it slightly drags down endowment, making "clean rollback" impossible.
        if float(self.foreclosed_self_mass) > 0.0:
            fc_pen = -float(self.foreclosed_endowment_kappa) * float(self.foreclosed_self_mass)
            fc_pen = max(-0.01, min(0.0, float(fc_pen)))
            self.value = float(self.value) + float(fc_pen)

        # --- clip ---
        self.value = max(float(self.min_value), min(float(self.max_value), float(self.value)))

        net_delta = float(self.value) - float(prev)
        return {
            "endowment": float(self.value),
            "score": float(score),
            "bank": float(self.bank),
            "decay": float(decay),
            "recovered": float(recovered),
            "net_delta": float(net_delta),
            "triggered": bool(triggered),
            "budget_left": float(budget_left),
            "outlook_p_survive": (None if p_in is None else float(p_in)),
            "outlook_ema_p_survive": (None if self.outlook_ema_p_survive is None else float(self.outlook_ema_p_survive)),
            "outlook_delta": float(outlook_delta),
            "foreclosed_self_mass_end": float(self.foreclosed_self_mass),
            "cf_trace_count": int(self.cf_trace_count),
            "cf_trace_total": float(self.cf_trace_total),
        }

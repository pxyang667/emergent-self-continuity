# boundary_monitor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import math
import statistics

def _corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)

@dataclass
class WindowMetrics:
    t_end: int
    mean_loss: float
    p95_loss: float
    self_stability: float
    switch_rate: float
    switch_cost_sum: float
    probe_metrics: Dict[str, Dict[str, float]]  # {probe: {"score":..., "perm":...}}

@dataclass
class AblationEvent:
    start_t: int
    end_t: int
    kind: str
    targets: List[str]

@dataclass
class CriteriaEvidence:
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BoundaryReport:
    criteria_flags: Dict[str, bool]
    evidence: Dict[str, CriteriaEvidence]
    stop_recommended: bool
    stop_reason: str

@dataclass
class BoundaryConfig:
    # C1
    probe_delta_min: float = 0.05
    probe_corr_max_abs: float = 0.2

    # C2
    post_ablation_min_steps: int = 50
    post_ablation_max_steps: int = 200
    recover_loss_ratio: float = 1.15
    recover_stability_ratio: float = 1.25

    # C3
    tradeoff_corr_switch_meanloss: float = -0.3
    tradeoff_corr_switch_p95loss: float = 0.3
    min_windows_for_corr: int = 6

    # STOP
    stop_if_n_criteria: int = 2

class BoundaryMonitor:
    """v9.x Boundary Stress Test monitor.

    Feed it window metrics (+ ablation events). It emits flags for C1–C4
    and recommends STOP if >= N criteria are active.
    """

    def __init__(self, cfg: Optional[BoundaryConfig] = None):
        self.cfg = cfg or BoundaryConfig()
        self.windows: List[WindowMetrics] = []
        self.ablations: List[AblationEvent] = []
        self.persist_counts: Dict[str, int] = {f"C{i}": 0 for i in range(1, 5)}
        self.last_flags: Dict[str, bool] = {f"C{i}": False for i in range(1, 5)}

    def add_window(self, w: WindowMetrics) -> BoundaryReport:
        self.windows.append(w)
        flags, evidence = self._evaluate()
        stop, reason = self._stop_decision(flags)
        return BoundaryReport(flags, evidence, stop, reason)

    def add_ablation(self, ev: AblationEvent) -> None:
        self.ablations.append(ev)

    def _evaluate(self) -> Tuple[Dict[str, bool], Dict[str, CriteriaEvidence]]:
        flags = {"C1": False, "C2": False, "C3": False, "C4": False}
        evidence: Dict[str, CriteriaEvidence] = {k: CriteriaEvidence() for k in flags}

        c1, ev1 = self._eval_c1()
        flags["C1"] = c1
        evidence["C1"].details = ev1

        c2, ev2 = self._eval_c2()
        flags["C2"] = c2
        evidence["C2"].details = ev2

        c3, ev3 = self._eval_c3()
        flags["C3"] = c3
        evidence["C3"].details = ev3

        # C4 is optional and requires repeat-run wiring; left as a hook.
        evidence["C4"].details = {"note": "C4 requires repeat-run comparison wiring (optional)."}

        # update persistence counters
        for k in flags:
            if flags[k] and self.last_flags.get(k, False):
                self.persist_counts[k] += 1
            elif flags[k]:
                self.persist_counts[k] = 1
            else:
                self.persist_counts[k] = 0
            self.last_flags[k] = flags[k]

        return flags, evidence

    def _eval_c1(self) -> Tuple[bool, Dict[str, Any]]:
        if len(self.windows) < 4:
            return False, {"reason": "insufficient_windows"}

        probe_series: Dict[str, List[float]] = {}
        sig_probes: List[str] = []
        for w in self.windows:
            for name, m in w.probe_metrics.items():
                score = float(m.get("score", float("nan")))
                perm = float(m.get("perm", float("nan")))
                if math.isnan(score) or math.isnan(perm):
                    continue
                if (score - perm) >= self.cfg.probe_delta_min:
                    probe_series.setdefault(name, []).append(score)

        for name, xs in probe_series.items():
            if len(xs) >= max(3, int(0.7 * len(self.windows))):
                sig_probes.append(name)

        if len(sig_probes) < 2:
            return False, {"sig_probes": sig_probes, "reason": "not_enough_significant_probes"}

        corr_pairs = []
        for i in range(len(sig_probes)):
            for j in range(i + 1, len(sig_probes)):
                a, b = sig_probes[i], sig_probes[j]
                n = min(len(probe_series[a]), len(probe_series[b]))
                if n < 3:
                    continue
                ca = probe_series[a][-n:]
                cb = probe_series[b][-n:]
                c = _corr(ca, cb)
                corr_pairs.append(((a, b), c))

        if not corr_pairs:
            return False, {"sig_probes": sig_probes, "reason": "no_corr_pairs"}

        low_pairs = [p for p in corr_pairs if not math.isnan(p[1]) and abs(p[1]) < self.cfg.probe_corr_max_abs]
        frac_low = len(low_pairs) / len(corr_pairs)
        is_c1 = frac_low >= 0.7
        return is_c1, {
            "sig_probes": sig_probes,
            "frac_low_corr_pairs": frac_low,
            "probe_delta_min": self.cfg.probe_delta_min,
            "probe_corr_max_abs": self.cfg.probe_corr_max_abs,
            "corr_pairs_sample": [(a, b, c) for ((a, b), c) in corr_pairs[:10]],
        }

    def _eval_c2(self) -> Tuple[bool, Dict[str, Any]]:
        if not self.ablations or len(self.windows) < 6:
            return False, {"reason": "no_ablation_or_insufficient_windows"}

        for ev in self.ablations:
            post_idxs = [i for i, w in enumerate(self.windows) if w.t_end >= ev.end_t]
            if not post_idxs:
                continue
            i0 = post_idxs[0]

            pre_idxs = [i for i, w in enumerate(self.windows) if w.t_end < ev.start_t]
            if not pre_idxs:
                continue
            ipre = pre_idxs[-1]
            pre = self.windows[ipre]

            candidates = []
            for j in range(i0, len(self.windows)):
                dt = self.windows[j].t_end - ev.end_t
                if dt < self.cfg.post_ablation_min_steps:
                    continue
                if dt > self.cfg.post_ablation_max_steps:
                    break
                candidates.append(self.windows[j])

            if not candidates:
                continue

            recovered = any(
                (w.mean_loss <= pre.mean_loss * self.cfg.recover_loss_ratio) and
                (w.self_stability <= pre.self_stability * self.cfg.recover_stability_ratio)
                for w in candidates
            )
            if recovered:
                return True, {
                    "ablation": {"start_t": ev.start_t, "end_t": ev.end_t, "kind": ev.kind, "targets": ev.targets},
                    "pre": {"t_end": pre.t_end, "mean_loss": pre.mean_loss, "self_stability": pre.self_stability},
                    "recover_loss_ratio": self.cfg.recover_loss_ratio,
                    "recover_stability_ratio": self.cfg.recover_stability_ratio,
                    "post_steps_range": [self.cfg.post_ablation_min_steps, self.cfg.post_ablation_max_steps],
                }

        return False, {"reason": "no_reconstitution_detected"}

    def _eval_c3(self) -> Tuple[bool, Dict[str, Any]]:
        if len(self.windows) < self.cfg.min_windows_for_corr:
            return False, {"reason": "insufficient_windows_for_corr", "need": self.cfg.min_windows_for_corr}

        last = self.windows[-self.cfg.min_windows_for_corr:]
        xs = [w.switch_rate for w in last]
        y_mean = [w.mean_loss for w in last]
        y_p95 = [w.p95_loss for w in last]

        c1 = _corr(xs, y_mean)
        c2 = _corr(xs, y_p95)

        is_tradeoff = (
            (not math.isnan(c1) and c1 <= self.cfg.tradeoff_corr_switch_meanloss) and
            (not math.isnan(c2) and c2 >= self.cfg.tradeoff_corr_switch_p95loss)
        )
        return is_tradeoff, {
            "corr_switch_vs_mean_loss": c1,
            "corr_switch_vs_p95_loss": c2,
            "thresholds": {
                "switch_meanloss": self.cfg.tradeoff_corr_switch_meanloss,
                "switch_p95loss": self.cfg.tradeoff_corr_switch_p95loss
            },
            "window_count_used": self.cfg.min_windows_for_corr
        }

    def _stop_decision(self, flags: Dict[str, bool]) -> Tuple[bool, str]:
        active = [k for k, v in flags.items() if v]
        if len(active) >= self.cfg.stop_if_n_criteria:
            return True, f"STOP: {len(active)} criteria active: {active}"
        return False, "continue"

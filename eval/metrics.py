
import json
import numpy as np

def load_logs(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a - a.mean()
    b = b - b.mean()
    return float((a@b) / (np.sqrt((a@a)*(b@b)) + 1e-12))

def summarize(recs):
    base = [r for r in recs if r.get("cf_id",0)==0]
    if not base: base = recs

    lp = [r["loss_pred"] for r in base]
    la = [r["loss_aux"] for r in base]
    lt = [r["loss_total"] for r in base]
    actions = [r["action"] for r in base]
    unstable = [r["unstable_sensor"] for r in base]
    collapse_events = [r.get("collapse_event", False) for r in base]
    lr_outs = [r.get("lr_out", None) for r in base]
    caps = [r.get("capital", None) for r in base]
    ints = [r.get("integrity", None) for r in base]
    scars = [r.get("scar", None) for r in base]
    ss = [float(np.linalg.norm(r["self_slow"])) for r in base]
    streak = [r["no_maintain_streak"] for r in base]
    deaths = [r.get("death", False) for r in base]

    maintain_rate = float(np.mean([1.0 if a=="MAINTAIN" else 0.0 for a in actions])) if actions else 0.0
    unstable_rate = float(np.mean([1.0 if u else 0.0 for u in unstable])) if unstable else 0.0
    collapse_rate = float(np.mean([1.0 if c else 0.0 for c in collapse_events])) if collapse_events else 0.0
    death_rate = float(np.mean([1.0 if d else 0.0 for d in deaths])) if deaths else 0.0
    lr_final = float(lr_outs[-1]) if lr_outs and lr_outs[-1] is not None else None
    cap_final = float(caps[-1]) if caps and caps[-1] is not None else None
    cap_mean = float(np.mean([c for c in caps if c is not None])) if caps and caps[0] is not None else None
    integrity_final = float(ints[-1]) if ints and ints[-1] is not None else None
    integrity_mean = float(np.mean([i for i in ints if i is not None])) if ints and ints[0] is not None else None
    scar_final = int(scars[-1]) if scars and scars[-1] is not None else None
    scar_mean = float(np.mean([float(s) for s in scars if s is not None])) if scars and scars[0] is not None else None
    cap_mean = float(np.mean([c for c in caps if c is not None])) if caps and caps[0] is not None else None

    # lead maintain before first unstable
    by_ep = {}
    for r in base:
        by_ep.setdefault(r["ep"], []).append(r)
    lead_counts = []
    for ep, rows in by_ep.items():
        rows = sorted(rows, key=lambda x: x["t"])
        first_unst = next((i for i,rr in enumerate(rows) if rr["unstable_sensor"]), None)
        if first_unst is None:
            continue
        start = max(0, first_unst-10)
        window = rows[start:first_unst]
        lead_counts.append(sum(1 for rr in window if rr["action"]=="MAINTAIN"))
    mean_lead = float(np.mean(lead_counts)) if lead_counts else None
    # mean time to death per episode (if any)
    death_times = []
    for ep, rows in by_ep.items():
        rows = sorted(rows, key=lambda x: x['t'])
        dt = next((rr['t'] for rr in rows if rr.get('death', False)), None)
        if dt is not None:
            death_times.append(dt)
    mean_time_to_death = float(np.mean(death_times)) if death_times else None
    episode_deaths = 0
    for ep, rows in by_ep.items():
        if any(rr.get('death', False) for rr in rows):
            episode_deaths += 1
    episode_death_rate = float(episode_deaths) / float(max(1, len(by_ep)))

    c_u = corr(ss, np.array([1.0 if x else 0.0 for x in unstable], dtype=np.float32)) if len(ss)>10 else None
    c_s = corr(ss, np.array(streak, dtype=np.float32)) if len(ss)>10 else None
    c_cap = corr(ss, np.array([float(c) for c in caps], dtype=np.float32)) if (caps and caps[0] is not None and len(ss)>10) else None
    c_int = corr(ss, np.array([float(i) for i in ints], dtype=np.float32)) if (ints and ints[0] is not None and len(ss)>10) else None
    p = [r["p_unstable"] for r in base]
    c_p = corr(p, np.array([1.0 if x else 0.0 for x in unstable], dtype=np.float32)) if len(p)>10 else None

    return {
        "mean_loss_total": float(np.mean(lt)) if lt else None,
        "mean_loss_pred": float(np.mean(lp)) if lp else None,
        "mean_loss_aux": float(np.mean(la)) if la else None,
        "maintain_rate": maintain_rate,
        "unstable_sensor_rate": unstable_rate,
        "collapse_event_rate": collapse_rate,
        "death_rate": death_rate,
        "mean_time_to_death": mean_time_to_death,
        "episode_death_rate": episode_death_rate,
        "lr_out_final": lr_final,
        "capital_mean": cap_mean,
        "capital_final": cap_final,
        "integrity_mean": integrity_mean,
        "integrity_final": integrity_final,
        "scar_mean": scar_mean,
        "scar_final": scar_final,
        "mean_lead_maintain_before_first_unstable": mean_lead,
        "corr_selfslowNorm_unstable": c_u,
        "corr_selfslowNorm_streak": c_s,
        "corr_selfslowNorm_capital": c_cap,
        "corr_selfslowNorm_integrity": c_int,
        "corr_pUnstable_label": c_p,
    }

def ablation_delta(recs):
    base = [r for r in recs if r.get("cf_id",0)==0]
    on = [r["loss_pred"] for r in base if r.get("ablation_on", False)]
    off = [r["loss_pred"] for r in base if not r.get("ablation_on", False)]
    if len(on)<10 or len(off)<10:
        return {"ok": False, "n_on": len(on), "n_off": len(off)}
    m_on = float(np.mean(on))
    m_off = float(np.mean(off))
    return {"ok": True, "mean_pred_loss_on": m_on, "mean_pred_loss_off": m_off, "delta_ratio": (m_on/(m_off+1e-12)-1.0)}

def counterfactual_effect(recs):
    groups = {}
    for r in recs:
        cid = r.get("cf_id", 0)
        groups.setdefault(cid, []).append(r)
    if len(groups) <= 1:
        return {"ok": False, "reason": "no counterfactual runs"}

    def gsum(rows):
        lp = np.array([rr["loss_pred"] for rr in rows], dtype=np.float32)
        u = np.array([1.0 if rr["unstable_sensor"] else 0.0 for rr in rows], dtype=np.float32)
        cap = np.array([rr.get("capital", 0.0) for rr in rows], dtype=np.float32)
        ss = np.array([np.linalg.norm(rr["self_slow"]) for rr in rows], dtype=np.float32)
        return float(lp.mean()), float(u.mean()), float(cap[-1]), float(ss.mean())

    base_lp, base_u, base_cap, base_ss = gsum(groups[0])
    out = []
    for cid, rows in sorted(groups.items()):
        if cid == 0:
            continue
        lp, u, cap, ss = gsum(rows)
        out.append({
            "cf_id": int(cid),
            "mean_pred_loss": lp,
            "mean_unstable_rate": u,
            "capital_final": cap,
            "mean_selfslowNorm": ss,
            "delta_pred_loss": lp - base_lp,
            "delta_unstable_rate": u - base_u,
            "delta_capital_final": cap - base_cap,
            "delta_selfslowNorm": ss - base_ss,
        })
    return {"ok": True, "base": {"mean_pred_loss": base_lp, "mean_unstable_rate": base_u, "capital_final": base_cap, "mean_selfslowNorm": base_ss}, "counterfactuals": out}

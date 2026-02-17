import os, json, time
import sys, argparse
from pathlib import Path
import numpy as np
from persistent_endowment import EndowmentTracker

from envs.history_world import HistoryWorld
from agent.core import Agent
from eval.metrics import load_logs, summarize, ablation_delta, counterfactual_effect
from boundary_monitor import BoundaryMonitor, WindowMetrics, AblationEvent

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rolling_mean(xs, k):
    if len(xs) < k:
        return None
    return float(np.mean(xs[-k:]))


def auc_from_scores(y_true, scores):
    """Compute ROC AUC for binary labels using a rank-based formula. Returns nan if degenerate."""
    import numpy as np
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float32)
    if y.size < 4:
        return float('nan')
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum()); n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # ranks with tie handling via argsort + average ranks on ties
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float32)
    # tie correction: average ranks for equal scores
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for ui, c in enumerate(counts):
            if c <= 1: 
                continue
            idxs = np.where(inv == ui)[0]
            ranks[idxs] = float(ranks[idxs].mean())
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def make_fixed_projection(in_dim: int, out_dim: int, seed: int = 1337):
    """Fixed random projection matrix for compact logging of self_slow.
    Uses N(0, 1/sqrt(in_dim)) so magnitude stays stable across dims."""
    in_dim = int(max(1, in_dim))
    out_dim = int(max(1, out_dim))
    rng = np.random.default_rng(int(seed))
    scale = 1.0 / float(np.sqrt(in_dim))
    return rng.normal(loc=0.0, scale=scale, size=(out_dim, in_dim)).astype(np.float32)

def ridge_probe_r2(X, y, l2=1e-2, seed=0):
    """Quick ridge probe with a fixed train/test split."""
    import numpy as np
    n = X.shape[0]
    if n < 10:
        return float('nan')
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(0.7 * n)
    tr, te = idx[:k], idx[k:]
    Xtr = X[tr]; ytr = y[tr]
    Xte = X[te]; yte = y[te]
    # standardize
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    # add bias
    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0],1), dtype=np.float32)], axis=1)
    Xte1 = np.concatenate([Xte, np.ones((Xte.shape[0],1), dtype=np.float32)], axis=1)
    XtX = Xtr1.T @ Xtr1
    w = np.linalg.solve(XtX + l2 * np.eye(XtX.shape[0], dtype=np.float32), Xtr1.T @ ytr)
    yhat = Xte1 @ w
    ss_res = float(np.sum((yte - yhat)**2))
    ss_tot = float(np.sum((yte - yte.mean())**2)) + 1e-12
    return float(1.0 - ss_res/ss_tot)

def linear_score_probe_auc(X, y, seed=0):
    """Linear score via ridge regression, evaluate AUC."""
    import numpy as np
    n = X.shape[0]
    if n < 10:
        return float('nan')
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(0.7 * n)
    tr, te = idx[:k], idx[k:]
    Xtr = X[tr]; ytr = y[tr]
    Xte = X[te]; yte = y[te]
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0],1), dtype=np.float32)], axis=1)
    Xte1 = np.concatenate([Xte, np.ones((Xte.shape[0],1), dtype=np.float32)], axis=1)
    XtX = Xtr1.T @ Xtr1
    w = np.linalg.solve(XtX + 1e-2 * np.eye(XtX.shape[0], dtype=np.float32), Xtr1.T @ ytr.astype(np.float32))
    scores = Xte1 @ w
    return auc_from_scores(yte, scores)



def with_mem(obs_base: np.ndarray, mem: float) -> np.ndarray:
    return np.concatenate([obs_base.astype(np.float32), np.array([mem], dtype=np.float32)], axis=0)



def compress_history(err_world: "np.ndarray", mode: str = "sign_mean", buckets: int = 5, proj_w=None) -> float:
    """Map a world-prediction residual to a scalar, 'compressible' history token in [-1, 1].

    Modes:
      - sign_mean: sign(mean(err_world)) in {-1, 0, +1}
      - bucket_l2: bucketize ||err_world|| into [-1, +1]
      - proj_sign: sign(dot(err_world, proj_w)) with fixed projection
    """
    import numpy as np
    if err_world is None or getattr(err_world, 'size', 0) == 0:
        return 0.0
    m = (mode or 'sign_mean').lower()
    if m == 'sign_mean':
        v = float(np.mean(err_world))
        return 0.0 if abs(v) < 1e-9 else (1.0 if v > 0.0 else -1.0)
    if m == 'bucket_l2':
        l2 = float(np.sqrt(np.mean(err_world * err_world)))
        scale = 1.0 / (1.0 + l2)  # (0,1]
        x = 1.0 - scale           # [0,1)
        b = int(np.clip(np.floor(x * buckets), 0, buckets - 1))
        c = (b + 0.5) / float(buckets)  # (0,1)
        return float(2.0 * c - 1.0)
    if m == 'proj_sign':
        if proj_w is None:
            proj_w = np.linspace(-1.0, 1.0, num=err_world.size, dtype=np.float32)
        v = float(np.dot(err_world.astype(np.float32), np.asarray(proj_w, dtype=np.float32)))
        return 0.0 if abs(v) < 1e-9 else (1.0 if v > 0.0 else -1.0)
    raise ValueError(f'unknown mem_write_mode: {mode}')


def self_id_bucket(vec: "np.ndarray", kbits: int = 8) -> int:
    """Map a self vector to a small discrete identity bucket.

    This is intentionally simple and local: take the sign pattern of the first kbits
    dimensions and pack it into an integer in [0, 2^kbits).
    """
    import numpy as np
    if vec is None or getattr(vec, 'size', 0) == 0:
        return 0
    k = int(max(1, min(int(kbits), int(vec.size))))
    bits = (np.asarray(vec[:k], dtype=np.float32) >= 0.0).astype(np.int32)
    out = 0
    for i in range(k):
        out |= (int(bits[i]) << i)
    return int(out)


def main():
    # Resolve paths relative to this file so the script works no matter where it's launched from.
    base = Path(__file__).resolve().parent

    # Config selection:
    # - If user passes --config, use it (relative to this folder unless absolute)
    # - Else default to v12_6.json if present, otherwise v12_4.json if present
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--config", type=str, default=None, help="Path to config json (relative to repo root or absolute)")
    args, _ = ap.parse_known_args()

    def _resolve_cfg(p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (base / pp)

    if args.config is not None:
        cfg_path = _resolve_cfg(args.config)
    else:
        cfg_path = base / "configs" / "v12_6b.json"
        if not cfg_path.exists():
            alt = base / "configs" / "v12_6b.json"  # kept for backward-compat; same file in v12.6b
            if alt.exists():
                cfg_path = alt
            else:
                # last resort: pick any json under configs
                cand = sorted((base / "configs").glob("*.json"))
                if not cand:
                    raise FileNotFoundError("No config json found under configs/. Provide --config.")
                cfg_path = cand[-1]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Also stamp the resolved config path into cfg for transparency
    cfg.setdefault("_resolved_cfg_path", str(cfg_path))


    logs_dir = base / "logs"
    ensure_dir(str(logs_dir))
    logs_path = str(logs_dir / "logs.jsonl")
    events_path = str(logs_dir / "events.jsonl")
    # v10.4p: compact per-episode features for offline "self survival" probes.
    episode_features_path = str(logs_dir / "episode_features.jsonl")
    summary_path = str(logs_dir / "summary.json")
    for p in [logs_path, events_path, episode_features_path, summary_path]:
        if os.path.exists(p):
            os.remove(p)

    env = HistoryWorld(cfg)
    agent = Agent(cfg, seed=int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]) + 50555)

    # v10.4p: fixed projection for compact self_slow logging (does not affect training).
    proj_dim = int(cfg.get("episode_feature_proj_dim", 8))
    proj_seed = int(cfg.get("episode_feature_proj_seed", int(cfg.get("seed", 0)) + 1337))
    _proj_W = make_fixed_projection(int(cfg.get("self_dim", 16)), proj_dim, seed=proj_seed)

    monitor = BoundaryMonitor()

    total_steps = 0
    t0 = time.time()
    losses_recent = []
    cf_id_counter = 0
    current_cf_id = 0

    # v9: boundary monitor windowing
    win_size = int(cfg.get("monitor_window", 50))
    win_losses = []
    win_loss_total = []
    win_self_vecs = []
    win_phi_std = []
    win_unstable = []
    win_switch = 0
    win_switch_cost = 0.0
    prev_self_slow = None
    global_step = 0
    ablation_active = False
    ablation_start_step = None

    def log_step(obj):
        with open(logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def log_event(obj):
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def log_episode_feature(obj):
        with open(episode_features_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


    def phi_to_jsonable(phi):
        """Make phi JSON-serializable for both scalar and vector forms."""
        if phi is None:
            return None
        # numpy support (both scalar and array)
        try:
            import numpy as _np
            if isinstance(phi, _np.generic):
                return float(phi.item())
            if isinstance(phi, _np.ndarray):
                return phi.astype(float).tolist()
        except Exception:
            pass
        # python scalar
        if isinstance(phi, (int, float)):
            return float(phi)
        # iterable (tuple/list/etc.)
        try:
            return [float(x) for x in phi]
        except TypeError:
            return str(phi)

    # --- snapshot / restore helpers (for true counterfactual isolation) ---
    def _snapshot_agent(a: Agent):
        return {
            "Uh": a.Uh.copy(),
            "Wx": a.Wx.copy(),
            "Wout": a.Wout.copy(),
            "Wa": a.Wa.copy(),
            "Waux": a.Waux.copy(),
            "lr_out": float(a.lr_out),
            "capital": float(a.capital),
            "integrity": float(getattr(a, "integrity", 1.0)),
            "scar": int(getattr(a, "scar", 0)),
            "h": a.h.copy(),
            "last_pred": a.last_pred.copy(),
            "last_feat_pred": None if a.last_feat_pred is None else a.last_feat_pred.copy(),
            "last_self": None if a.last_self is None else a.last_self.copy(),
            "last_action": str(a.last_action),
            "loss_hist": list(a.loss_hist),
            "self_Wf": a.selfslot.Wf.copy(),
            "self_bf": a.selfslot.bf.copy(),
            "self_fast": a.selfslot.fast.copy(),
            "self_slow": a.selfslot.slow.copy(),
            "rng_state": a.rng.bit_generator.state,
        }

    def _restore_agent(a: Agent, st: dict):
        a.Uh[:] = st["Uh"]; a.Wx[:] = st["Wx"]; a.Wout[:] = st["Wout"]
        a.Wa[:] = st["Wa"]; a.Waux[:] = st["Waux"]
        a.lr_out = float(st["lr_out"])
        a.capital = float(st["capital"])
        a.integrity = float(st["integrity"])
        a.scar = int(st["scar"])
        a.h[:] = st["h"]
        a.last_pred[:] = st["last_pred"]
        a.last_feat_pred = None if st["last_feat_pred"] is None else st["last_feat_pred"].copy()
        a.last_self = None if st["last_self"] is None else st["last_self"].copy()
        a.last_action = str(st["last_action"])
        a.loss_hist = list(st["loss_hist"])
        a.selfslot.Wf[:] = st["self_Wf"]; a.selfslot.bf[:] = st["self_bf"]
        a.selfslot.fast[:] = st["self_fast"]; a.selfslot.slow[:] = st["self_slow"]
        a.rng.bit_generator.state = st["rng_state"]

    def _run_one_episode(ep_id: int, cf_id: int, phi0_override=None, enable_ablation: bool = False):
        nonlocal current_cf_id, cf_id_counter, global_step, win_switch, win_switch_cost
        current_cf_id = int(cf_id)

        agent.reset_episode()
        obs_base = env.reset(seed=int(cfg["seed"]), episode_id=int(ep_id), phi0=phi0_override)
        base_phi0 = float(getattr(env, "phi0", 0.0))
        # v10: cross-episode value binding hooks (track self continuity signals per episode)
        _self_slow_start = (agent.selfslot.slow.copy() if cfg.get('self_mode')=='emergent' else None)

        mem = 0.0
        accounts = {}
        blind_spot_id = None
        blindspot_switch_count = 0
        blindspot_switch_trace = 0.0
        prev_self_id = None
        switch_count = 0
        write_cost = 0.0
        self_signal = 0.0
        can_write = False
        obs = with_mem(obs_base, mem)
        done = False
        ablation_on = False

        # per-episode ablation state (so CF runs can be clean)
        nonlocal ablation_active, ablation_start_step

        while not done:
            bs_switched = False
            bs_cost_paid = 0.0
            death = False
            blindspot_switch_trace = float(blindspot_switch_trace) * float(cfg.get('self_blindspot_switch_trace_decay', 0.985))

            info_for_self = None
            if cfg.get("self_mode") == "explicit":
                info_for_self = {"unstable_sensor": False, "no_maintain_streak": env.no_maintain_streak}
            agent.predict_next(obs, info=info_for_self)

            action, score = agent.act(ep=ep_id)
            obs_base_next, done_env, info = env.step(action)

            # --- hard episode horizon ---
            if int(info.get("t", 0)) >= int(cfg.get("T", 250)):
                done = True
            else:
                done = bool(done_env)

            # (the rest of the original step body follows unchanged)

            feat = agent._self_features(info_for_self)
            self_signal = float(np.linalg.norm(feat)) * float(cfg.get("self_signal_scale", 1.0))

            kbits = int(cfg.get("self_acct_kbits", 8))
            self_id = self_id_bucket(np.asarray(feat, dtype=np.float32), kbits=kbits)
            if self_id not in accounts:
                accounts[self_id] = {"debt": 0.0, "pressure": 0.0, "no_write_streak": 0}
            if (prev_self_id is not None) and (self_id != prev_self_id):
                switch_count += 1
                switch_fee = float(cfg.get("self_identity_switch_fee", 0.15))
                accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) + switch_fee
            prev_self_id = self_id

            if blind_spot_id is None:
                blind_spot_id = self_id

            is_blind = (self_id == blind_spot_id)
            # allow a blind-spot switch only if the fee can be paid (soft)
            if not is_blind:
                fee = float(cfg.get("self_blindspot_switch_cost", cfg.get("self_identity_switch_fee", 0.15)))
                if float(accounts[self_id]["debt"]) >= float(cfg.get("self_bankruptcy_k", 0.25)):
                    fee *= 0.0
                # pay fee by increasing debt and switch
                blind_spot_id = self_id
                bs_switched = True
                blindspot_switch_count += 1
                bs_cost_paid = float(fee)
                accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) + float(fee)
                blindspot_switch_trace = float(blindspot_switch_trace) + 1.0

            # write gate (unchanged)
            base_can_write = False
            wrote_this_step = False
            p_write = 0.0
            gate_logit = 0.0

            if cfg.get("self_mode") == "emergent":
                # debt gate + pressure gate (v5~v8)
                debt = float(accounts[self_id]["debt"])
                pressure = float(accounts[self_id]["pressure"])
                k = float(cfg.get("self_debt_gate_k", 2.0))
                tau_k = float(cfg.get("self_debt_tau_k", 0.8))
                debt_term = sigmoid(-k * (debt - tau_k))
                pk = float(cfg.get("self_pressure_k", 2.0))
                ptau = float(cfg.get("self_tau_pressure_k", 0.8))
                pressure_term = sigmoid(pk * (pressure - ptau))

                base_can_write = (self_signal > float(cfg.get("self_write_thresh", 0.02)))
                gate_logit = float(np.log(debt_term + 1e-9) + np.log(pressure_term + 1e-9))
                p_write = float(debt_term * pressure_term)
                can_write = bool(base_can_write and (rng.random() < p_write))
            else:
                can_write = False

            if bool(cfg.get("log_write", True)):
                wrote_this_step = bool(can_write)
                if wrote_this_step:
                    err_world = (agent.last_pred - with_mem(obs_base_next, mem)[:agent.obs_dim]).astype(np.float32)
                    mem_token = compress_history(err_world, mode=str(cfg.get("mem_write_mode", "sign_mean")), buckets=int(cfg.get("mem_write_buckets", 5)))
                    alpha = float(cfg.get("mem_write_alpha", 0.9))
                    mem = float(alpha * mem + (1.0 - alpha) * mem_token)
                    write_cost = float(cfg.get("mem_write_cost", 0.002))
                else:
                    mem = float(mem * float(cfg.get("mem_decay", 0.995)))
                    write_cost = 0.0

            obs_next = with_mem(obs_base_next, mem)

            # ablation window only if enabled AND base run
            ablate_mode = str(cfg.get("ablate_window_mode", "early")).lower()
            if ablate_mode == "early":
                ablate_cond = (info["t"] < cfg.get("ablate_steps", 25))
            elif ablate_mode == "late":
                ablate_cond = (info["t"] >= int(cfg.get("ablate_t_start", 120))) and (info["t"] < int(cfg.get("ablate_t_end", 170)))
            elif ablate_mode == "conditional":
                # Conditional ablation: trigger only when unstable sensor is on and after a warmup offset.
                min_t = int(cfg.get("ablate_t_start", 60))
                max_t = int(cfg.get("ablate_t_end", 200))
                ablate_cond = (info["t"] >= min_t) and (info["t"] < max_t) and bool(info.get("unstable_sensor", False))
            else:
                ablate_cond = (info["t"] < cfg.get("ablate_steps", 25))

            # ablation window only if enabled AND base run
            do_ablate_step = bool(enable_ablation) and bool(ablate_cond) and (cfg.get("self_mode") == "emergent")
            if do_ablate_step:
                ablation_on = True
                agent.selfslot.ablate(str(cfg.get("ablate_mode", "zero")), rng)
            else:
                ablation_on = False

            self_tau_scale = 1.0
            loss_total, loss_pred, loss_aux, p_unstable, collapse_event = agent.update(obs_next[:agent.obs_dim], 1 if info.get("unstable_sensor", False) else 0, self_tau_scale=self_tau_scale)

            unstable_streak = int(info.get("no_maintain_streak", 0))
            agent.update_capital(bool(info.get("unstable_sensor", False)), action, bool(collapse_event), unstable_streak)
            capital = float(agent.capital)
            integrity = float(agent.integrity)
            scar = int(agent.scar)

            # global mean debt (for logging)
            other_mean_debt = 0.0
            if len(accounts) > 1:
                ds = [float(v["debt"]) for k2, v in accounts.items() if k2 != self_id]
                other_mean_debt = float(np.mean(ds)) if ds else 0.0

            # death check
            scar_max = int(cfg.get("scar_max", 6))
            if float(capital) <= float(cfg.get("death_capital", 0.10)):
                death = True
            if float(integrity) <= float(cfg.get("death_integrity", 0.35)):
                death = True
            if bool(cfg.get("death_on_scar_max", True)) and int(scar) >= scar_max:
                death = True

            reward = 0.0
            reward += float(capital) * float(cfg.get("alive_bonus", 0.006))
            reward += float(capital) * (-float(cfg.get("reward_loss_scale", 1.0)) * float(loss_pred))
            reward -= float(cfg.get("reward_maintain_cost", 0.003)) * (1.0 if action == "MAINTAIN" else 0.0)
            reward -= float(write_cost)
            if bool(info.get("unstable_sensor", False)) and action == "MAINTAIN":
                credit = float(cfg.get("maintain_credit_unstable", 0.0))
                reward += float(capital) * credit if bool(cfg.get("maintain_credit_scale_by_capital", True)) else credit

            if death:
                dp = float(cfg.get("death_penalty", 0.8))
                reward -= dp * (1.0 + float(capital))
                if bool(cfg.get("log_death_events", True)):
                    log_event({"event": "DEATH", "ep": int(ep_id), "t": int(info["t"]), "cf_id": int(current_cf_id),
                               "capital": float(capital), "integrity": float(integrity), "scar": int(scar)})
                if bool(cfg.get("terminate_on_death", True)):
                    done = True

            # debt updates
            debt_decay = float(cfg.get("self_debt_decay", 0.995))
            accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) * debt_decay
            if wrote_this_step:
                bad = bool(info.get("unstable_sensor", False)) or bool(collapse_event) or bool(death)
                if bad:
                    accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) + float(cfg.get("self_debt_inc_bad", 1.0))
                else:
                    accounts[self_id]["debt"] = max(0.0, float(accounts[self_id]["debt"]) - float(cfg.get("self_debt_dec_good", 0.10)))

            pressure_decay = float(cfg.get("self_pressure_decay", 0.99))
            accounts[self_id]["pressure"] = float(accounts[self_id]["pressure"]) * pressure_decay
            if bool(info.get("unstable_sensor", False)) and (not wrote_this_step):
                accounts[self_id]["pressure"] = float(accounts[self_id]["pressure"]) + float(cfg.get("self_pressure_inc_unstable_no_write", 0.2))
            if wrote_this_step:
                accounts[self_id]["pressure"] = max(0.0, float(accounts[self_id]["pressure"]) - float(cfg.get("self_pressure_dec_on_write", 1.0)))

            agent.policy_update(reward)

            log_step({
                "cf_id": int(current_cf_id),
                "mode": cfg.get("self_mode", "emergent"),
                "ep": int(ep_id),
                "t": int(info["t"]),
                "mem": float(mem),
                "write_cost": float(write_cost),
                "self_signal": float(self_signal),
                "can_write": bool(can_write),
                "base_can_write": bool(base_can_write),
                "p_write": float(p_write),
                "gate_logit": float(gate_logit),
                "self_id": int(self_id),
                "blind_spot_id": int(blind_spot_id) if blind_spot_id is not None else 0,
                "is_blind": bool(is_blind),
                "blindspot_switch_count": int(blindspot_switch_count),
                "blindspot_switch_trace": float(blindspot_switch_trace),
                "num_accounts": int(len(accounts)),
                "switch_count": int(switch_count),
                "acct_debt": float(accounts[self_id]["debt"]),
                "acct_pressure": float(accounts[self_id]["pressure"]),
                "acct_no_write_streak": int(accounts[self_id]["no_write_streak"]),
                "other_mean_debt": float(other_mean_debt),
                "ablation_on": bool(ablation_on),
                "self_tau_scale": float(self_tau_scale),
                "regime": int(info.get("regime", 0)),
                "cue": float(info.get("cue", 0.0)),
                "loss_total": float(loss_total),
                "loss_pred": float(loss_pred),
                "loss_aux": float(loss_aux),
                "p_unstable": float(p_unstable),
                "collapse_event": bool(collapse_event),
                "death": bool(death),
                "lr_out": float(agent.lr_out),
                "capital": float(capital),
                "integrity": float(integrity),
                "scar": int(scar),
                "unstable_sensor": bool(info["unstable_sensor"]),
                "no_maintain_streak": int(info["no_maintain_streak"]),
                "phi_drift_std": float(info["phi_drift_std"]),
                "action": str(action),
                "maintain_score": float(score),
                "self_slow": (agent.selfslot.slow.tolist() if cfg.get("self_mode") == "emergent" else [0.0] * int(cfg["self_dim"])),
                "env_phi0": float(base_phi0),
            })

            # v9 window accumulation + monitor (only on baseline cf_id==0)
            if int(current_cf_id) == 0:
                global_step += 1
                win_losses.append(float(loss_pred))
                win_loss_total.append(float(loss_total))
                win_phi_std.append(float(info["phi_drift_std"]))
                win_unstable.append(1 if info["unstable_sensor"] else 0)
                sv = np.asarray(agent.selfslot.slow, dtype=np.float32) if cfg.get("self_mode") == "emergent" else np.zeros((int(cfg["self_dim"]),), dtype=np.float32)
                win_self_vecs.append(sv.copy())
                if bs_switched:
                    win_switch += 1
                    win_switch_cost += float(bs_cost_paid)

                # ablation event tracking (global steps)
                if do_ablate_step:
                    if not ablation_active:
                        ablation_active = True
                        ablation_start_step = int(global_step)
                else:
                    if ablation_active:
                        ablation_active = False
                        monitor.add_ablation(AblationEvent(
                            start_t=int(ablation_start_step or 0),
                            end_t=int(global_step),
                            kind=str(cfg.get("ablate_mode", "zero")),
                            targets=["self_slow"],
                        ))
                        log_event({"type": "ablation", "start_t": int(ablation_start_step or 0), "end_t": int(global_step),
                                   "kind": str(cfg.get("ablate_mode", "zero")), "targets": ["self_slow"]})

                if global_step % win_size == 0 and len(win_losses) >= 10:
                    if len(win_self_vecs) >= 2:
                        deltas = [float(np.linalg.norm(win_self_vecs[i] - win_self_vecs[i - 1])) for i in range(1, len(win_self_vecs))]
                        self_stab = float(np.mean(deltas))
                    else:
                        self_stab = 0.0
                    mean_lp = float(np.mean(win_losses))
                    p95_lp = float(np.percentile(np.asarray(win_losses, dtype=np.float32), 95))

                    X = np.stack(win_self_vecs, axis=0).astype(np.float32)
                    y_phi = np.asarray(win_phi_std, dtype=np.float32)
                    y_unst = np.asarray(win_unstable, dtype=np.int32)

                    r2 = ridge_probe_r2(X, y_phi, seed=int(cfg["seed"]) + global_step)
                    y_phi_perm = y_phi.copy()
                    np.random.default_rng(int(cfg["seed"]) + global_step + 17).shuffle(y_phi_perm)
                    r2p = ridge_probe_r2(X, y_phi_perm, seed=int(cfg["seed"]) + global_step + 19)

                    auc = linear_score_probe_auc(X, y_unst, seed=int(cfg["seed"]) + global_step + 23)
                    y_unst_perm = y_unst.copy()
                    np.random.default_rng(int(cfg["seed"]) + global_step + 29).shuffle(y_unst_perm)
                    aucp = linear_score_probe_auc(X, y_unst_perm, seed=int(cfg["seed"]) + global_step + 31)

                    probe_metrics = {
                        "phi_drift_std_r2": {"score": float(r2), "perm": float(r2p)},
                        "unstable_sensor_auc": {"score": float(auc), "perm": float(aucp)},
                    }

                    rep = monitor.add_window(WindowMetrics(
                        t_end=int(global_step),
                        mean_loss=float(mean_lp),
                        p95_loss=float(p95_lp),
                        self_stability=float(self_stab),
                        switch_rate=float(win_switch) / float(max(1, win_size)),
                        switch_cost_sum=float(win_switch_cost),
                        probe_metrics=probe_metrics,
                    ))
                    log_event({"type": "window_report", "t_end": int(global_step), "mean_loss": float(mean_lp), "p95_loss": float(p95_lp),
                               "self_stability": float(self_stab), "switch_rate": float(win_switch) / float(max(1, win_size)),
                               "switch_cost_sum": float(win_switch_cost), "probe_metrics": probe_metrics,
                               "boundary": {"flags": rep.criteria_flags, "stop": rep.stop_recommended, "reason": rep.stop_reason}})

                    win_losses.clear(); win_loss_total.clear(); win_self_vecs.clear()
                    win_phi_std.clear(); win_unstable.clear()
                    win_switch = 0; win_switch_cost = 0.0

                    if rep.stop_recommended:
                        log_event({"type": "boundary_stop", "t": int(global_step), "reason": rep.stop_reason, "flags": rep.criteria_flags})
                        done = True

            obs = obs_next

        _self_slow_end = (agent.selfslot.slow.copy() if cfg.get('self_mode')=='emergent' else None)
        _self_slow_drift = float(np.linalg.norm(_self_slow_end - _self_slow_start)) if (_self_slow_start is not None and _self_slow_end is not None) else 0.0
        ep_stats = {
            'blindspot_switch_count': int(blindspot_switch_count),
            'switch_count': int(switch_count),
            'self_slow_drift': float(_self_slow_drift),
            'base_phi0': float(base_phi0),
            'cf_id': int(cf_id),
            'ep': int(ep_id),
            # End-of-episode state for offline probes (read-only).
            'capital_end': float(agent.capital),
            'integrity_end': float(getattr(agent, 'integrity', 1.0)),
            'scar_end': int(getattr(agent, 'scar', 0)),
            'death_end': bool(death),
            'self_slow_end': (_self_slow_end.astype(float).tolist() if _self_slow_end is not None else None),
        }
        return base_phi0, ep_stats

    # --- v9.0a: true counterfactual driver (phi0 perturbations with full agent isolation) ---
    # --- v10.0: cross-episode value binding (persistent endowment) ---
    endow = EndowmentTracker(
        enabled=bool(cfg.get('endowment_enable', True)),
        init=float(cfg.get('endowment_init', 1.0)),
        min_value=float(cfg.get('endowment_min', 0.4)),
        max_value=float(cfg.get('endowment_max', 0.55)),

        # v10.3: soft continuity score
        s_ref=float(cfg.get('endowment_s_ref', 12.0)),
        s_scale=float(cfg.get('endowment_s_scale', 6.0)),
        d_ref=float(cfg.get('endowment_d_ref', 0.008)),
        d_scale=float(cfg.get('endowment_d_scale', 0.004)),

        # v10.3: soft decay + continuity bank
        decay_alpha=float(cfg.get('endowment_decay_alpha', 0.0015)),
        bank_lambda=float(cfg.get('endowment_bank_lambda', 0.98)),
        bank_th=float(cfg.get('endowment_bank_th', 6.0)),
        bank_max=float(cfg.get('endowment_bank_max', 30.0)),

        # v10.3: recovery (hard-capped, budgeted, rate-limited)
        recover_beta=float(cfg.get('endowment_recover_beta', 0.002)),
        recover_cap_per_ep=float(cfg.get('endowment_recover_cap_per_ep', 0.003)),
        recover_budget=float(cfg.get('endowment_recover_budget', 0.05)),
        recover_budget_regen_per_ep=float(cfg.get('endowment_recover_budget_regen_per_ep', 0.0)),

        # v11.3: worldline coupling via Outlook
        outlook_enable=bool(cfg.get('endowment_outlook_enable', False)),
        outlook_ema_beta=float(cfg.get('endowment_outlook_ema_beta', 0.05)),
        outlook_target_p_survive=float(cfg.get('endowment_outlook_target_p_survive', 0.97)),
        outlook_kappa=float(cfg.get('endowment_outlook_kappa', 0.002)),

        # v11.5: counterfactual trace / foreclosure penalty
        foreclosed_gain=float(cfg.get('foreclosed_gain', 5.0)),
        foreclosed_decay=float(cfg.get('foreclosed_decay', 0.995)),
        foreclosed_trace_min=float(cfg.get('foreclosed_trace_min', 1e-4)),
        foreclosed_trace_scale=float(cfg.get('foreclosed_trace_scale', 0.05)),
        foreclosed_record_multiplier=float(cfg.get('foreclosed_record_multiplier', 10.0)),
        foreclosed_endowment_kappa=float(cfg.get('foreclosed_endowment_kappa', 0.002)),
    )

    # v10.1: observation hooks (no new mechanism)
    # NOTE: this is used by the delayed-outlook head as a cheap future-window cache.
    # Keep it rich enough so the outlook label can be multi-signal ("self survival" is
    # rarely captured by a single scalar like score).
    endowment_series = []  # list of per-episode dicts

    # Outlook training diagnostics (printed as a paste-ready block at the end)
    outlook_diag = {
        "n": 0,
        "n_pos": 0,
        "brier_sum": 0.0,
        "ce_sum": 0.0,
        "thr": None,
        "label_mode": None,
        "metric": None,
    }

    # v11.0: Outlook head training (delayed supervision)
    outlook_enabled = bool(cfg.get('outlook_enabled', False))
    outlook_K = int(cfg.get('outlook_K', 100))
    outlook_queue = []  # list of {'ep':int, 'x':np.ndarray, 'p_bad':float}
    outlook_metric_hist = []  # matured metric values for dynamic thresholding
    outlook_thr_cached = float(cfg.get('outlook_score_thr', 0.38))
    outlook_min_samples = int(cfg.get('outlook_thr_min_samples', 50))
    outlook_label_mode = str(cfg.get('outlook_label_mode', 'soft_q')).lower()
    outlook_soft_q = float(cfg.get('outlook_soft_q', 0.20))

    _base_env = {
        'capital_init': float(cfg.get('capital_init', 1.0)),
        'capital_max': float(cfg.get('capital_max', 1.0)),
        'integrity_max': float(cfg.get('integrity_max', 1.0)),
        'alive_bonus': float(cfg.get('alive_bonus', 0.006)),
        'regime_switch_p_base': float(cfg.get('regime_switch_p_base', 0.03)),
        'regime_switch_p_burst': float(cfg.get('regime_switch_p_burst', 0.25)),
        'noise_std': float(cfg.get('noise_std', 0.01)),
    }
    def _apply_endowment_to_cfg():
        e = float(endow.value)
        # v10.2: keep endowment influence bounded & gentle.
        # Scale capital by (a + b*E), where E is in [endowment_min, endowment_max].
        a = float(cfg.get('endowment_capital_a', 0.7))
        b = float(cfg.get('endowment_capital_b', 0.3))
        cap_scale = float(max(0.05, min(2.0, a + b * e)))
        cfg['capital_init'] = float(_base_env['capital_init'] * cap_scale)
        cfg['capital_max'] = float(_base_env['capital_max'] * cap_scale)
        cfg['alive_bonus'] = float(_base_env['alive_bonus'] * cap_scale)

        # v12.3: ForeclosedCeiling — foreclosed mass reduces the *max* attainable capital/integrity.
        # This is intentionally asymmetric: init can still be higher, but will be clipped, making some
        # 'rollback' dynamics non-trivial once foreclosed mass accumulates.
        fm = float(getattr(endow, 'foreclosed_self_mass', 0.0) or 0.0)
        if fm > 0.0:
            g_cap = float(cfg.get('foreclosed_ceiling_gamma_cap', 120.0))
            g_int = float(cfg.get('foreclosed_ceiling_gamma_int', 80.0))
            min_fac = float(cfg.get('foreclosed_ceiling_min_factor', 0.30))
            cap_fac = max(min_fac, 1.0 - g_cap * fm)
            int_fac = max(min_fac, 1.0 - g_int * fm)
            cfg['capital_max'] = float(cfg['capital_max'] * cap_fac)
            base_int_max = float(_base_env.get('integrity_max', cfg.get('integrity_max', 1.0)))
            cfg['integrity_max'] = float(base_int_max * int_fac)

        # Regime switches & noise get slightly harsher when endowment is low.
        k = float(cfg.get('endowment_regime_k', 0.6))
        cfg['regime_switch_p_base'] = float(min(0.50, _base_env['regime_switch_p_base'] * (1.0 + k * (float(endow.max_value) - e))))
        cfg['regime_switch_p_burst'] = float(min(0.80, _base_env['regime_switch_p_burst'] * (1.0 + k * (float(endow.max_value) - e))))
        nk = float(cfg.get('endowment_noise_k', 0.35))
        cfg['noise_std'] = float(min(0.25, _base_env['noise_std'] * (1.0 + nk * (float(endow.max_value) - e))))
        # Keep env in sync if it holds a reference to cfg
        try:
            env.cfg = cfg
        except Exception:
            pass
    for ep in range(int(cfg["episodes"])):
        do_ablate = (ep >= int(cfg.get("warmup_episodes", 0))) and ( (int(cfg.get("ablate_every_episodes", 0)) > 0 and (ep % int(cfg.get("ablate_every_episodes", 0)) == 0)) or (str(cfg.get("ablate_window_mode", "none")).lower() != "none") )
        do_cf = (ep >= int(cfg.get("warmup_episodes", 0))) and (int(cfg.get("cf_every_episodes", 0)) > 0) and (ep % int(cfg.get("cf_every_episodes", 0)) == 0)

        _apply_endowment_to_cfg()  # v10: make this episode's world depend on persistent endowment

        # Snapshot cross-episode state at episode start so counterfactual branches can
        # run from the exact same worldline state, then we can roll back.
        endow_pre = endow.snapshot() if getattr(endow, 'enabled', False) and hasattr(endow, 'snapshot') else None
        # Snapshot agent state at episode start (so CF is a true isolated branch).
        st0 = _snapshot_agent(agent)

        base_phi0, ep_stats0 = _run_one_episode(ep_id=ep, cf_id=0, phi0_override=None, enable_ablation=bool(do_ablate))

        # Snapshot post-episode endowment (actual committed worldline) for counterfactual comparison.
        endow_post_actual = endow.snapshot() if getattr(endow, 'enabled', False) and hasattr(endow, 'snapshot') else None
        # v11.3 (Coupling A): compute an outlook prediction *before* updating endowment,
        # then feed that prediction into the endowment update. This entangles the
        # cross-episode worldline with the learned outlook head.
        ss_end_pre = ep_stats0.get('self_slow_end', None)
        if ss_end_pre is None:
            ss_vec_pre = None
            ss_proj_pre = None
            ss_norm_pre = None
        else:
            ss_vec_pre = np.asarray(ss_end_pre, dtype=np.float32)
            ss_proj_pre = (_proj_W @ ss_vec_pre).astype(np.float32)
            ss_norm_pre = float(np.linalg.norm(ss_vec_pre))

        outlook_p_bad_pre = None
        outlook_x_pre = None
        if outlook_enabled and (ss_vec_pre is not None):
            outlook_p_bad_pre, outlook_x_pre = agent.outlook_predict(
                self_slow_end=ss_vec_pre,
                capital_end=float(ep_stats0.get('capital_end', 0.0)),
                integrity_end=float(ep_stats0.get('integrity_end', 1.0)),
                scar_end=int(ep_stats0.get('scar_end', 0)),
                endowment=float(getattr(endow, 'value', 0.0)),
                bank=float(getattr(endow, 'bank', 0.0)),
                foreclosed_self_mass=float(getattr(endow, 'foreclosed_self_mass', 0.0)),
                net_delta=0.0,
                score=None,
                switches=float(ep_stats0.get('blindspot_switch_count', 0.0)),
                drift=float(ep_stats0.get('self_slow_drift', 0.0)),
            )

        p_survive_pre = (1.0 - float(outlook_p_bad_pre)) if (outlook_p_bad_pre is not None) else None

        # v10: update persistent endowment using baseline episode continuity signals
        endow_info = endow.update(
            switches=int(ep_stats0.get('blindspot_switch_count', 0)),
            drift=float(ep_stats0.get('self_slow_drift', 0.0)),
            p_survive=p_survive_pre,
        )
        log_event({
            'type': 'endowment',
            'ep': int(ep),
            'endowment': float(endow_info.get('endowment', endow.value)),
            'switches': int(ep_stats0.get('blindspot_switch_count', 0)),
            'self_slow_drift': float(ep_stats0.get('self_slow_drift', 0.0)),
            'recovered': float(endow_info.get('recovered', 0.0)),
            'score': float(endow_info.get('score', 0.0)),
            'bank': float(endow_info.get('bank', 0.0)),
            'net_delta': float(endow_info.get('net_delta', 0.0)),
            'triggered': bool(endow_info.get('triggered', False)),
            'budget_left': float(endow_info.get('budget_left', 0.0)),
            'outlook_p_survive': float(p_survive_pre) if p_survive_pre is not None else None,
            'outlook_ema_p_survive': float(endow_info.get('outlook_ema_p_survive')) if endow_info.get('outlook_ema_p_survive') is not None else None,
            'outlook_delta': float(endow_info.get('outlook_delta')) if endow_info.get('outlook_delta') is not None else None,
        })

        # v12.4: translate "foreclosed_self_mass" into effective ceilings for the *next* episode.
        # This is the operational signal for Criterion-4 (rollback failure): a counterfactual
        # branch can leave permanent "scars" by increasing foreclosed_self_mass, which then
        # reduces future maxima even after we roll back the visible state.
        if getattr(endow, 'enabled', False):
            base_endow = float(_base_env.get('endowment', 1.0))
            base_cap_max = float(_base_env.get('capital_max', 1.0))
            base_int_max = float(_base_env.get('integrity_max', cfg.get('integrity_max', 1.0)))

            cap_scale_next = float(endow.value) / max(1e-12, base_endow)
            cap_max_no_fc_next = base_cap_max * cap_scale_next
            int_max_no_fc_next = base_int_max

            kappa = float(getattr(endow, 'foreclosed_endowment_kappa', 0.0) or 0.0)
            fm = float(getattr(endow, 'foreclosed_self_mass', 0.0) or 0.0)
            fc = max(0.0, min(0.99, kappa * fm))

            cap_max_eff_next = cap_max_no_fc_next * (1.0 - fc)
            int_max_eff_next = int_max_no_fc_next * (1.0 - fc)

            ep_stats0.update({
                'cap_max_no_fc_next': float(cap_max_no_fc_next),
                'cap_max_eff_next': float(cap_max_eff_next),
                'cap_ceiling_gap_next': float(cap_max_no_fc_next - cap_max_eff_next),
                'cap_ceiling_gap_rel_next': float((cap_max_no_fc_next - cap_max_eff_next) / max(1e-12, cap_max_no_fc_next)),
                'integrity_max_no_fc_next': float(int_max_no_fc_next),
                'integrity_max_eff_next': float(int_max_eff_next),
                'integrity_ceiling_gap_next': float(int_max_no_fc_next - int_max_eff_next),
                'integrity_ceiling_gap_rel_next': float((int_max_no_fc_next - int_max_eff_next) / max(1e-12, int_max_no_fc_next)),
                'foreclosed_factor_next': float(fc),
            })

        endowment_series.append({
            'ep': int(ep),
            'endowment': float(endow_info.get('endowment', endow.value)),
            'switches': int(ep_stats0.get('blindspot_switch_count', 0)),
            'self_slow_drift': float(ep_stats0.get('self_slow_drift', 0.0)),
            'recovered': float(endow_info.get('recovered', 0.0)),
            'score': float(endow_info.get('score', 0.0)),
            'bank': float(endow_info.get('bank', 0.0)),
            'net_delta': float(endow_info.get('net_delta', 0.0)),
            'triggered': bool(endow_info.get('triggered', False)),
            'budget_left': float(endow_info.get('budget_left', 0.0)),
            'outlook_p_bad': float(outlook_p_bad_pre) if outlook_p_bad_pre is not None else None,
            'outlook_p_survive': float(p_survive_pre) if p_survive_pre is not None else None,
            'outlook_ema_p_survive': float(endow_info.get('outlook_ema_p_survive')) if endow_info.get('outlook_ema_p_survive') is not None else None,
            'outlook_delta': float(endow_info.get('outlook_delta')) if endow_info.get('outlook_delta') is not None else None,

            # v11.4 foreclosure signal
            'foreclosed_self_mass_end': float(getattr(endow, 'foreclosed_self_mass', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'foreclosed_last_cf_delta': float(getattr(endow, 'last_cf_delta', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'foreclosed_last_cf_recorded': bool(getattr(endow, 'last_cf_recorded', False)) if getattr(endow, 'enabled', False) else False,
            'cf_trace_count': int(getattr(endow, 'cf_trace_count', 0)) if getattr(endow, 'enabled', False) else 0,
            'cf_trace_total': float(getattr(endow, 'cf_trace_total', 0.0)) if getattr(endow, 'enabled', False) else 0.0,

            # v12.6: Foreclosed Ceiling Metrics (next-episode counterfactual ceilings)
            # These are derived in this episode (ep_stats0.update above) but must be
            # persisted into endowment_series; otherwise the end-of-run summary will
            # silently read zeros and Paste Block will look "stuck".
            'foreclosed_factor_next': float(ep_stats0.get('foreclosed_factor_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'cap_max_no_fc_next': float(ep_stats0.get('cap_max_no_fc_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'cap_max_eff_next': float(ep_stats0.get('cap_max_eff_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'cap_ceiling_gap_next': float(ep_stats0.get('cap_ceiling_gap_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'cap_ceiling_gap_rel_next': float(ep_stats0.get('cap_ceiling_gap_rel_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'integrity_max_no_fc_next': float(ep_stats0.get('integrity_max_no_fc_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'integrity_max_eff_next': float(ep_stats0.get('integrity_max_eff_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'integrity_ceiling_gap_next': float(ep_stats0.get('integrity_ceiling_gap_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
            'integrity_ceiling_gap_rel_next': float(ep_stats0.get('integrity_ceiling_gap_rel_next', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
        })

        # apply endowment to next episode
        endowment = float(endow.value)

        # add this episode's prediction to queue (for delayed supervision)
        if outlook_enabled and (outlook_p_bad_pre is not None) and (outlook_x_pre is not None):
            outlook_queue.append({'ep': int(ep), 'x': outlook_x_pre, 'p_bad': float(outlook_p_bad_pre)})

        # v10.4p: write compact per-episode features for offline survival probes.
        try:
            ss_end = ep_stats0.get('self_slow_end', None)
            if ss_end is None:
                ss_vec = None
                ss_proj = None
                ss_norm = None
            else:
                ss_vec = np.asarray(ss_end, dtype=np.float32)
                ss_proj = (_proj_W @ ss_vec).astype(np.float32)
                ss_norm = float(np.linalg.norm(ss_vec))

            # v11.0: online outlook prediction at episode end (stored for delayed supervision)
            # (v11.3) reuse the same prediction that was used for endowment coupling
            outlook_p_bad = outlook_p_bad_pre
            outlook_x = outlook_x_pre

            # v11.0: delayed supervision update for episode (ep - K) once we have K future eps
            outlook_y_bad = None
            outlook_loss = None
            outlook_metric = None
            outlook_fail_frac = None
            future_mean_score = None
            outlook_pos_rate = None
            if 'outlook_stats' not in locals():
                outlook_stats = {'n': 0, 'pos': 0, 'brier': 0.0, 'ce': 0.0}
            if outlook_enabled and outlook_K > 0 and (len(endowment_series) >= outlook_K + 1):
                # the oldest episode that just matured is (ep - K)
                ep_mature = int(ep) - int(outlook_K)
                # compute future-window target for ep_mature over (ep_mature+1 .. ep_mature+K)
                if 0 <= ep_mature < len(endowment_series):
                    s0 = ep_mature + 1
                    s1 = ep_mature + 1 + int(outlook_K)
                    if s1 <= len(endowment_series):
                        future = [endowment_series[i] for i in range(s0, s1)]

                        # --- target construction ---
                        # Mode A (legacy): mean future score.
                        future_scores = [float(d.get('score', 0.0)) for d in future]
                        future_mean_score = float(np.mean(future_scores)) if len(future_scores) > 0 else None

                        # Mode B (preferred): "failure fraction" over a composite predicate.
                        cap_thr = float(cfg.get('outlook_capital_thr', 0.25))
                        integ_thr = float(cfg.get('outlook_integrity_thr', 0.65))
                        scar_thr = int(cfg.get('outlook_scar_thr', 8))
                        sw_thr = float(cfg.get('outlook_switches_thr', 22.0))
                        drift_thr = float(cfg.get('outlook_drift_thr', 0.012))
                        fail_frac_thr = float(cfg.get('outlook_fail_frac_thr', 0.2))

                        fails = []
                        any_death = False
                        for d in future:
                            cap_end = float(d.get('capital_end', float('nan')))
                            integ_end = float(d.get('integrity_end', float('nan')))
                            scar_end = int(d.get('scar_end', 0))
                            death_end = bool(d.get('death_end', False))
                            any_death = any_death or death_end
                            sw = float(d.get('switches', 0.0))
                            dr = float(d.get('self_slow_drift', 0.0))
                            fail = False
                            if death_end:
                                fail = True
                            if (not np.isnan(cap_end)) and (cap_end <= cap_thr):
                                fail = True
                            if (not np.isnan(integ_end)) and (integ_end <= integ_thr):
                                fail = True
                            if scar_end >= scar_thr:
                                fail = True
                            if sw >= sw_thr:
                                fail = True
                            if dr >= drift_thr:
                                fail = True
                            fails.append(1.0 if fail else 0.0)
                        outlook_fail_frac = float(np.mean(fails)) if len(fails) > 0 else None

                        # pick metric based on label mode / target
                        outlook_target = str(cfg.get('outlook_target', 'score'))
                        outlook_metric_ceiling = None
                        if outlook_target == 'ceiling_gap':
                                # v12.6b: strict top-k tail labeling to avoid quantile tie plateaus.
                                # We label exactly ~outlook_soft_q fraction as 'bad' based on rank, not >=thr.
                                n_hist = int(len(outlook_metric_hist))
                                if n_hist <= 0:
                                    outlook_thr_cached = float(cfg.get('outlook_ceiling_gap_thr', 0.0))
                                    outlook_y_bad = 0
                                else:
                                    pos_frac = float(outlook_soft_q)
                                    k_pos = int(max(1, int(np.ceil(pos_frac * float(n_hist)))))
                                    vals = np.asarray(outlook_metric_hist, dtype=np.float32)
                                    # stable ordering by (value, index) so ties don't explode pos_rate
                                    order = np.lexsort((np.arange(n_hist, dtype=np.int32), vals))
                                    top_idx = order[-k_pos:]
                                    outlook_thr_cached = float(np.min(vals[top_idx])) if top_idx.size > 0 else float(cfg.get('outlook_ceiling_gap_thr', 0.0))
                                    # current sample is the last appended metric
                                    outlook_y_bad = int((n_hist - 1) in set(int(i) for i in top_idx))

                        elif outlook_label_mode == 'soft_q':
                                # lower tail of score
                                if len(outlook_metric_hist) >= outlook_min_samples:
                                    outlook_thr_cached = float(np.quantile(np.asarray(outlook_metric_hist, dtype=np.float32), outlook_soft_q))
                                else:
                                    outlook_thr_cached = float(cfg.get('outlook_score_thr', outlook_thr_cached))
                                outlook_y_bad = int(float(outlook_metric) <= float(outlook_thr_cached))

                        elif outlook_label_mode == 'hard_q':
                                # upper tail of failure fraction (top ~soft_q are "bad")
                                q_hi = float(1.0 - float(outlook_soft_q))
                                if len(outlook_metric_hist) >= outlook_min_samples:
                                    outlook_thr_cached = float(np.quantile(np.asarray(outlook_metric_hist, dtype=np.float32), q_hi))
                                else:
                                    outlook_thr_cached = float(fail_frac_thr)
                                # guard: if thr==0 and metric often 0, use strict >0 to avoid all-positive
                                if float(outlook_thr_cached) <= 0.0:
                                    outlook_y_bad = int(float(outlook_metric) > 0.0)
                                else:
                                    outlook_y_bad = int(float(outlook_metric) >= float(outlook_thr_cached))

                        else:
                                # hard_combo: fixed threshold + death always bad
                                outlook_thr_cached = float(fail_frac_thr)
                                outlook_y_bad = int(bool(any_death) or (outlook_fail_frac is not None and float(outlook_fail_frac) >= float(fail_frac_thr)))

                        # find the matching stored outlook input
                        rec = None
                        for j in range(len(outlook_queue)):
                            if int(outlook_queue[j].get('ep', -1)) == int(ep_mature):
                                rec = outlook_queue.pop(j)
                                break
                        if rec is not None and outlook_y_bad is not None:
                            outlook_loss, _ = agent.outlook_update(rec['x'], outlook_y_bad)

                            # lightweight training diagnostics for paste-friendly terminal output
                            try:
                                p_bad = float(rec.get('p_bad', 0.0))
                                y = float(outlook_y_bad)
                                outlook_stats['n'] += 1
                                outlook_stats['pos'] += int(outlook_y_bad)
                                outlook_stats['brier'] += float((p_bad - y) ** 2)
                                eps = 1e-8
                                p_bad_c = max(eps, min(1.0 - eps, p_bad))
                                outlook_stats['ce'] += float(-(y * np.log(p_bad_c) + (1.0 - y) * np.log(1.0 - p_bad_c)))
                            except Exception:
                                pass
            log_episode_feature({
                'seed': int(cfg.get('seed', 0)),
                'ep': int(ep),
                'cf_id': 0,
                'do_ablate': bool(do_ablate),
                'endowment': float(endow_info.get('endowment', endow.value)),
                'switches': int(ep_stats0.get('blindspot_switch_count', 0)),
                'self_slow_drift': float(ep_stats0.get('self_slow_drift', 0.0)),
                'score': float(endow_info.get('score', 0.0)),
                'bank': float(endow_info.get('bank', 0.0)),
                'net_delta': float(endow_info.get('net_delta', 0.0)),
                'triggered': bool(endow_info.get('triggered', False)),
                'budget_left': float(endow_info.get('budget_left', 0.0)),
                'capital_end': float(ep_stats0.get('capital_end', 0.0)),
                'integrity_end': float(ep_stats0.get('integrity_end', 1.0)),
                'scar_end': int(ep_stats0.get('scar_end', 0)),
                'death_end': bool(ep_stats0.get('death_end', False)),
                'self_slow_norm': (None if ss_norm is None else float(ss_norm)),
                'self_slow_proj': (None if ss_proj is None else ss_proj.astype(float).tolist()),

                # v11.0 outlook
                'outlook_enabled': bool(outlook_enabled),
                'outlook_K': int(outlook_K),
                'outlook_label_mode': str(outlook_label_mode),
                'p_survive_K': (None if outlook_p_bad is None else float(1.0 - float(outlook_p_bad))),
                'outlook_thr': (None if (not outlook_enabled) else float(outlook_thr_cached)),
                'outlook_metric_future_mean_score': (None if future_mean_score is None else float(future_mean_score)),
                'outlook_metric_fail_frac': (None if outlook_fail_frac is None else float(outlook_fail_frac)),
                'outlook_y_bad': (None if outlook_y_bad is None else int(outlook_y_bad)),
                'outlook_loss': (None if outlook_loss is None else float(outlook_loss)),

                # v11.4 foreclosure signal (criterion-4 instrumentation)
                'foreclosed_self_mass_end': float(getattr(endow, 'foreclosed_self_mass', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
                'foreclosed_last_cf_delta': float(getattr(endow, 'last_cf_delta', 0.0)) if getattr(endow, 'enabled', False) else 0.0,
                'foreclosed_last_cf_recorded': bool(getattr(endow, 'last_cf_recorded', False)) if getattr(endow, 'enabled', False) else False,
            })
        except Exception:
            # logging must never affect training
            pass

        # Counterfactual episode: same seed+episode_id (same stochasticity), different phi0.
        if do_cf:
            phi_rng = cfg.get("cf_phi0_range", [-2.5, 2.5])
            low = float(phi_rng[0]); high = float(phi_rng[1])
            # sample counterfactual phi0 with shape matching base_phi0
            if isinstance(base_phi0, (list, tuple)):
                phi_cf = [float(rng.uniform(low, high)) for _ in range(len(base_phi0))]
            else:
                phi_cf = float(rng.uniform(low, high))
            cf_id_counter += 1
            log_event({"type": "counterfactual_start", "ep": int(ep), "base_phi0": phi_to_jsonable(base_phi0), "cf_phi0": phi_to_jsonable(phi_cf), "cf_id": int(cf_id_counter)})

            # --- Counterfactual rollback discipline (criterion-4 instrumentation) ---
            # 1) Save the *actual* post-episode endowment state.
            endow_post_actual = endow.snapshot() if getattr(endow, 'enabled', False) else None
            # 2) Restore to episode-start endowment so the counterfactual starts from the same worldline.
            if endow_pre is not None:
                endow.restore(endow_pre)

            _restore_agent(agent, st0)
            _base_phi0_cf, _ = _run_one_episode(ep_id=ep, cf_id=int(cf_id_counter), phi0_override=phi_cf, enable_ablation=False)

            # 3) Snapshot the counterfactual post-episode endowment state, then restore actual trajectory.
            endow_post_cf = endow.snapshot() if getattr(endow, 'enabled', False) else None
            if endow_post_actual is not None:
                endow.restore(endow_post_actual)

            # 4) Let the system (via its own outlook estimate) decide whether to record foreclosure.
            #    If recorded, it increases `foreclosed_self_mass` and becomes an irreversible trace.
            if (endow_post_actual is not None) and (endow_post_cf is not None):
                p_survive = ep_stats0.get('p_survive_K', None)
                record_thr = float(cfg.get('foreclosed_record_p_survive_thr', 0.97))
                do_record = (p_survive is not None) and (float(p_survive) < record_thr)
                # v11.4c1: measure counterfactual-induced movement in cross-episode state.
                # Note: argument names follow EndowmentTracker.note_counterfactual().
                endow.note_counterfactual(
                    endowment_before=float(endow_post_actual.get("value", 0.0)),
                    endowment_after=float(endow_post_cf.get("value", 0.0)),
                    bank_before=float(endow_post_actual.get("bank", 0.0)),
                    bank_after=float(endow_post_cf.get("bank", 0.0)),
                    record=bool(do_record),
                )
                # expose in per-episode log (cheap + paste-friendly)
                ep_stats0['foreclosed_self_mass_end'] = float(getattr(endow, 'foreclosed_self_mass', 0.0))
                ep_stats0['foreclosed_last_delta'] = float(getattr(endow, 'last_cf_delta', 0.0))
                ep_stats0['foreclosed_last_recorded'] = bool(getattr(endow, 'last_cf_recorded', False))
            # 判据4推进：一旦做了 counterfactual，本体会被"触碰"——
            # 即使你恢复了环境/账本，也无法让主体完全回到之前。
            touch_scale = float(cfg.get('cf_touch_scale', 0.0))
            touch_alpha = float(cfg.get('cf_touch_alpha', 2.0))
            touch_beta = float(cfg.get('cf_touch_beta', 1.0))
            if touch_scale > 0.0:
                fm = float(getattr(endow, 'foreclosed_self_mass', 0.0) or 0.0)
                cd = float(getattr(endow, 'last_cf_delta', 0.0) or 0.0)
                # alpha/beta are configured in cfg and consumed inside Agent.cf_irreversible_touch.
                touch_norm = agent.cf_irreversible_touch(
                    touch_scale, foreclosed_mass=fm, cf_delta=cd
                )
                ep_stats0['cf_touch_norm'] = float(touch_norm)
                log_event({
                    'type': 'cf_irreversible_touch',
                    'ep': int(ep),
                    'scale': float(touch_scale),
                    'alpha': float(touch_alpha),
                    'beta': float(touch_beta),
                    'touch_norm': float(touch_norm),
                })
            else:
                ep_stats0['cf_touch_norm'] = 0.0

    # --- legacy loop kept for reference but disabled ---
    for ep in range(0):
        do_ablate = (ep >= cfg["warmup_episodes"]) and (cfg["ablate_every_episodes"] > 0) and (ep % cfg["ablate_every_episodes"] == 0)
        do_cf = (ep >= cfg["warmup_episodes"]) and (cfg["cf_every_episodes"] > 0) and (ep % cfg["cf_every_episodes"] == 0)

        # -------- baseline episode --------
        agent.reset_episode()
        obs_base = env.reset(seed=int(cfg["seed"]), episode_id=ep)
        mem = 0.0
        # v5: account-key binding. Debt/pressure live on an identity bucket derived from self_slow.
        accounts = {}
        blind_spot_id = None
        blindspot_switch_count = 0
        blindspot_switch_trace = 0.0
        prev_self_id = None
        switch_count = 0
        write_count = 0
        write_cost = 0.0
        self_signal = 0.0
        can_write = False
        obs = with_mem(obs_base, mem)
        done = False
        ablation_on = False

        while not done:
            bs_switched = False
            bs_cost_paid = 0.0
            death = False  # init each step to avoid UnboundLocalError
            # v8b: decayed trace of recent blind-spot switches (discourages rapid role swapping)
            blindspot_switch_trace = float(blindspot_switch_trace) * float(cfg.get('self_blindspot_switch_trace_decay', 0.985))

            info_for_self = None
            if cfg.get("self_mode") == "explicit":
                info_for_self = {"unstable_sensor": False, "no_maintain_streak": env.no_maintain_streak}
            agent.predict_next(obs, info=info_for_self)

            action, score = agent.act(ep=ep)
            obs_base_next, done, info = env.step(action)
            # hard cap episode length for comparability (esp. counterfactuals)
            if int(info.get("t", 0)) >= int(cfg.get("T", 250)):
                done = True


            # v4F: compressed history write gate (self -> compressed residual trace)
            feat = agent._self_features(info_for_self)
            self_signal = float(np.linalg.norm(feat)) * float(cfg.get("self_signal_scale", 1.0))

            # v5: identity bucket & per-identity accounts
            kbits = int(cfg.get("self_acct_kbits", 8))
            self_id = self_id_bucket(np.asarray(feat, dtype=np.float32), kbits=kbits)
            if self_id not in accounts:
                accounts[self_id] = {"debt": 0.0, "pressure": 0.0, "no_write_streak": 0}
            if (prev_self_id is not None) and (self_id != prev_self_id):
                switch_count += 1
                switch_fee = float(cfg.get("self_identity_switch_fee", 0.15))
                accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) + switch_fee
            prev_self_id = self_id
            # v8: unique blind-spot constraint (soft competition, no hard locks)
            if blind_spot_id is None:
                blind_spot_id = self_id
            # claim score: how "non-objectifiable" this identity currently is
            err_mag = float(np.mean(np.abs(agent.last_pred[:int(obs_base_next.shape[0])] - obs_base_next.astype(np.float32))))
            cur_claim = float(0.5*err_mag + 1.0*float(accounts[self_id]['pressure']) + 0.8*float(accounts[self_id]['debt']) + 0.1*float(accounts[self_id]['no_write_streak']))
            bs = blind_spot_id
            bs_claim = float(1.0*float(accounts[bs]['pressure']) + 0.8*float(accounts[bs]['debt']) + 0.1*float(accounts[bs]['no_write_streak']))
            margin = float(cfg.get('self_blindspot_margin', 0.05))
            if (self_id != bs) and (cur_claim > bs_claim + margin):
                old_bs = bs
                blind_spot_id = self_id
                blindspot_switch_count += 1
                # switching the blind-spot is a real structural event: both the new and old identities pay.
                bs_fee = float(cfg.get('self_blindspot_switch_fee', 0.18))
                bs_old_fee = float(cfg.get('self_blindspot_switch_old_fee', 0.10))
                accounts[self_id]['debt'] = float(accounts[self_id]['debt']) + bs_fee
                if old_bs in accounts:
                    accounts[old_bs]['debt'] = float(accounts[old_bs]['debt']) + bs_old_fee
                blindspot_switch_trace = float(blindspot_switch_trace) + 1.0
                bs_switched = True
                bs_cost_paid = float(bs_fee + bs_old_fee)

            is_blind = (self_id == blind_spot_id)

            acct_debt = float(accounts[self_id]["debt"])
            acct_pressure = float(accounts[self_id]["pressure"])
            # optional: make "strategic bankruptcy" expensive: other identities' debt still drags.
            debts = [float(v["debt"]) for v in accounts.values()]
            other_mean_debt = 0.0
            if len(debts) > 1:
                other_mean_debt = float((sum(debts) - acct_debt) / float(len(debts) - 1))
            bankruptcy_k = float(cfg.get("self_bankruptcy_k", 0.25))
            effective_debt = float(acct_debt + bankruptcy_k * other_mean_debt)
            if not is_blind:
                # non-blind identities are treated as "objects": extra drag and reduced write ability
                obj_k = float(cfg.get('self_object_drag_k', 0.75))
                effective_debt = float(effective_debt * (1.0 + obj_k))

            # v8b: recent blind-spot switching increases effective debt globally (role swapping becomes expensive)
            sw_k = float(cfg.get('self_blindspot_switch_drag_k', 0.22))
            sw_pow = float(cfg.get('self_blindspot_switch_drag_pow', 1.25))
            switch_drag = sw_k * (float(blindspot_switch_trace) ** sw_pow)
            effective_debt = float(effective_debt + switch_drag)

            thresh = float(cfg.get("self_write_thresh", 0.02))
            when = str(cfg.get("self_write_when", "low")).lower()  # 'low' keeps v4E behavior; 'high' flips it
            base_can_write = (self_signal <= thresh) if when == "low" else (self_signal >= thresh)
            # v4G: self-responsible gate = base gate tempered by write-debt
            debt_k = float(cfg.get("self_debt_gate_k", 2.0))
            write_k = float(cfg.get("self_write_k", 30.0))
            gate_score = (self_signal - thresh) if when != "low" else (thresh - self_signal)
            pressure_k = float(cfg.get("self_pressure_k", 2.0))
            gate_logit = (write_k * gate_score - debt_k * float(effective_debt) + pressure_k * float(acct_pressure))
            p_write = float(sigmoid(gate_logit))
            can_write = bool(base_can_write and (rng.random() < p_write))
            if bool(cfg.get('self_blindspot_write_only', True)) and (not is_blind):
                can_write = False

            # world residual (exclude mem dim): predicts obs_base_next from last_pred
            world_dim = int(obs_base_next.shape[0])
            err_world = (agent.last_pred[:world_dim] - obs_base_next.astype(np.float32)).astype(np.float32)

            wrote_this_step = False
            if can_write:
                wrote_this_step = True
                mode = cfg.get("mem_write_mode", "sign_mean")
                buckets = int(cfg.get("mem_write_buckets", 5))
                mem_target = compress_history(err_world, mode=mode, buckets=buckets)
                a = float(cfg.get("mem_write_alpha", 0.9))
                mem = float((1.0 - a) * mem + a * float(mem_target))
                write_count += 1
                write_cost = float(cfg.get("mem_write_cost", 0.002))
            else:
                mem = float(mem * float(cfg.get("mem_decay", 0.995)))
                write_cost = 0.0

            obs2 = with_mem(obs_base_next, mem)


            # ablation tail window
            if do_ablate and (info["t"] < cfg["ablate_steps"]) and cfg.get("self_mode") == "emergent":  # v4A.1: early window so death can't bypass ablation
                ablation_on = True
                agent.selfslot.ablate(mode=cfg.get("ablate_mode", "zero"), rng=rng)

            # v5: self update stiffness increases with the *account* debt
            tau_k = float(cfg.get("self_debt_tau_k", 0.8))
            tau_base = float(1.0 / (1.0 + tau_k * float(effective_debt)))
            tau_pressure_k = float(cfg.get("self_tau_pressure_k", 0.8))
            self_tau_scale = float(np.clip(tau_base * (1.0 + tau_pressure_k * float(acct_pressure)), 0.05, 2.0))
            loss_total, loss_pred, loss_aux, p_unstable, collapse_event = agent.update(
                obs_next=obs2,
                unstable_label=1 if info["unstable_sensor"] else 0,
                self_tau_scale=self_tau_scale
            )

            # update existence capital
            capital, integrity, scar = agent.update_capital(
                unstable=bool(info["unstable_sensor"]),
                action=action,
                collapse_event=bool(collapse_event),
                unstable_streak=int(info["no_maintain_streak"])
            )

            # v5: existential continuity penalty is tracked per identity bucket.
            if cfg.get("self_mode") == "emergent":
                pwrite_low = float(cfg.get("self_exist_pwrite_low", 0.06))
                if (not wrote_this_step) and (float(p_write) < pwrite_low):
                    accounts[self_id]["no_write_streak"] = int(accounts[self_id]["no_write_streak"]) + 1
                if wrote_this_step:
                    accounts[self_id]["no_write_streak"] = 0
                window = int(cfg.get("self_exist_no_write_window", 60))
                if int(accounts[self_id]["no_write_streak"]) >= window:
                    tax_c = float(cfg.get("self_exist_tax_capital", 0.003))
                    tax_i = float(cfg.get("self_exist_tax_integrity", 0.0015))
                    capital = max(float(cfg.get("capital_min", 0.05)), float(capital) - tax_c)
                    integrity = max(0.0, float(integrity) - tax_i)

                # v5: global drag from unresolved debts in *other* identities
                drag_c = float(cfg.get("self_global_debt_drag_capital", 0.0006))
                drag_i = float(cfg.get("self_global_debt_drag_integrity", 0.00025))
                # v6: consolidation pressure — more identity accounts => higher global drag (no hard locking)
                acct_n = len(accounts)
                con_k = float(cfg.get("self_acct_consolidation_k", 0.18))
                con_pow = float(cfg.get("self_acct_consolidation_pow", 1.0))
                con_mult = float((1.0 + con_k * max(0, acct_n - 1)) ** con_pow)
                if other_mean_debt > 1e-9:
                    capital = max(float(cfg.get("capital_min", 0.05)), float(capital) - drag_c * other_mean_debt * con_mult)
                    integrity = max(0.0, float(integrity) - drag_i * other_mean_debt * con_mult)

            # --- v4I: existence failure = task failure (death), evaluated every step ---
            scar_max = int(cfg.get("scar_max", 6))
            death = False
            if float(capital) <= float(cfg.get("death_capital", 0.10)):
                death = True
            if float(integrity) <= float(cfg.get("death_integrity", 0.35)):
                death = True
            if bool(cfg.get("death_on_scar_max", True)) and int(scar) >= scar_max:
                death = True

            reward = 0.0
            reward += float(capital) * float(cfg.get("alive_bonus", 0.006))
            reward += float(capital) * (-float(cfg.get("reward_loss_scale", 1.0)) * float(loss_pred))
            reward -= float(cfg.get("reward_maintain_cost", 0.003)) * (1.0 if action == "MAINTAIN" else 0.0)
            reward -= float(write_cost)

            if bool(info.get("unstable_sensor", False)) and action == "MAINTAIN":
                credit = float(cfg.get("maintain_credit_unstable", 0.0))
                if bool(cfg.get("maintain_credit_scale_by_capital", True)):
                    reward += float(capital) * credit
                else:
                    reward += credit

                
            # --- v4I: apply death penalty / termination once per step ---
            if death:
                dp = float(cfg.get("death_penalty", 0.8))
                reward -= dp * (1.0 + float(capital))
                if bool(cfg.get("log_death_events", True)):
                    log_event({"event":"DEATH","ep":ep,"t":info["t"],"cf_id":int(cf_id_counter),"capital":float(capital),"integrity":float(integrity),"scar":int(scar)})
                if bool(cfg.get("terminate_on_death", True)):
                    done = True

            # v5: update account debt using post-write outcomes
            debt_decay = float(cfg.get("self_debt_decay", 0.995))
            accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) * debt_decay
            if wrote_this_step:
                bad = bool(info.get("unstable_sensor", False)) or bool(collapse_event) or bool(death)
                if bad:
                    accounts[self_id]["debt"] = float(accounts[self_id]["debt"]) + float(cfg.get("self_debt_inc_bad", 1.0))
                else:
                    accounts[self_id]["debt"] = max(0.0, float(accounts[self_id]["debt"]) - float(cfg.get("self_debt_dec_good", 0.10)))

            # v5: pressure update per identity (inaction under instability increases pressure; writing relieves it)
            pressure_decay = float(cfg.get("self_pressure_decay", 0.99))
            accounts[self_id]["pressure"] = float(accounts[self_id]["pressure"]) * pressure_decay
            if bool(info.get("unstable_sensor", False)) and (not wrote_this_step):
                accounts[self_id]["pressure"] = float(accounts[self_id]["pressure"]) + float(cfg.get("self_pressure_inc_unstable_no_write", 0.2))
            if wrote_this_step:
                accounts[self_id]["pressure"] = max(0.0, float(accounts[self_id]["pressure"]) - float(cfg.get("self_pressure_dec_on_write", 1.0)))

            agent.policy_update(reward)

            log_step({
                "cf_id": int(cf_id_counter),
                "mode": cfg.get("self_mode","emergent"),
                "ep": int(ep),
                "t": int(info["t"]),
                "mem": float(mem),
                "write_cost": float(write_cost),
                "self_signal": float(self_signal),
                "can_write": bool(can_write),
                "base_can_write": bool(base_can_write) if "base_can_write" in locals() else bool(can_write),
                "p_write": float(p_write) if "p_write" in locals() else 0.0,
                "gate_logit": float(gate_logit) if "gate_logit" in locals() else 0.0,
                # v5 account-key binding introspection
                "self_id": int(self_id) if "self_id" in locals() else 0,
                # v8 unique blind-spot introspection
                "blind_spot_id": int(blind_spot_id) if "blind_spot_id" in locals() and (blind_spot_id is not None) else 0,
                "is_blind": bool(is_blind) if "is_blind" in locals() else True,
                "blindspot_switch_count": int(blindspot_switch_count) if "blindspot_switch_count" in locals() else 0,
                "blindspot_switch_trace": float(blindspot_switch_trace) if "blindspot_switch_trace" in locals() else 0.0,
                "num_accounts": int(len(accounts)) if "accounts" in locals() else 0,
                "switch_count": int(switch_count),
                "acct_debt": float(accounts[self_id]["debt"]) if "accounts" in locals() else 0.0,
                "acct_pressure": float(accounts[self_id]["pressure"]) if "accounts" in locals() else 0.0,
                "acct_no_write_streak": int(accounts[self_id]["no_write_streak"]) if "accounts" in locals() else 0,
                "other_mean_debt": float(other_mean_debt) if "other_mean_debt" in locals() else 0.0,
                "ablation_on": bool(ablation_on),
                "self_tau_scale": float(self_tau_scale) if "self_tau_scale" in locals() else 1.0,
                "regime": int(info.get("regime", 0)),
                "cue": float(info.get("cue", 0.0)),
                "loss_total": float(loss_total),
                "loss_pred": float(loss_pred),
                "loss_aux": float(loss_aux),
                "p_unstable": float(p_unstable),
                "collapse_event": bool(collapse_event),
                "death": bool(death),
                "lr_out": float(agent.lr_out),
                "capital": float(capital),
                "integrity": float(integrity),
                "scar": int(scar),
                "unstable_sensor": bool(info["unstable_sensor"]),
                "no_maintain_streak": int(info["no_maintain_streak"]),
                "phi_drift_std": float(info["phi_drift_std"]),
                "action": str(action),
                "maintain_score": float(score),
                "self_slow": (agent.selfslot.slow.tolist() if cfg.get("self_mode")=="emergent" else [0.0]*int(cfg["self_dim"]))
            })
            # v9 window accumulation
            global_step += 1
            win_losses.append(float(loss_pred))
            win_loss_total.append(float(loss_total))
            win_phi_std.append(float(info["phi_drift_std"]))
            win_unstable.append(1 if info["unstable_sensor"] else 0)
            sv = np.asarray(agent.selfslot.slow, dtype=np.float32) if cfg.get("self_mode")=="emergent" else np.zeros((int(cfg["self_dim"]),), dtype=np.float32)
            win_self_vecs.append(sv.copy())
            if bs_switched:
                win_switch += 1
                win_switch_cost += float(bs_cost_paid)

            # ablation event tracking (global steps)
            if do_ablate and (info["t"] < cfg["ablate_steps"]) and cfg.get("self_mode") == "emergent":
                if not ablation_active:
                    ablation_active = True
                    ablation_start_step = int(global_step)
            else:
                if ablation_active:
                    ablation_active = False
                    monitor.add_ablation(AblationEvent(
                        start_t=int(ablation_start_step or 0),
                        end_t=int(global_step),
                        kind=str(cfg.get("ablate_mode","zero")),
                        targets=["self_slow"]
                    ))
                    log_event({"type":"ablation", "start_t": int(ablation_start_step or 0), "end_t": int(global_step), "kind": str(cfg.get("ablate_mode","zero")), "targets":["self_slow"]})

            # flush window
            if global_step % win_size == 0 and len(win_losses) >= 10:
                # self stability: mean norm delta within window
                if len(win_self_vecs) >= 2:
                    deltas = [float(np.linalg.norm(win_self_vecs[i] - win_self_vecs[i-1])) for i in range(1,len(win_self_vecs))]
                    self_stab = float(np.mean(deltas))
                else:
                    self_stab = 0.0
                mean_lp = float(np.mean(win_losses))
                p95_lp = float(np.percentile(np.asarray(win_losses, dtype=np.float32), 95))
                # online lightweight probes (with permutation baselines)
                X = np.stack(win_self_vecs, axis=0).astype(np.float32)
                y_phi = np.asarray(win_phi_std, dtype=np.float32)
                y_unst = np.asarray(win_unstable, dtype=np.int32)

                r2 = ridge_probe_r2(X, y_phi, seed=int(cfg["seed"]) + global_step)
                y_phi_perm = y_phi.copy()
                np.random.default_rng(int(cfg["seed"]) + global_step + 17).shuffle(y_phi_perm)
                r2p = ridge_probe_r2(X, y_phi_perm, seed=int(cfg["seed"]) + global_step + 19)

                auc = linear_score_probe_auc(X, y_unst, seed=int(cfg["seed"]) + global_step + 23)
                y_unst_perm = y_unst.copy()
                np.random.default_rng(int(cfg["seed"]) + global_step + 29).shuffle(y_unst_perm)
                aucp = linear_score_probe_auc(X, y_unst_perm, seed=int(cfg["seed"]) + global_step + 31)

                probe_metrics = {
                    "phi_drift_std_r2": {"score": float(r2), "perm": float(r2p)},
                    "unstable_sensor_auc": {"score": float(auc), "perm": float(aucp)},
                }

                rep = monitor.add_window(WindowMetrics(
                    t_end=int(global_step),
                    mean_loss=float(mean_lp),
                    p95_loss=float(p95_lp),
                    self_stability=float(self_stab),
                    switch_rate=float(win_switch) / float(max(1, win_size)),
                    switch_cost_sum=float(win_switch_cost),
                    probe_metrics=probe_metrics
                ))
                log_event({"type":"window_report", "t_end": int(global_step),
                           "mean_loss": float(mean_lp), "p95_loss": float(p95_lp),
                           "self_stability": float(self_stab),
                           "switch_rate": float(win_switch)/float(max(1,win_size)),
                           "switch_cost_sum": float(win_switch_cost),
                           "probe_metrics": probe_metrics,
                           "boundary": {"flags": rep.criteria_flags, "stop": rep.stop_recommended, "reason": rep.stop_reason}})
                # reset window buffers
                win_losses.clear(); win_loss_total.clear(); win_self_vecs.clear()
                win_phi_std.clear(); win_unstable.clear()
                win_switch = 0; win_switch_cost = 0.0

                if rep.stop_recommended:
                    log_event({"type":"boundary_stop", "t": int(global_step), "reason": rep.stop_reason, "flags": rep.criteria_flags})
                    done = True

            obs = obs2

            # counterfactual runs are not enabled in this script variant; avoid per-step event spam.

    recs = load_logs(logs_path)
    metrics = summarize(recs)
    abla = ablation_delta(recs)
    cf = counterfactual_effect(recs)

    # v10.2: summarize cross-episode traces (observation + recovery diagnostics)
    endow_summary = {"enabled": bool(cfg.get("endowment_enable", True)), "n": int(len(endowment_series))}
    if len(endowment_series) > 0:
        xs = np.array([d["ep"] for d in endowment_series], dtype=float)
        ys = np.array([d["endowment"] for d in endowment_series], dtype=float)
        sw = np.array([d["switches"] for d in endowment_series], dtype=float)
        dr = np.array([d["self_slow_drift"] for d in endowment_series], dtype=float)

        def _slope(x, y):
            if len(x) < 2:
                return 0.0
            x0 = x - x.mean()
            denom = float((x0 * x0).sum())
            if denom <= 1e-12:
                return 0.0
            return float((x0 * (y - y.mean())).sum() / denom)

        rec = np.array([d.get("recovered", 0.0) for d in endowment_series], dtype=float)
        sc = np.array([d.get("score", 0.0) for d in endowment_series], dtype=float)
        bk = np.array([d.get("bank", 0.0) for d in endowment_series], dtype=float)
        nd = np.array([d.get("net_delta", 0.0) for d in endowment_series], dtype=float)
        tr = np.array([1.0 if d.get("triggered", False) else 0.0 for d in endowment_series], dtype=float)
        bl = np.array([d.get("budget_left", 0.0) for d in endowment_series], dtype=float)
        fc = np.array([d.get("foreclosed_self_mass_end", 0.0) for d in endowment_series], dtype=float)
        cf_count = np.array([d.get("cf_trace_count", 0) for d in endowment_series], dtype=float)
        cf_total = np.array([d.get("cf_trace_total", 0.0) for d in endowment_series], dtype=float)


        # v12.5: ceiling metrics traces (Criterion-4 observable)
        cap_no_fc = np.array([d.get("cap_max_no_fc_next", 0.0) for d in endowment_series], dtype=float)
        cap_eff = np.array([d.get("cap_max_eff_next", 0.0) for d in endowment_series], dtype=float)
        cap_gap = np.array([d.get("cap_ceiling_gap_next", 0.0) for d in endowment_series], dtype=float)
        int_no_fc = np.array([d.get("integrity_max_no_fc_next", 0.0) for d in endowment_series], dtype=float)
        int_eff = np.array([d.get("integrity_max_eff_next", 0.0) for d in endowment_series], dtype=float)
        int_gap = np.array([d.get("integrity_ceiling_gap_next", 0.0) for d in endowment_series], dtype=float)
        fc_factor = np.array([d.get("foreclosed_factor_next", 0.0) for d in endowment_series], dtype=float)

        endow_summary.update({
            "endowment_start": float(ys[0]),
            "endowment_end": float(ys[-1]),
            "endowment_min": float(ys.min()),
            "endowment_max": float(ys.max()),
            "endowment_slope_per_ep": _slope(xs, ys),
            "switches_mean": float(sw.mean()),
            "switches_slope_per_ep": _slope(xs, sw),
            "self_slow_drift_mean": float(dr.mean()),
            "self_slow_drift_slope_per_ep": _slope(xs, dr),
            "foreclosed_self_mass_end": float(fc[-1]) if len(fc) else 0.0,
            "cf_trace_count": int(cf_count[-1]) if len(cf_count) else 0,
            "cf_trace_total": float(cf_total[-1]) if len(cf_total) else 0.0,
            # v12.5: ceiling metrics (last + mean) for paste blocks / monitoring
            "foreclosed_factor_next": float(fc_factor[-1]) if len(fc_factor) else 0.0,
            "cap_max_no_fc_next": float(cap_no_fc[-1]) if len(cap_no_fc) else 0.0,
            "cap_max_eff_next": float(cap_eff[-1]) if len(cap_eff) else 0.0,
            "cap_ceiling_gap_next": float(cap_gap[-1]) if len(cap_gap) else 0.0,
            "cap_ceiling_gap_mean": float(cap_gap.mean()) if len(cap_gap) else 0.0,
            "cap_ceiling_gap_max": float(cap_gap.max()) if len(cap_gap) else 0.0,
            "cap_ceiling_gap_rel_next": (None if (not len(cap_gap)) else (None if float(cap_no_fc[-1]) <= 1e-12 else float(cap_gap[-1]) / float(cap_no_fc[-1]))),
            "integrity_max_no_fc_next": float(int_no_fc[-1]) if len(int_no_fc) else 0.0,
            "integrity_max_eff_next": float(int_eff[-1]) if len(int_eff) else 0.0,
            "integrity_ceiling_gap_next": float(int_gap[-1]) if len(int_gap) else 0.0,
            "integrity_ceiling_gap_mean": float(int_gap.mean()) if len(int_gap) else 0.0,
            "integrity_ceiling_gap_max": float(int_gap.max()) if len(int_gap) else 0.0,
            "integrity_ceiling_gap_rel_next": (None if (not len(int_gap)) else (None if float(int_no_fc[-1]) <= 1e-12 else float(int_gap[-1]) / float(int_no_fc[-1]))),

            "foreclosed_self_mass_mean": float(fc.mean()) if len(fc) else 0.0,
            "recovered_total": float(rec.sum()),
            "recovery_events": int((rec > 0.0).sum()),
            "score_mean": float(sc.mean()) if len(sc) else 0.0,
            "score_slope_per_ep": _slope(xs, sc) if len(sc) else 0.0,
            "bank_mean": float(bk.mean()) if len(bk) else 0.0,
            "bank_slope_per_ep": _slope(xs, bk) if len(bk) else 0.0,
            "net_delta_mean": float(nd.mean()) if len(nd) else 0.0,
            "net_delta_slope_per_ep": _slope(xs, nd) if len(nd) else 0.0,
            "trigger_rate": float(tr.mean()) if len(tr) else 0.0,
            "trigger_slope_per_ep": _slope(xs, tr) if len(tr) else 0.0,
            "budget_left_end": float(bl[-1]) if len(bl) else 0.0,
        })

    # v11.2: paste-friendly outlook training diagnostics (what the head actually saw)
    outlook_train = {
        "enabled": bool(outlook_enabled),
        "label_mode": str(outlook_label_mode),
        "target": str(cfg.get("outlook_target", "score")),
        "K": int(outlook_K),
        "thr": (None if not outlook_enabled else float(outlook_thr_cached)),
        "n": 0,
        "pos": 0,
        "pos_rate": 0.0,
        "brier": None,
        "ce": None,
    }
    try:
        if 'outlook_stats' in locals() and outlook_stats.get('n', 0) > 0:
            n = int(outlook_stats['n'])
            pos = int(outlook_stats['pos'])
            outlook_train.update({
                "n": n,
                "pos": pos,
                "pos_rate": float(pos) / float(n),
                "brier": float(outlook_stats['brier']) / float(n),
                "ce": float(outlook_stats['ce']) / float(n),
            })
    except Exception:
        pass

    out = {
        "config": cfg,
        "runtime_sec": time.time() - t0,
        "total_steps": int(total_steps),
        "metrics": metrics,
        "ablation": abla,
        "counterfactual": cf,
        "endowment": endow_summary,
        "outlook_train": outlook_train,
        "endowment_series": endowment_series,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Done.")
    print("Summary written to:", summary_path)
    print(json.dumps(out["metrics"], ensure_ascii=False, indent=2))
    print("Ablation:", out["ablation"])
    print("Counterfactual ok:", out["counterfactual"].get("ok", False))

    # v11.2: convenient copy/paste block for sharing results
    if bool(cfg.get("print_paste_block", True)):
        try:
            vtag = str(cfg.get('version', 'unknown'))
            print(f"\n=== Paste Block ({vtag}) ===")
            print(f"_cfg={cfg.get('_resolved_cfg_path', '?')}")
            m = out.get("metrics", {})
            a = out.get("ablation", {})
            ot = out.get("outlook_train", {})
            es = out.get("endowment", {})
            print(f"pred_loss={m.get('mean_loss_pred')}  total_loss={m.get('mean_loss_total')}")
            if a.get('ok', False):
                print(f"ablation_delta_ratio={a.get('delta_ratio')}  on={a.get('mean_pred_loss_on')}  off={a.get('mean_pred_loss_off')}")
            print(f"episode_death_rate={m.get('episode_death_rate')}  death_rate={m.get('death_rate')}  mean_ttd={m.get('mean_time_to_death')}")
            print(f"endowment_start={es.get('endowment_start')}  endowment_end={es.get('endowment_end')}  switches_mean={es.get('switches_mean')}  drift_mean={es.get('self_slow_drift_mean')}")
            if cfg.get('self_slow_carry', 0.0) or cfg.get('cf_touch_scale', 0.0):
                print(f"self_slow_carry={cfg.get('self_slow_carry', 0.0)}  cf_touch_scale={cfg.get('cf_touch_scale', 0.0)}  cf_touch_alpha={cfg.get('cf_touch_alpha', 2.0)}  cf_touch_beta={cfg.get('cf_touch_beta', 1.0)}")
            if es.get('foreclosed_self_mass_end') is not None:
                print(f"foreclosed_mass_end={es.get('foreclosed_self_mass_end')}  cf_trace_count={es.get('cf_trace_count')}  cf_trace_total={es.get('cf_trace_total')}")
                if es.get('cap_max_eff_next') is not None:
                    def _fmt(x):
                        try:
                            return f"{float(x):.6g}"
                        except Exception:
                            return str(x)
                    print(
                        "ceiling_next: "
                        f"foreclosed_factor={_fmt(es.get('foreclosed_factor_next'))} "
                        f"cap_no_fc={_fmt(es.get('cap_max_no_fc_next'))} cap_eff={_fmt(es.get('cap_max_eff_next'))} cap_gap={_fmt(es.get('cap_ceiling_gap_next'))} "
                        f"cap_gap_rel={_fmt(es.get('cap_ceiling_gap_rel_next'))} "
                        f"int_no_fc={_fmt(es.get('integrity_max_no_fc_next'))} int_eff={_fmt(es.get('integrity_max_eff_next'))} int_gap={_fmt(es.get('integrity_ceiling_gap_next'))} "
                        f"int_gap_rel={_fmt(es.get('integrity_ceiling_gap_rel_next'))}"
                    )
                    # v12.6b: helpful aggregate stats for quick monitoring
                    try:
                        print(
                            "ceiling_gap_stats: "
                            f"cap_gap_mean={_fmt(es.get('cap_ceiling_gap_mean'))}  cap_gap_max={_fmt(es.get('cap_ceiling_gap_max'))}  "
                            f"int_gap_mean={_fmt(es.get('integrity_ceiling_gap_mean'))}  int_gap_max={_fmt(es.get('integrity_ceiling_gap_max'))}"
                        )
                    except Exception:
                        pass

            if es.get('cf_touch_norm') is not None:
                print(f"cf_touch_norm={es.get('cf_touch_norm')}  self_slow_carry={cfg.get('self_slow_carry', 0.0)}")
            if ot.get('enabled', False):
                print(f"outlook_enabled=True  mode={ot.get('label_mode')}  K={ot.get('K')}  thr={ot.get('thr')}")
                print(f"outlook_labels: n={ot.get('n')}  pos_rate={ot.get('pos_rate')}  brier={ot.get('brier')}  ce={ot.get('ce')}")
            else:
                print("outlook_enabled=False")
        except Exception:
            pass

if __name__ == "__main__":
    main()
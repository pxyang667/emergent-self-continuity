import json
from pathlib import Path

def main():
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    summary_path = logs_dir / "summary.json"
    if not summary_path.exists():
        print("No summary found:", summary_path)
        return
    with open(summary_path, "r", encoding="utf-8") as f:
        out = json.load(f)

    endow = out.get("endowment", {})
    series = out.get("endowment_series", [])

    print("=== Endowment Summary (v10.4) ===")
    keys = [
        "enabled","n",
        "endowment_start","endowment_end","endowment_min","endowment_max","endowment_slope_per_ep",
        "switches_mean","switches_slope_per_ep",
        "self_slow_drift_mean","self_slow_drift_slope_per_ep",
        "recovered_total","recovery_events","budget_left_end",
        "score_mean","score_slope_per_ep",
        "bank_mean","bank_slope_per_ep",
        "net_delta_mean","net_delta_slope_per_ep",
        "trigger_rate","trigger_slope_per_ep",
    ]
    for k in keys:
        if k in endow:
            print(f"{k}: {endow[k]}")

    if series:
        print("\nLast 5 episodes:")
        for d in series[-5:]:
            print(d)

if __name__ == "__main__":
    main()

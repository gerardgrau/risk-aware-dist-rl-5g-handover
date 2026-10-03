import pickle
import numpy as np
import pandas as pd
import os

def generate_legacy_csv():
    pkl_path = "results/simulations/Performance_all.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, "rb") as f:
        performance_all = pickle.load(f)

    # NBS and Max_iter from legacy_simulation.py
    NBS = 21
    Max_iter = 30000
    minutes = 5.0

    data = []
    for perf in performance_all:
        # Extract metrics
        # Performance["Capacity"] is an array, we want its mean
        cap = np.mean(perf["Capacity"])
        rlf = perf["RL_problems"]
        ho = perf["Number_HO"]
        pp = perf["Number_ping_pongs"]
        rel = perf["Reliability"]
        res = perf["Resource_reservation"]
        hof = perf["HOF"]
        
        # prep_rate derivation:
        # Resource_reservation = (np.sum(ReservedBSSectors) / (NBS * Max_iter)) * 100
        # prep_rate = (np.sum(ReservedBSSectors) / 10.0) / minutes
        sum_reserved = (res / 100.0) * NBS * Max_iter
        prep = (sum_reserved / 10.0) / minutes
        
        data.append({
            "ho_rate": ho,
            "hof_rate": hof,
            "pp_rate": pp,
            "capacity_avg": cap,
            "rlf_rate": rlf,
            "reliability_pct": rel,
            "prep_rate": prep,
            "res_reservation_pct": res
        })

    df = pd.DataFrame(data)
    
    # Define metric order
    metric_order = [
        "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
        "reliability_pct", "prep_rate", "res_reservation_pct"
    ]
    
    summary = []
    # Calculate stats for the main metrics
    for col in metric_order:
        if col in df.columns:
            summary.append({
                "metric": col,
                "mean": df[col].mean(),
                "std": df[col].std()
            })

    summary_df = pd.DataFrame(summary)
    
    # Sort according to standard order
    summary_df['metric'] = pd.Categorical(summary_df['metric'], categories=metric_order, ordered=True)
    summary_df = summary_df.sort_values('metric')
    
    output_path = "results/final_metrics/legacy_baseline_summary.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"Final legacy metrics saved to {output_path}")
    print(summary_df)

if __name__ == "__main__":
    generate_legacy_csv()

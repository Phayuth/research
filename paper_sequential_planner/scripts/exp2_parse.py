import json
import numpy as np
import os
import glob


def load_all_logs_from_folder(folder_path):
    """Load all JSON log files from the specified folder."""
    json_pattern = os.path.join(folder_path, "robotsp_solver_log_*.json")
    json_files = glob.glob(json_pattern)

    logs = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                log_data = json.load(f)
                logs.append(log_data)
            print(f"Loaded: {os.path.basename(json_file)}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"\nTotal logs loaded: {len(logs)}")
    return logs


def summarize_logs(logs):
    keys = [
        "tspace_tour_solvetime",
        "cspace_candidate_num",
        "cspace_optimal_config_cost",
        "cspace_optimal_config_selection_solvetime",
        "cspace_collisionfree_tour_total_cost",
        "cspace_collisionfree_tour_solvetime",
        "total_solvetime",
    ]

    stats = {}
    for key in keys:
        values = [log[key] for log in logs if key in log and log[key] is not None]
        if values:
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        else:
            stats[key] = {
                "mean": 0.0,
                "std": 0.0,
                "count": 0,
                "min": 0.0,
                "max": 0.0,
            }

    return stats


# Load all logs from case folder
case_folder = "case4"
logs = load_all_logs_from_folder(case_folder)

if logs:
    stats = summarize_logs(logs)

    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)

    for k, v in stats.items():
        print(f"{k:45s} mean={v['mean']:.6f}, std={v['std']:.6f}, n={v['count']}")

    print("\n" + "="*80)
    print("LATEX FORMAT")
    print("="*80)

    for k, v in stats.items():
        print(f"{k:45s} ${v['mean']:.2f} \\pm {v['std']:.2f}$, n={v['count']}")

    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    for k, v in stats.items():
        print(f"{k:45s} mean={v['mean']:.6f}, std={v['std']:.6f}, min={v['min']:.6f}, max={v['max']:.6f}, n={v['count']}")

else:
    print("No log files found or loaded successfully.")

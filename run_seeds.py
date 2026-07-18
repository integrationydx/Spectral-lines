import os
import subprocess
import re
import numpy as np
import json

seeds = [42, 43, 44]

results = {
    "phase_4": {
        "pde_residual": [],
        "cnn": []
    },
    "phase_5b": {
        "re_22_gnn": [],
        "re_22_temporal": [],
        "re_30_gnn": [],
        "re_30_temporal": [],
        "re_40_gnn": [],
        "re_40_temporal": []
    }
}

for seed in seeds:
    print(f"================ RUNNING SEED {seed} ================")
    env = os.environ.copy()
    env["PINN_SEED"] = str(seed)
    
    # --- PHASE 4 ---
    print(f"Running phase_4.py with seed {seed}...")
    result_4 = subprocess.run([".venv/bin/python", "phase_4.py"], env=env, capture_output=True, text=True)
    out_4 = result_4.stdout
    
    # Parse Phase 4
    match_res = re.search(r"PDE Residual Correlation :\s*([\-\.\d]+)", out_4)
    match_cnn = re.search(r"DMD\+CNN Correlation\s*:\s*([\-\.\d]+)", out_4)
    
    if match_res and match_cnn:
        results["phase_4"]["pde_residual"].append(float(match_res.group(1)))
        results["phase_4"]["cnn"].append(float(match_cnn.group(1)))
    else:
        print(f"Failed to parse Phase 4 output for seed {seed}")
        print(out_4)
        
    # --- PHASE 5B ---
    print(f"Running phase_5b_generalization.py with seed {seed}...")
    result_5b = subprocess.run([".venv/bin/python", "phase_5b_generalization.py"], env=env, capture_output=True, text=True)
    out_5b = result_5b.stdout
    
    # Parse Phase 5B
    for line in out_5b.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and parts[0].isdigit():
            re_val = parts[0]
            gnn_val = float(parts[3])
            temp_val = float(parts[4])
            
            if re_val == "22":
                results["phase_5b"]["re_22_gnn"].append(gnn_val)
                results["phase_5b"]["re_22_temporal"].append(temp_val)
            elif re_val == "30":
                results["phase_5b"]["re_30_gnn"].append(gnn_val)
                results["phase_5b"]["re_30_temporal"].append(temp_val)
            elif re_val == "40":
                results["phase_5b"]["re_40_gnn"].append(gnn_val)
                results["phase_5b"]["re_40_temporal"].append(temp_val)

print("\n\n================ SUMMARY ================")
summary = {}
for phase, metrics in results.items():
    summary[phase] = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values)
            summary[phase][metric_name] = f"{mean:.4f} \\pm {std:.4f}"
            print(f"{phase} - {metric_name}: {mean:.4f} ± {std:.4f} (values: {values})")
        else:
            print(f"{phase} - {metric_name}: NO DATA")

with open("seed_results.json", "w") as f:
    json.dump(summary, f, indent=4)

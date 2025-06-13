# aggregate_wandb_table.py
import wandb
import pandas as pd

def main():
    # Define projects for both models
    DENSE_PROJECT = "dense-transformer-training"
    SPARSE_PROJECT = "sparse-transformer-training"
    
    print("Aggregating WandB results for statistical analysis...")
    
    # Initialize API
    api = wandb.Api()
    
    # Collect all runs from both projects
    all_runs = []
    
    # Get dense model runs
    try:
        dense_runs = api.runs(DENSE_PROJECT)
        print(f"Found {len(dense_runs)} runs in {DENSE_PROJECT}")
        for r in dense_runs:
            if r.state == "finished":
                all_runs.append(("dense", r))
    except Exception as e:
        print(f"Error accessing {DENSE_PROJECT}: {e}")
    
    # Get sparse model runs
    try:
        sparse_runs = api.runs(SPARSE_PROJECT)
        print(f"Found {len(sparse_runs)} runs in {SPARSE_PROJECT}")
        for r in sparse_runs:
            if r.state == "finished":
                all_runs.append(("sparse", r))
    except Exception as e:
        print(f"Error accessing {SPARSE_PROJECT}: {e}")
    
    print(f"\nTotal finished runs found: {len(all_runs)}")
    
    # Process runs into rows
    rows = []
    for model_type, run in all_runs:
        try:
            # Extract seed from config or run name
            seed = run.config.get("seed")
            if seed is None:
                # Try to extract from run name (e.g., "dense_seed_111")
                if "seed_" in run.name:
                    seed = int(run.name.split("seed_")[1].split("_")[0])
                else:
                    print(f"Warning: Could not extract seed from run {run.name}")
                    continue
            
            # Get final metrics from run summary
            summary = run.summary
            
            row = {
                "model": model_type,
                "seed": seed,
                "run_name": run.name,
                "loss": summary.get("final_val_loss"),
                "ppl": summary.get("final_val_ppl"),
                "speed": summary.get("tokens_per_sec"),
                "memory": summary.get("peak_gpu_mem_MB")
            }
            
            # Check if all required metrics are present
            missing_metrics = [k for k, v in row.items() if v is None and k not in ["run_name"]]
            if missing_metrics:
                print(f"Warning: Run {run.name} missing metrics: {missing_metrics}")
                continue
                
            rows.append(row)
            print(f"✓ {model_type} seed {seed}: loss={row['loss']:.4f}, ppl={row['ppl']:.2f}")
            
        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue
    
    if not rows:
        print("No valid runs found with complete metrics!")
        return
    
    # Create DataFrame and compute statistics
    df = pd.DataFrame(rows)
    print(f"\nProcessed {len(df)} runs total:")
    print(df.groupby("model").size())
    
    # Group by model and compute mean ± std
    table = df.groupby("model").agg(["mean", "std"]).round(3)
    
    print("\n" + "="*60)
    print("STATISTICAL RESULTS (Mean ± Std)")
    print("="*60)
    print(table)
    
    # Create a more readable summary table
    summary_rows = []
    for model in ["dense", "sparse"]:
        model_data = df[df["model"] == model]
        if len(model_data) > 0:
            summary_rows.append({
                "Model": model.capitalize(),
                "N": len(model_data),
                "Loss": f"{model_data['loss'].mean():.3f} ± {model_data['loss'].std():.3f}",
                "Perplexity": f"{model_data['ppl'].mean():.2f} ± {model_data['ppl'].std():.2f}",
                "Speed (tok/s)": f"{model_data['speed'].mean():.0f} ± {model_data['speed'].std():.0f}",
                "Memory (MB)": f"{model_data['memory'].mean():.1f} ± {model_data['memory'].std():.1f}"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR THESIS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save outputs
    table.to_csv("detailed_metrics.csv")
    summary_df.to_csv("summary_metrics.csv", index=False)
    
    # Generate LaTeX table
    latex_table = summary_df.to_latex(index=False, escape=False)
    with open("table_metrics.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\nFiles saved:")
    print(f"- detailed_metrics.csv: Full statistical breakdown")
    print(f"- summary_metrics.csv: Clean summary table")
    print(f"- table_metrics.tex: LaTeX formatted table")
    
    # Check for completeness
    print(f"\n" + "="*60)
    print("COMPLETENESS CHECK")
    print("="*60)
    for model in ["dense", "sparse"]:
        model_data = df[df["model"] == model]
        seeds = sorted(model_data["seed"].tolist())
        print(f"{model.capitalize()}: {len(seeds)} runs with seeds {seeds}")
        if len(seeds) >= 3:
            print(f"✓ {model.capitalize()} has enough runs for statistical analysis")
        else:
            print(f"⚠ {model.capitalize()} needs more runs (recommended: 3+)")

if __name__ == "__main__":
    main()

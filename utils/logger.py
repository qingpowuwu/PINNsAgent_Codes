# utils/logger.py

import json
import os
import time
import numpy as np
from typing import Dict, List, Any
from .formatter import format_mse, format_time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ExperimentLogger:
    """Experiment logger, responsible for saving experiment results and summaries"""
    
    def __init__(self, output_dir: str):
        """
        Initialize experiment logger
        
        Args:
            output_dir: Root path of output directory
        """
        self.output_dir = output_dir
    
    def plot_run_metrics(self, iteration_history: List[Dict], run_output_dir: str, 
                        pde_name: str, run_id: int):
        """
        Plot MSE and run_time vs iterations for a single run
        
        Args:
            iteration_history: List of iteration history records
            run_output_dir: Output directory for this run
            pde_name: PDE name
            run_id: Run ID
        """
        if not iteration_history:
            return
        
        # Extract data
        iter_ids = [record["iter_id"] for record in iteration_history]
        mses = [record["mse"] for record in iteration_history]
        run_times = [record["run_time"] for record in iteration_history]
        
        # Filter out nan values for plotting
        valid_data = [(i, m, t) for i, m, t in zip(iter_ids, mses, run_times) 
                     if not (np.isnan(m) or np.isinf(m))]
        
        if not valid_data:
            print(f"Warning: No valid MSE values to plot for {pde_name} Run {run_id}")
            return
        
        valid_iter_ids, valid_mses, valid_run_times = zip(*valid_data)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'{pde_name} - Run {run_id}', fontsize=14, fontweight='bold')
        
        # Plot MSE vs iterations
        ax1.plot(valid_iter_ids, valid_mses, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('MSE', fontsize=11)
        ax1.set_title('MSE vs Iterations', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_yscale('log')  # Use log scale for MSE
        
        # Add best MSE annotation
        best_idx = np.argmin(valid_mses)
        ax1.plot(valid_iter_ids[best_idx], valid_mses[best_idx], 'r*', markersize=15, 
                label=f'Best MSE: {format_mse(valid_mses[best_idx])}')
        ax1.legend(loc='best', fontsize=10)
        
        # Plot run_time vs iterations (use all run_times)
        ax2.plot(iter_ids, run_times, marker='s', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Run Time (s)', fontsize=11)
        ax2.set_title('Run Time vs Iterations', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add average time annotation
        avg_time = np.mean(run_times)
        ax2.axhline(y=avg_time, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Avg Time: {format_time(avg_time)}s')
        ax2.legend(loc='best', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plot_file = os.path.join(run_output_dir, "run_metrics.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Run metrics plot saved: {plot_file}")
    
    def save_run_summary(self, iteration_history: List[Dict], run_output_dir: str, 
                        pde_name: str, run_id: int):
        """
        Save experiment summary for a single run
        
        Args:
            iteration_history: List of iteration history records
            run_output_dir: Output directory for this run
            pde_name: PDE name
            run_id: Run ID
        """
        if not iteration_history:
            return
        
        # Filter out nan values when finding best iteration
        valid_iterations = [iter_data for iter_data in iteration_history 
                          if not (np.isnan(iter_data["mse"]) or np.isinf(iter_data["mse"]))]
        
        if valid_iterations:
            best_iteration = min(valid_iterations, key=lambda x: x["mse"])
        else:
            # If all MSEs are nan, use first iteration as fallback
            best_iteration = iteration_history[0]
        
        summary = {
            "run_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pde_name": pde_name,
                "run_id": run_id,
                "total_iters": len(iteration_history)
            },
            "best_result": {
                "best_mse": format_mse(best_iteration["mse"]),
                "best_run_time": format_time(best_iteration["run_time"]),
                "best_iter_id": best_iteration["iter_id"]
            },
            "iterations_detail": [
                {
                    "iter_id": iter_data["iter_id"],
                    "mse": format_mse(iter_data["mse"]),
                    "run_time": format_time(iter_data["run_time"]),
                    "config": iter_data["config"]
                } for iter_data in iteration_history
            ]
        }
        
        # Save to run directory
        summary_file = os.path.join(run_output_dir, "run_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Run summary saved: {summary_file}")
        
        # Generate and save plot
        self.plot_run_metrics(iteration_history, run_output_dir, pde_name, run_id)
    
    def save_pde_summary(self, pde_name: str, all_results: Dict):
        """
        Save summary results of all runs for a single PDE
        
        Args:
            pde_name: PDE name
            all_results: Dictionary containing all results for this PDE
        """
        pde_results = all_results.get(pde_name, [])
        if not pde_results:
            return
        
        # Group by run_id
        runs_by_id = {}
        for result in pde_results:
            run_id = result["run_id"]
            if run_id not in runs_by_id:
                runs_by_id[run_id] = []
            runs_by_id[run_id].append(result)
        
        # Extract best result from each run (filter out nan)
        runs_detail = {}
        all_best_mses = []
        all_best_times = []
        
        for run_id, run_results in sorted(runs_by_id.items()):
            # Filter out nan values for this run
            valid_run_results = [r for r in run_results 
                                if not (np.isnan(r["mse"]) or np.isinf(r["mse"]))]
            
            if valid_run_results:
                run_best = min(valid_run_results, key=lambda x: x["mse"])
                all_best_mses.append(run_best["mse"])
                all_best_times.append(run_best["run_time"])
                
                # Only save core information
                runs_detail[f"run_{run_id}"] = {
                    "total_iters": len(run_results),
                    "best_mse": format_mse(run_best["mse"]),
                    "best_run_time": format_time(run_best["run_time"]),
                    "best_iter_id": run_best["iter_id"]
                }
            else:
                # All MSEs are nan for this run
                run_best = run_results[0]  # Use first iteration as fallback
                runs_detail[f"run_{run_id}"] = {
                    "total_iters": len(run_results),
                    "best_mse": format_mse(run_best["mse"]),
                    "best_run_time": format_time(run_best["run_time"]),
                    "best_iter_id": run_best["iter_id"]
                }
        
        # Calculate statistics (only on valid MSEs)
        summary = {
            "pde_info": {
                "pde_name": pde_name,
                "total_runs": len(runs_by_id),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            
            "statistics": {
                "best_mse": {
                    "mean": format_mse(float(np.mean(all_best_mses))) if all_best_mses else "nan",
                    "std": format_mse(float(np.std(all_best_mses))) if all_best_mses else "nan"
                },
                "training_time": {
                    "mean": format_time(float(np.mean(all_best_times))) if all_best_times else 0.0,
                    "std": format_time(float(np.std(all_best_times))) if all_best_times else 0.0
                }
            },
            
            "best_overall": {
                "best_mse": format_mse(min(all_best_mses)) if all_best_mses else "nan",
                "best_run_id": int(np.argmin(all_best_mses) + 1) if all_best_mses else 1
            },
            
            "runs_detail": runs_detail
        }
        
        # Save as JSON file
        summary_file = os.path.join(self.output_dir, f"{pde_name}_pde_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"PDE summary saved: {summary_file}")
    
    def save_experiment_summary(self, all_results: Dict[str, List[Dict]], 
                               args, completed_runs: int = None):
        """
        Save experiment summary to file, supports real-time updates
        
        Args:
            all_results: Dictionary of all results, key is pde_name, value is result list
            args: Command line arguments
            completed_runs: Number of completed runs (optional)
        """
        summary = {
            "experiment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": args.mode,
                "num_iters": args.num_iters,
                "num_runs": args.num_runs,
                "completed_runs": completed_runs or args.num_runs,
                "simulate_new_pde": args.simulate_new_pde,
                "device": args.device,
                "iter": args.iter,
                "seed": args.seed
            },
            "results": {}
        }
        
        for pde_name, results in all_results.items():
            # Group results by run_id
            runs_by_id = {}
            for result in results:
                run_id = result["run_id"]
                if run_id not in runs_by_id:
                    runs_by_id[run_id] = []
                runs_by_id[run_id].append(result)
            
            # Calculate best MSE and corresponding run_time for each run (filter out nan)
            run_best_mses = []
            run_best_times = []
            for run_id, run_results in runs_by_id.items():
                # Filter out nan values
                valid_run_results = [r for r in run_results 
                                    if not (np.isnan(r["mse"]) or np.isinf(r["mse"]))]
                if valid_run_results:
                    run_best = min(valid_run_results, key=lambda x: x["mse"])
                    run_best_mses.append(run_best["mse"])
                    run_best_times.append(run_best["run_time"])
            
            if results:
                # Find global best result (filter out nan)
                valid_results = [r for r in results 
                               if not (np.isnan(r["mse"]) or np.isinf(r["mse"]))]
                
                if valid_results:
                    best_result = min(valid_results, key=lambda x: x["mse"])
                else:
                    best_result = results[0]  # Fallback to first result if all are nan
                
                pde_summary = {
                    "total_iters": len(results),
                    "num_runs": len(runs_by_id),
                    "best_config_results": {
                        "best_mse": format_mse(best_result["mse"]),
                        "best_run_time": format_time(best_result["run_time"]),
                        "best_config": best_result["config"],
                        "best_run_id": best_result["run_id"],
                        "best_iter_id": best_result["iter_id"]
                    },
                    "avg_best_mse": format_mse(sum(run_best_mses) / len(run_best_mses)) if run_best_mses else "nan",
                    "avg_run_time": format_time(sum(run_best_times) / len(run_best_times)) if run_best_times else 0.0,
                    "runs_detail": {}
                }
                
                # Detailed information for each run
                for run_id, run_results in runs_by_id.items():
                    # Filter out nan values for this run
                    valid_run_results = [r for r in run_results 
                                        if not (np.isnan(r["mse"]) or np.isinf(r["mse"]))]
                    
                    if valid_run_results:
                        run_best = min(valid_run_results, key=lambda x: x["mse"])
                    else:
                        run_best = run_results[0]  # Fallback if all nan
                    
                    pde_summary["runs_detail"][f"run_{run_id}"] = {
                        "iters": len(run_results),
                        "best_mse": format_mse(run_best["mse"]),
                        "best_run_time": format_time(run_best["run_time"]),
                        "best_iter_id": run_best["iter_id"],
                        "iterations_detail": [
                            {
                                "iter_id": r["iter_id"],
                                "mse": format_mse(r["mse"]),
                                "run_time": format_time(r["run_time"]),
                                "config": r["config"]
                            } for r in run_results
                        ]
                    }
            else:
                pde_summary = {
                    "total_iters": 0,
                    "num_runs": 0,
                    "best_config_results": {},
                    "runs_detail": {}
                }
            
            summary["results"][pde_name] = pde_summary
        
        # Save detailed summary
        summary_file = os.path.join(self.output_dir, "runs_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save simplified CSV format results
        csv_file = os.path.join(self.output_dir, "results.csv")
        with open(csv_file, 'w') as f:
            f.write("PDE,Run_ID,Iter_ID,MSE,Run_Time,Config\n")
            for pde_name, results in all_results.items():
                for result in results:
                    config_str = json.dumps(result["config"]).replace(',', ';')
                    formatted_mse = format_mse(result['mse'])
                    formatted_time = format_time(result['run_time'])
                    f.write(f"{pde_name},{result['run_id']},{result['iter_id']},{formatted_mse},{formatted_time},\"{config_str}\"\n")
        
        print(f"Runs summary: {summary_file}")
        print(f"CSV results: {csv_file}")
    
    def print_all_pdes_summary(self, all_results: Dict[str, List[Dict]]):
        """
        Print concise table summary of all PDEs and save to .txt file
        
        Args:
            all_results: Dictionary of all results
        """
        # Prepare output content
        lines = []
        lines.append("=" * 120)
        lines.append("📊 ALL PDEs SUMMARY WITH BATCH STATISTICS")
        lines.append("=" * 120)
        lines.append("")
        
        # Header
        header = f"{'PDE Name':<35} {'Best MSE (mean±std)':<30} {'Avg Time (mean±std)':<25} {'Total Runs':<15}"
        lines.append(header)
        lines.append("-" * 120)
        
        for pde_name, results in all_results.items():
            if not results:
                continue
            
            # Group by run_id
            runs_by_id = {}
            for result in results:
                run_id = result["run_id"]
                if run_id not in runs_by_id:
                    runs_by_id[run_id] = []
                runs_by_id[run_id].append(result)
            
            # Extract best result from each run (filter out nan)
            all_best_mses = []
            all_best_times = []
            
            for run_results in runs_by_id.values():
                # Filter out nan values
                valid_run_results = [r for r in run_results 
                                    if not (np.isnan(r["mse"]) or np.isinf(r["mse"]))]
                if valid_run_results:
                    run_best = min(valid_run_results, key=lambda x: x["mse"])
                    all_best_mses.append(run_best["mse"])
                    all_best_times.append(run_best["run_time"])
            
            # Calculate statistics (only on valid MSEs)
            if all_best_mses:
                mse_mean = np.mean(all_best_mses)
                mse_std = np.std(all_best_mses)
                mse_str = f"{mse_mean:.2e}±{mse_std:.2e}"
            else:
                mse_str = "nan±nan"
            
            if all_best_times:
                time_mean = np.mean(all_best_times)
                time_std = np.std(all_best_times)
                time_str = f"{time_mean:.0f}±{time_std:.0f}s"
            else:
                time_str = "0±0s"
            
            num_runs = len(runs_by_id)
            
            row = f"{pde_name:<35} {mse_str:<30} {time_str:<25} {num_runs:<15}"
            lines.append(row)
        
        lines.append("-" * 120)
        lines.append(f"\nTotal PDEs: {len(all_results)}\n")
        lines.append("=" * 120)
        
        # Print to terminal
        summary_text = "\n".join(lines)
        print(f"\n{summary_text}\n")
        
        # Save to file
        summary_file = os.path.join(self.output_dir, "all_pdes_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"All PDEs summary saved: {summary_file}\n")
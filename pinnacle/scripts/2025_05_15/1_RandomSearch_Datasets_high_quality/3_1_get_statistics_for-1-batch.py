import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_pde_order():
    """Define the display order of PDEs"""
    return [
        # 1D PDEs
        'burgers1d',
        'wave1d',
        # 2D PDEs  
        'burgers2d',
        'wave2d_heterogeneous',
        'heat2d_complexgeometry',
        'ns2d_backstep',
        'grayscottequation',
        'heat2d_multiscale',
        'heat2d_varyingcoef',
        'poisson2d_manyarea',
        # 3D PDEs
        'poisson3d_complexgeometry',
        # ND PDEs
        'poissonnd',
        'heatnd',
        # Extra PDEs to ignore
        'ns2d_liddriven',
        'poisson2d_classic',
        'poissonboltzmann2d',
    ]

def sort_pdes(pde_names):
    """Sort PDE names according to predefined order"""
    order = get_pde_order()
    
    # Split into two groups: those in order and those not in order
    ordered_pdes = []
    other_pdes = []
    
    for pde in pde_names:
        if pde in order:
            ordered_pdes.append(pde)
        else:
            other_pdes.append(pde)
    
    # Sort ordered_pdes according to the order list
    ordered_pdes.sort(key=lambda x: order.index(x))
    
    # Sort others alphabetically
    other_pdes.sort()
    
    # Merge the two groups
    return ordered_pdes + other_pdes

def analyze_batch_results(batch_path):
    """Analyze batch results, calculate minimum MSE and average runtime for each PDE"""
    
    results = {}
    
    # Get all PDE directories
    pde_dirs = [d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))]
    
    for pde_name in pde_dirs:
        pde_path = os.path.join(batch_path, pde_name)
        print(f"Processing {pde_name}...")
        
        # Get all experiment directories
        exp_dirs = [d for d in os.listdir(pde_path) if os.path.isdir(os.path.join(pde_path, d))]
        
        mse_values = []
        run_times = []
        
        for exp_dir in exp_dirs:
            # Check if final_results.csv exists
            results_file = os.path.join(pde_path, exp_dir, "0-0", "final_results.csv")
            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)
                    if not df.empty:
                        mse = df['2_mse'].iloc[0]
                        run_time = df['0_run_time'].iloc[0]
                        
                        mse_values.append(mse)
                        run_times.append(run_time)
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
        
        if mse_values:
            # Filter out nan and inf values
            valid_indices = [i for i, mse in enumerate(mse_values) 
                           if not (np.isnan(mse) or np.isinf(mse))]
            valid_mse_values = [mse_values[i] for i in valid_indices]
            valid_run_times = [run_times[i] for i in valid_indices]
            
            results[pde_name] = {
                'min_mse': min(valid_mse_values) if valid_mse_values else np.nan,
                'avg_run_time': np.mean(valid_run_times) if valid_run_times else np.nan,
                'num_completed': len(mse_values),
                'num_valid': len(valid_mse_values),
                'all_mse': mse_values,
                'all_run_times': run_times
            }
            print(f"  - {len(mse_values)} completed experiments")
            print(f"  - {len(valid_mse_values)} valid experiments (non-nan)")
            if valid_mse_values:
                print(f"  - Min MSE: {min(valid_mse_values):.6e}")
                print(f"  - Avg run time: {np.mean(valid_run_times):.2f}s")
        else:
            print(f"  - No completed experiments found")
    
    return results

def save_results_to_txt(results, batch_path):
    """Save results to txt file"""
    
    output_file = os.path.join(batch_path, "batch_analysis_results.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Batch Analysis Results\n")
        f.write(f"Batch Path: {batch_path}\n")
        f.write("=" * 60 + "\n\n")
        
        # Sort PDEs by custom order
        sorted_pdes = sort_pdes(list(results.keys()))
        
        # Detailed results
        f.write("DETAILED RESULTS BY PDE:\n")
        f.write("-" * 40 + "\n")
        
        for pde_name in sorted_pdes:
            data = results[pde_name]
            f.write(f"\n{pde_name}:\n")
            f.write(f"  Completed experiments: {data['num_completed']}\n")
            if not np.isnan(data['min_mse']):
                f.write(f"  Minimum MSE: {data['min_mse']:.6e}\n")
                f.write(f"  Average run time: {data['avg_run_time']:.2f} seconds\n")
            else:
                f.write(f"  Minimum MSE: nan (no valid experiments)\n")
                f.write(f"  Average run time: nan\n")
            f.write(f"  All MSE values: {[f'{mse:.6e}' for mse in data['all_mse']]}\n")
            f.write(f"  All run times: {[f'{t:.2f}s' for t in data['all_run_times']]}\n")
        
        # Summary table
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("SUMMARY TABLE:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'PDE Name':<25} {'Min MSE':<15} {'Avg Runtime':<15} {'Valid':<8} {'Completed':<10}\n")
        f.write("-" * 73 + "\n")
        
        for pde_name in sorted_pdes:
            data = results[pde_name]
            if not np.isnan(data['min_mse']):
                mse_str = f"{data['min_mse']:.6e}"
                runtime_str = f"{data['avg_run_time']:.2f}"
            else:
                mse_str = "nan"
                runtime_str = "nan"
            f.write(f"{pde_name:<25} {mse_str:<15} {runtime_str:<15} {data['num_valid']:<8} {data['num_completed']:<10}\n")
        
        # Overall statistics
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("OVERALL STATISTICS:\n")
        f.write("=" * 60 + "\n")
        
        all_min_mses = [data['min_mse'] for data in results.values() if not np.isnan(data['min_mse'])]
        all_avg_runtimes = [data['avg_run_time'] for data in results.values() if not np.isnan(data['avg_run_time'])]
        total_completed = sum(data['num_completed'] for data in results.values())
        total_valid = sum(data['num_valid'] for data in results.values())
        
        if all_min_mses:
            f.write(f"Total PDEs analyzed: {len(results)}\n")
            f.write(f"Total completed experiments: {total_completed}\n")
            f.write(f"Total valid experiments: {total_valid}\n")
            f.write(f"Best overall MSE: {min(all_min_mses):.6e}\n")
            f.write(f"Worst best MSE: {max(all_min_mses):.6e}\n")
            f.write(f"Average of minimum MSEs: {np.mean(all_min_mses):.6e}\n")
            f.write(f"Average runtime across all PDEs: {np.mean(all_avg_runtimes):.2f} seconds\n")
            total_runtime = sum(data['avg_run_time'] * data['num_valid'] 
                              for data in results.values() if not np.isnan(data['avg_run_time']))
            f.write(f"Total runtime for valid experiments: {total_runtime:.2f} seconds\n")
    
    print(f"\nResults saved to: {output_file}")

def main():
    batch_path = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/output/2025_05_15/1_RandomSearch_Datasets_high_quality/1_ICML_2025_configs/Batch_1"
    
    print(f"Analyzing batch: {batch_path}")
    
    if not os.path.exists(batch_path):
        print(f"Error: Batch path does not exist: {batch_path}")
        return
    
    # Analyze results
    results = analyze_batch_results(batch_path)
    
    if not results:
        print("No completed experiments found!")
        return
    
    # Save results
    save_results_to_txt(results, batch_path)
    
    # Print summary
    print("\n" + "="*60)
    print("QUICK SUMMARY:")
    print("="*60)
    sorted_pdes = sort_pdes(list(results.keys()))
    for pde_name in sorted_pdes:
        data = results[pde_name]
        if not np.isnan(data['min_mse']):
            print(f"{pde_name:<25} | Min MSE: {data['min_mse']:.6e} | Avg Time: {data['avg_run_time']:.2f}s | Valid: {data['num_valid']} | Completed: {data['num_completed']}")
        else:
            print(f"{pde_name:<25} | Min MSE: nan | Avg Time: nan | Valid: {data['num_valid']} | Completed: {data['num_completed']}")

if __name__ == "__main__":
    main()
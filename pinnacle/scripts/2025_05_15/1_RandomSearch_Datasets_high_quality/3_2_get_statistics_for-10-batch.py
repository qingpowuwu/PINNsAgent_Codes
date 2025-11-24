import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

def format_scientific(value, decimals=2):
    """格式化科学计数法，保留指定小数位数"""
    if value == 0:
        return "0.00e+00"
    return f"{value:.{decimals}e}"

def format_time(value, decimals=0):
    """格式化时间，保留指定小数位数"""
    return f"{value:.{decimals}f}"

def get_pde_order():
    """定义PDE的显示顺序"""
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
        'heatnd'
    ]

def sort_pdes_by_custom_order(pde_dict):
    """按照自定义顺序排序PDEs"""
    order = get_pde_order()
    sorted_pdes = []
    
    # 首先添加按顺序排列的PDEs
    for pde_name in order:
        if pde_name in pde_dict:
            sorted_pdes.append(pde_name)
    
    # 然后添加任何不在预定义顺序中的PDEs
    for pde_name in sorted(pde_dict.keys()):
        if pde_name not in sorted_pdes:
            sorted_pdes.append(pde_name)
    
    return sorted_pdes

def analyze_single_batch(batch_path):
    """分析单个batch的结果"""
    results = {}
    
    if not os.path.exists(batch_path):
        print(f"Warning: Batch path does not exist: {batch_path}")
        return results
    
    # 获取所有PDE目录
    pde_dirs = [d for d in os.listdir(batch_path) 
                if os.path.isdir(os.path.join(batch_path, d)) 
                and not d.endswith('.txt') and not d.endswith('.py')]
    
    for pde_name in pde_dirs:
        pde_path = os.path.join(batch_path, pde_name)
        
        # 获取所有实验目录
        exp_dirs = [d for d in os.listdir(pde_path) if os.path.isdir(os.path.join(pde_path, d))]
        
        mse_values = []
        run_times = []
        
        for exp_dir in exp_dirs:
            results_file = os.path.join(pde_path, exp_dir, "0-0", "final_results.csv")
            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)
                    if not df.empty:
                        mse = df['2_mse'].iloc[0]
                        run_time = df['0_run_time'].iloc[0]
                        
                        # 检查是否为有效数值
                        if pd.notna(mse) and pd.notna(run_time) and np.isfinite(mse) and np.isfinite(run_time):
                            mse_values.append(mse)
                            run_times.append(run_time)
                        else:
                            print(f"  Skipping invalid data in {results_file}: MSE={mse}, Time={run_time}")
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
        
        if mse_values:
            results[pde_name] = {
                'min_mse': min(mse_values),
                'avg_run_time': np.mean(run_times),
                'num_completed': len(mse_values),
                'all_mse': mse_values,
                'all_run_times': run_times
            }
    
    return results

def analyze_all_batches(base_path, num_batches=10):
    """分析所有batch的结果"""
    
    all_batch_results = {}
    
    print("Analyzing all batches...")
    print("=" * 60)
    
    for batch_num in range(1, num_batches + 1):
        batch_path = os.path.join(base_path, f"Batch_{batch_num}")
        print(f"Processing Batch_{batch_num}...")
        
        batch_results = analyze_single_batch(batch_path)
        all_batch_results[f"Batch_{batch_num}"] = batch_results
        
        if batch_results:
            print(f"  Found {len(batch_results)} PDEs with completed experiments")
            # 显示每个PDE的有效实验数量
            for pde_name, data in batch_results.items():
                print(f"    {pde_name}: {data['num_completed']} valid experiments")
        else:
            print(f"  No results found")
    
    return all_batch_results

def calculate_pde_statistics(all_batch_results, target_runs=5):
    """计算每个PDE的统计信息，过滤掉包含NaN的结果"""
    
    # 收集所有PDE名称
    all_pde_names = set()
    for batch_results in all_batch_results.values():
        all_pde_names.update(batch_results.keys())
    
    # 分类统计
    exact_runs_stats = {}  # 正好target_runs次的
    all_runs_stats = {}    # 所有runs的
    
    print(f"\nCalculating statistics for PDEs...")
    print("=" * 60)
    
    for pde_name in sorted(all_pde_names):
        print(f"Processing {pde_name}...")
        
        # 收集所有batch中该PDE的数据
        exact_runs_data = []  # 正好target_runs次的batch数据
        all_runs_data = []    # 所有batch数据
        
        exact_runs_min_mses = []
        exact_runs_avg_times = []
        
        all_runs_min_mses = []
        all_runs_avg_times = []
        
        batch_run_counts = []  # 记录每个batch的运行次数
        
        for batch_name, batch_results in all_batch_results.items():
            if pde_name in batch_results:
                data = batch_results[pde_name]
                num_runs = data['num_completed']
                min_mse = data['min_mse']
                avg_time = data['avg_run_time']
                
                # 检查数据有效性
                if pd.notna(min_mse) and pd.notna(avg_time) and np.isfinite(min_mse) and np.isfinite(avg_time):
                    batch_run_counts.append(num_runs)
                    
                    # 所有runs的统计
                    all_runs_min_mses.append(min_mse)
                    all_runs_avg_times.append(avg_time)
                    all_runs_data.append({
                        'batch': batch_name,
                        'min_mse': min_mse,
                        'avg_run_time': avg_time,
                        'num_runs': num_runs
                    })
                    
                    # 正好target_runs次的统计
                    if num_runs == target_runs:
                        exact_runs_min_mses.append(min_mse)
                        exact_runs_avg_times.append(avg_time)
                        exact_runs_data.append({
                            'batch': batch_name,
                            'min_mse': min_mse,
                            'avg_run_time': avg_time,
                            'num_runs': num_runs
                        })
                else:
                    print(f"  Skipping invalid data from {batch_name}: MSE={min_mse}, Time={avg_time}")
        
        # 计算正好target_runs次的统计
        if exact_runs_min_mses and len(exact_runs_min_mses) > 0:
            exact_runs_stats[pde_name] = {
                'min_mse_mean': np.mean(exact_runs_min_mses),
                'min_mse_std': np.std(exact_runs_min_mses, ddof=1) if len(exact_runs_min_mses) > 1 else 0,
                'avg_time_mean': np.mean(exact_runs_avg_times),
                'avg_time_std': np.std(exact_runs_avg_times, ddof=1) if len(exact_runs_avg_times) > 1 else 0,
                'num_batches': len(exact_runs_min_mses),
                'data': exact_runs_data
            }
        
        # 计算所有runs的统计
        if all_runs_min_mses and len(all_runs_min_mses) > 0:
            all_runs_stats[pde_name] = {
                'min_mse_mean': np.mean(all_runs_min_mses),
                'min_mse_std': np.std(all_runs_min_mses, ddof=1) if len(all_runs_min_mses) > 1 else 0,
                'avg_time_mean': np.mean(all_runs_avg_times),
                'avg_time_std': np.std(all_runs_avg_times, ddof=1) if len(all_runs_avg_times) > 1 else 0,
                'num_batches': len(all_runs_min_mses),
                'run_counts': batch_run_counts,
                'data': all_runs_data
            }
        
        print(f"  Exact {target_runs} runs: {len(exact_runs_min_mses)} valid batches")
        print(f"  All runs: {len(all_runs_min_mses)} valid batches, run counts: {batch_run_counts}")
    
    return exact_runs_stats, all_runs_stats

def save_comprehensive_results(exact_runs_stats, all_runs_stats, base_path, target_runs=5):
    """保存详细的分析结果"""
    
    output_file = os.path.join(base_path, f"comprehensive_analysis_results_formatted.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE RANDOM SEARCH ANALYSIS (FORMATTED RESULTS)\n")
        f.write(f"Target runs per PDE per batch: {target_runs}\n")
        f.write("=" * 80 + "\n\n")
        
        # 第一部分:正好target_runs次的结果
        f.write("PART 1: PDEs WITH EXACTLY {} RUNS PER BATCH (VALID DATA ONLY)\n".format(target_runs))
        f.write("=" * 73 + "\n\n")
        
        if exact_runs_stats:
            f.write(f"{'PDE Name':<25} {'Min MSE (mean±std)':<20} {'Avg Time (mean±std)':<20} {'Batches':<8}\n")
            f.write("-" * 73 + "\n")
            
            # 按自定义顺序排序
            sorted_pde_names = sort_pdes_by_custom_order(exact_runs_stats)
            
            # 定义分组
            group_1d = ['burgers1d', 'wave1d']
            group_2d = ['burgers2d', 'wave2d_heterogeneous', 'heat2d_complexgeometry', 'ns2d_backstep', 
                       'grayscottequation', 'heat2d_multiscale', 'heat2d_varyingcoef', 'poisson2d_manyarea']
            group_3d = ['poisson3d_complexgeometry']
            group_nd = ['poissonnd', 'heatnd']
            
            # 获取预定义顺序列表
            predefined_order = get_pde_order()
            
            for i, pde_name in enumerate(sorted_pde_names):
                if pde_name in exact_runs_stats:
                    data = exact_runs_stats[pde_name]
                    mse_mean_str = format_scientific(data['min_mse_mean'], 2)
                    mse_std_str = format_scientific(data['min_mse_std'], 2)
                    time_mean_str = format_time(data['avg_time_mean'], 0)
                    time_std_str = format_time(data['avg_time_std'], 0)
                    
                    mse_str = f"{mse_mean_str}±{mse_std_str}"
                    time_str = f"{time_mean_str}±{time_std_str}s"
                    
                    f.write(f"{pde_name:<25} {mse_str:<20} {time_str:<20} {data['num_batches']:<8}\n")
                    
                    # 添加分组分隔线
                    next_pde = sorted_pde_names[i+1] if i+1 < len(sorted_pde_names) else None
                    if next_pde:
                        # 在预定义组之间添加分隔线
                        if (pde_name in group_1d and next_pde in group_2d) or \
                           (pde_name in group_2d and next_pde in group_3d) or \
                           (pde_name in group_3d and next_pde in group_nd):
                            f.write("-" * 73 + "\n")
                        # 在预定义顺序的最后一个PDE和额外PDE之间添加分隔线
                        elif pde_name in predefined_order and next_pde not in predefined_order:
                            f.write("-" * 73 + "\n")
            
            f.write(f"\nTotal PDEs with exactly {target_runs} runs and valid data: {len(exact_runs_stats)}\n")
        else:
            f.write(f"No PDEs found with exactly {target_runs} runs per batch and valid data.\n")
        
        # 第二部分:所有runs的结果
        f.write("\n\n" + "=" * 103 + "\n")
        f.write("PART 2: ALL PDEs WITH VALID DATA (INCLUDING VARYING RUN COUNTS)\n")
        f.write("=" * 103 + "\n\n")
        
        if all_runs_stats:
            f.write(f"{'PDE Name':<25} {'Min MSE (mean±std)':<20} {'Avg Time (mean±std)':<20} {'Batches':<8} {'Run Counts':<30}\n")
            f.write("-" * 103 + "\n")
            
            # 按自定义顺序排序
            sorted_pde_names = sort_pdes_by_custom_order(all_runs_stats)
            
            # 定义分组（与PART 1相同）
            group_1d = ['burgers1d', 'wave1d']
            group_2d = ['burgers2d', 'wave2d_heterogeneous', 'heat2d_complexgeometry', 'ns2d_backstep', 
                       'grayscottequation', 'heat2d_multiscale', 'heat2d_varyingcoef', 'poisson2d_manyarea']
            group_3d = ['poisson3d_complexgeometry']
            group_nd = ['poissonnd', 'heatnd']
            
            # 获取预定义顺序列表
            predefined_order = get_pde_order()
            
            for i, pde_name in enumerate(sorted_pde_names):
                if pde_name in all_runs_stats:
                    data = all_runs_stats[pde_name]
                    mse_mean_str = format_scientific(data['min_mse_mean'], 2)
                    mse_std_str = format_scientific(data['min_mse_std'], 2)
                    time_mean_str = format_time(data['avg_time_mean'], 0)
                    time_std_str = format_time(data['avg_time_std'], 0)
                    
                    mse_str = f"{mse_mean_str}±{mse_std_str}"
                    time_str = f"{time_mean_str}±{time_std_str}s"
                    # 显示完整的运行次数列表,不省略
                    run_counts_str = str(data['run_counts'])
                    
                    f.write(f"{pde_name:<25} {mse_str:<20} {time_str:<20} {data['num_batches']:<8} {run_counts_str:<30}\n")
                    
                    # 添加分组分隔线（与PART 1相同逻辑）
                    next_pde = sorted_pde_names[i+1] if i+1 < len(sorted_pde_names) else None
                    if next_pde:
                        # 在预定义组之间添加分隔线
                        if (pde_name in group_1d and next_pde in group_2d) or \
                           (pde_name in group_2d and next_pde in group_3d) or \
                           (pde_name in group_3d and next_pde in group_nd):
                            f.write("-" * 103 + "\n")
                        # 在预定义顺序的最后一个PDE和额外PDE之间添加分隔线
                        elif pde_name in predefined_order and next_pde not in predefined_order:
                            f.write("-" * 103 + "\n")
            
            f.write(f"\nTotal PDEs with valid data: {len(all_runs_stats)}\n")
        
        # 总体统计（格式化对齐）
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("OVERALL STATISTICS (VALID DATA ONLY)\n")
        f.write("=" * 80 + "\n")
        
        if exact_runs_stats:
            exact_mse_means = [data['min_mse_mean'] for data in exact_runs_stats.values()]
            exact_time_means = [data['avg_time_mean'] for data in exact_runs_stats.values()]
            
            f.write(f"\nFor PDEs with exactly {target_runs} runs:\n")
            f.write(f"  Number of PDEs:        {len(exact_runs_stats):>8}\n")
            f.write(f"  Best average MSE:      {format_scientific(min(exact_mse_means), 2):>12}\n")
            f.write(f"  Worst average MSE:     {format_scientific(max(exact_mse_means), 2):>12}\n")
            f.write(f"  Overall average MSE:   {format_scientific(np.mean(exact_mse_means), 2):>12} ± {format_scientific(np.std(exact_mse_means, ddof=1) if len(exact_mse_means) > 1 else 0, 2)}\n")
            f.write(f"  Average runtime:       {format_time(np.mean(exact_time_means), 2):>12} ± {format_time(np.std(exact_time_means, ddof=1) if len(exact_time_means) > 1 else 0, 2)} seconds\n")
        
        if all_runs_stats:
            all_mse_means = [data['min_mse_mean'] for data in all_runs_stats.values()]
            all_time_means = [data['avg_time_mean'] for data in all_runs_stats.values()]
            
            f.write(f"\nFor all PDEs with valid data:\n")
            f.write(f"  Number of PDEs:        {len(all_runs_stats):>8}\n")
            f.write(f"  Best average MSE:      {format_scientific(min(all_mse_means), 2):>12}\n")
            f.write(f"  Worst average MSE:     {format_scientific(max(all_mse_means), 2):>12}\n")
            f.write(f"  Overall average MSE:   {format_scientific(np.mean(all_mse_means), 2):>12} ± {format_scientific(np.std(all_mse_means, ddof=1) if len(all_mse_means) > 1 else 0, 2)}\n")
            f.write(f"  Average runtime:       {format_time(np.mean(all_time_means), 2):>12} ± {format_time(np.std(all_time_means, ddof=1) if len(all_time_means) > 1 else 0, 2)} seconds\n")
    
    print(f"\nFormatted results saved to: {output_file}")

def main():
    # base_path = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/output/ICML_2025/RandomSearch_Table1"
    base_path = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/output/2025_11_20/1_RandomSearch_Datasets_high_quality/1_ICML_2025_configs"
    
    target_runs = 5  # 期望的每个batch中每个PDE的运行次数
    
    print("Comprehensive Random Search Analysis (Custom Order)")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"Target runs per PDE per batch: {target_runs}")
    
    # 分析所有batch
    all_batch_results = analyze_all_batches(base_path)
    
    if not all_batch_results:
        print("No batch results found!")
        return
    
    # 计算统计信息
    exact_runs_stats, all_runs_stats = calculate_pde_statistics(all_batch_results, target_runs)
    
    # 保存结果
    save_comprehensive_results(exact_runs_stats, all_runs_stats, base_path, target_runs)
    
    # 打印快速汇总（按自定义顺序）
    print("\n" + "="*75)
    print(f"QUICK SUMMARY (Exact {target_runs} runs, custom order):")
    print("="*75)
    if exact_runs_stats:
        print(f"{'PDE':<25} {'MSE (mean±std)':<20} {'Time (mean±std)':<18} {'Batches':<8}")
        print("-" * 71)
        
        sorted_pde_names = sort_pdes_by_custom_order(exact_runs_stats)
        for pde_name in sorted_pde_names:
            if pde_name in exact_runs_stats:
                data = exact_runs_stats[pde_name]
                mse_str = f"{format_scientific(data['min_mse_mean'], 2)}±{format_scientific(data['min_mse_std'], 2)}"
                time_str = f"{format_time(data['avg_time_mean'], 0)}±{format_time(data['avg_time_std'], 0)}s"
                print(f"{pde_name:<25} {mse_str:<20} {time_str:<18} {data['num_batches']:<8}")
    else:
        print(f"No PDEs with exactly {target_runs} runs and valid data found.")
    
    # 打印总体统计（格式化对齐）
    if exact_runs_stats:
        exact_mse_means = [data['min_mse_mean'] for data in exact_runs_stats.values()]
        exact_time_means = [data['avg_time_mean'] for data in exact_runs_stats.values()]
        
        print(f"\nOVERALL STATISTICS (Exact {target_runs} runs):")
        print(f"  Number of PDEs:        {len(exact_runs_stats):>8}")
        print(f"  Best average MSE:      {format_scientific(min(exact_mse_means), 2):>12}")
        print(f"  Worst average MSE:     {format_scientific(max(exact_mse_means), 2):>12}")
        print(f"  Overall average MSE:   {format_scientific(np.mean(exact_mse_means), 2):>12} ± {format_scientific(np.std(exact_mse_means, ddof=1) if len(exact_mse_means) > 1 else 0, 2)}")
        print(f"  Average runtime:       {format_time(np.mean(exact_time_means), 2):>12} ± {format_time(np.std(exact_time_means, ddof=1) if len(exact_time_means) > 1 else 0, 2)} seconds")

if __name__ == "__main__":
    main()
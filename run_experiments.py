# src/run_experiments.py

import argparse
import os
import sys
import time

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retriever import PGKR
from database.knowledge_base import KnowledgeBase
from database.memory_tree import MemoryTree
from agents.programmer import Programmer
from agents.planner import Planner
from utils.llm_client import LLMClient
from utils.config_loader import ConfigLoader
from utils.logger import ExperimentLogger
from utils.formatter import format_mse, format_time
from database.pde_encoder import PDE_LABELS

def parse_args():
    parser = argparse.ArgumentParser(description="PINNsAgent experiment script")
    # Basic experiment settings
    parser.add_argument('--mode', type=str, choices=['random', 'llm'], default='random',
                       help='Optimization mode: random or llm')
    # Prompt strategy arguments
    parser.add_argument('--prompt_strategy', type=str, default='zero_shot',
                       choices=['zero_shot', 'full_history', 'memory_tree', 'pgkr', 'pinns_agent'],
                       help='Prompt strategy for LLM mode (default: zero_shot)')
    # PGKR arguments for pgkr and pinns_agent strategies
    parser.add_argument('--use_pgkr', action='store_true',
                       help='Enable PGKR (PDE-Guided Knowledge Retrieval) for pinns_agent strategy')
    parser.add_argument('--use_memory_tree', action='store_true',
                       help='Enable MemoryTree exploration scores for pinns_agent strategy')
    parser.add_argument('--pgkr_top_k', type=int, default=1,
                       help='Number of similar PDEs to retrieve from knowledge base (default: 1)')
    parser.add_argument('--use_composite_score', action='store_true',
                       help='Use composite score (MSE + runtime) for PGKR best config selection')
    
    # NEW: UCT-related arguments
    parser.add_argument('--use_uct', action='store_true',
                       help='Use UCT (Upper Confidence Bound) scores instead of static exploration scores')
    parser.add_argument('--uct_lambda', type=float, default=1.4,
                       help='UCT exploration weight lambda (default: 1.4)')
    
    parser.add_argument('--pde_type', type=str, choices=['1d', '2d', '3d', 'nd'], default=None,
                       help='PDE dimension type, choose one between --pde_type and --pde_name, cannot specify both')
    parser.add_argument('--pde_name', type=str, default=None,
                       help='Specify single PDE name, choose one between --pde_type and --pde_name, cannot specify both')
    parser.add_argument('--num_iters', type=int, default=5,
                       help='Number of optimization iterations per PDE')
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./outputs/experiments',
                       help='Experiment relative directory output path for both agent logs and PINNacle results')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Experiment run name, used to distinguish different experiments')
    # Training settings
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--iter', type=int, default=20000, help='Training iteration count')
    parser.add_argument('--seed', type=int, default=44, help='Random seed')
    # Path settings
    parser.add_argument('--config_path', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--csv_path', type=str, 
                       default='./project-2364_1282-filtered-all_pdes_105-top_mse_rows.csv',
                       help='Knowledge base CSV file path')
    parser.add_argument('--train_code_dir', type=str, default="./pinnacle",
                       help='Training code directory (where benchmark.py is located)')
    parser.add_argument('--conda_python', type=str, default="python",
                       help='Conda environment Python executable path')
    # Repeated experiment settings
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of repeated runs per PDE')
    # New PDE scenario simulation
    parser.add_argument('--simulate_new_pde', action='store_true',
                       help='Simulate new PDE scenario: do not use this PDE\'s historical records for retrieval')
    # Knowledge base save control
    parser.add_argument('--save_kb', action='store_true',
                       help='Save updated knowledge base to CSV file after experiments')
    parser.add_argument('--kb_save_path', type=str, default=None,
                       help='Custom path to save knowledge base (default: overwrite original CSV)')
    # Verbose control
    parser.add_argument('--verbose_llm', action='store_true',
                       help='Whether to print LLM interaction process (including retry information)')
    parser.add_argument('--verbose_training', action='store_true',
                       help='Whether to print detailed output of PINNacle training process')
    
    args = parser.parse_args()
    
    # Validate mutual exclusivity of pde_type and pde_name
    if args.pde_type is None and args.pde_name is None:
        parser.error("Must specify either --pde_type or --pde_name")
    if args.pde_type is not None and args.pde_name is not None:
        parser.error("Cannot specify both --pde_type and --pde_name, please choose only one")

    # Validate PGKR/MemoryTree arguments - only valid for pinns_agent strategy
    if args.prompt_strategy != "pinns_agent" and args.use_pgkr:
        parser.error("--use_pgkr can only be used with --prompt_strategy pinns_agent")
    if args.prompt_strategy != "pinns_agent" and args.use_memory_tree:
        parser.error("--use_memory_tree can only be used with --prompt_strategy pinns_agent")
    
    # Validate UCT arguments
    if args.use_uct and not args.use_memory_tree:
        parser.error("--use_uct requires --use_memory_tree to be enabled")
    
    return args

def setup_experiment(args):
    """Set up experiment environment"""
    # Load configuration
    config_loader = ConfigLoader(args.config_path)
    
    # Update fixed parameters using args
    config_loader.update_fixed_params(
        device=args.device,
        iter=args.iter,
        seed=args.seed
    )
    
    # Create output directory (absolute path)
    base_output = os.path.abspath(args.output_dir)
    output_dir = os.path.join(base_output, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return config_loader, output_dir

def initialize_agents(args, config_loader, output_dir):
    """Initialize all agents"""
    # Knowledge base
    kb = KnowledgeBase(args.csv_path)
    
    # Retriever (PGKR)
    pgkr = PGKR()
    
    # Memory Tree (for memory_tree and pinns_agent strategies)
    memory_tree = None
    if args.mode == "llm" and args.prompt_strategy in ["memory_tree", "pinns_agent"]:
        memory_tree = MemoryTree(knowledge_base=kb)
    
    # Planner
    if args.mode == "llm":
        llm_config = config_loader.get_llm_config()
        llm_client = LLMClient(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            model=llm_config["model"]
        )
        # Prepare kwargs for different prompt strategies
        prompt_kwargs = {}
        if args.prompt_strategy == "pinns_agent":
            prompt_kwargs['use_pgkr'] = args.use_pgkr
            prompt_kwargs['use_memory_tree'] = args.use_memory_tree
        
        planner = Planner(
            mode=args.mode, 
            llm_client=llm_client, 
            log_dir=output_dir,
            search_space=config_loader.get_search_space(),
            verbose=args.verbose_llm,
            prompt_strategy=args.prompt_strategy,
            max_iterations=args.num_iters,
            **prompt_kwargs
        )
    else:
        planner = Planner(
            mode=args.mode, 
            log_dir=output_dir,
            search_space=config_loader.get_search_space(),
            verbose=args.verbose_llm,
            max_iterations=args.num_iters,
        )
    
    # Programmer - initialize with basic parameters only, specific output_dir set in each run
    fixed_params = config_loader.get_fixed_params()
    fixed_params.update({
        "name": "iter",  # Initial name, will be updated in each run
        "output_dir": "./outputs/temp",  # Temporary directory, will be updated in each run
        "general_method": "none",
        "loss_weight": "none", 
        "num_test_points": "Default"
    })
    
    programmer = Programmer(
        template_fixed=fixed_params, 
        train_dir=args.train_code_dir,
        conda_python=args.conda_python,
        verbose=args.verbose_training
    )
    
    # Logger
    logger = ExperimentLogger(output_dir)
    
    return kb, pgkr, memory_tree, planner, programmer, logger

def run_single_pde_experiment(pde_name, args, config_loader, kb, pgkr, memory_tree, planner, 
                              programmer, logger, output_dir, run_id=1):
    """Run experiment for a single PDE"""
    print(f"\n{'='*120}")
    # Base info string
    info_str = (f"Experiment: {pde_name} | Run {run_id}/{args.num_runs} | Mode: {args.mode.upper()} | "
                f"Num Iters: {args.num_iters} | Prompt Strategy: {args.prompt_strategy}")
    print(info_str)
    
    # Add strategy-specific info
    if args.mode == "llm":
        if args.prompt_strategy == "pinns_agent":
            # PINNsAgent: show both PGKR and MemoryTree status
            info = f"PGKR: {'ON' if args.use_pgkr else 'OFF'}"
            if args.use_pgkr:
                info += f" (Top-K={args.pgkr_top_k} | Composite Score={args.use_composite_score})"
            info += f" | MemoryTree: {'ON' if args.use_memory_tree else 'OFF'}"
            # NEW: Show UCT status
            if args.use_memory_tree:
                info += f" | UCT: {'ON' if args.use_uct else 'OFF'}"
                if args.use_uct:
                    info += f" (λ={args.uct_lambda})"
            info += f" | Simulate New PDE: {args.simulate_new_pde}"
            print(info)
        elif args.prompt_strategy == "pgkr":
            # pgkr strategy always uses PGKR
            pgkr_info = f"PGKR Top-K: {args.pgkr_top_k} | Simulate New PDE: {args.simulate_new_pde}"
            pgkr_info += f" | Use Composite Score: {args.use_composite_score}"
            print(pgkr_info)
        elif args.prompt_strategy == "memory_tree":
            # memory_tree strategy always uses MemoryTree
            memory_info = f"Use MemoryTree: True | Simulate New PDE: {args.simulate_new_pde}"
            # NEW: Show UCT status for memory_tree strategy
            memory_info += f" | UCT: {'ON' if args.use_uct else 'OFF'}"
            if args.use_uct:
                memory_info += f" (λ={args.uct_lambda})"
            print(memory_info)
    
    print(f"{'='*120}")
    
    # NEW: Reset online visit counts for this PDE before starting new run
    if memory_tree and args.use_uct:
        memory_tree.reset_online_visit_counts(pde_name)
        print(f"\n🔄 Reset online visit counts for {pde_name}")
    
    # Retrieve similar PDEs and their best configurations if using PGKR
    similar_pdes_configs = None
    if args.mode == "llm" and args.prompt_strategy in ["pgkr", "pinns_agent"]:
        # For 'pgkr' strategy: always use PGKR
        # For 'pinns_agent' strategy: use PGKR only if args.use_pgkr is True
        should_use_pgkr = (args.prompt_strategy == "pgkr") or (args.use_pgkr)
        
        if should_use_pgkr:
            print(f"\nRetrieving best configurations from {args.pgkr_top_k} similar PDEs...")
            similar_pdes_configs = pgkr.retrieve_similar_pdes_configs(
                target_pde=pde_name,
                kb=kb,
                pgkr_top_k=args.pgkr_top_k,
                simulate_new_pde=args.simulate_new_pde,
                use_composite_score=args.use_composite_score
            )
            
            if similar_pdes_configs:
                print(f"Retrieved configurations from {len(similar_pdes_configs)} similar PDEs:")
                for pde, info in similar_pdes_configs.items():
                    print(f"  • {pde}: Similarity={info['similarity']:.4f}, Best MSE={info['best_mse']:.2e}")
            else:
                print("Warning: No similar PDEs configurations retrieved")
    
    # Experiment loop
    iteration_history = []
    run_output_dir = os.path.join(output_dir, f"{pde_name}_run_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    for iter_id in range(1, args.num_iters + 1):
        print(f"\n{'─'*80}")
        print(f"Iteration {iter_id}/{args.num_iters}")
        print(f"{'─'*80}")
        
        # NEW: Get exploration scores (static or UCT-based)
        exploration_scores = None
        if args.mode == "llm" and args.prompt_strategy in ["memory_tree", "pinns_agent"] and memory_tree:
            # For 'memory_tree' strategy: always use MemoryTree
            # For 'pinns_agent' strategy: use MemoryTree only if args.use_memory_tree is True
            should_use_memory_tree = (args.prompt_strategy == "memory_tree") or (args.use_memory_tree)
            
            if should_use_memory_tree:
                if args.use_uct:
                    # Use UCT scores (dynamic)
                    print(f"\n🎯 Retrieving UCT scores from MemoryTree (λ={args.uct_lambda})...")
                    exploration_scores = memory_tree.get_uct_scores(
                        pde_name=pde_name,
                        simulate_new_pde=args.simulate_new_pde,
                        lambda_val=args.uct_lambda
                    )
                    score_type = "UCT"
                else:
                    # Use static exploration scores
                    print(f"\n📊 Retrieving static exploration scores from MemoryTree...")
                    exploration_scores = memory_tree.get_scores_for_pde(
                        pde_name=pde_name,
                        simulate_new_pde=args.simulate_new_pde
                    )
                    score_type = "Static"
                
                if exploration_scores:
                    print(f"Top-10 parameters by {score_type} score:")
                    top_5 = list(exploration_scores.items())
                    for param, score in top_5:
                        # Show visit count if using UCT
                        if args.use_uct and pde_name in memory_tree.online_visit_counts:
                            visits = memory_tree.online_visit_counts[pde_name].get(param, 0)
                            print(f"  • [{visits}x] {param}: {score:.3f}")
                        else:
                            print(f"  • {param}: {score:.3f}")
                else:
                    print("Warning: No exploration scores retrieved")
        
        # Generate configuration - pass exploration_scores (UCT or static)
        config = planner.generate_config(
            history=iteration_history,
            pde_name=pde_name,
            run_id=run_id,
            iter_id=iter_id,
            similar_pdes_configs=similar_pdes_configs,
            exploration_scores=exploration_scores
        )
        config["task"] = pde_name
        config["pde_list"] = [pde_name]
        
        # Simplified configuration display
        key_params = {k: v for k, v in config.items() 
                     if k not in ['task', 'pde_list']}
        print(f"Config: {key_params}")
        
        # Set iteration directory
        iter_dir = os.path.join(run_output_dir, f"iter_{iter_id}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Set PINNacle output directory
        programmer.fixed["output_dir"] = run_output_dir
        programmer.fixed["name"] = f"iter_{iter_id}"
        
        # Generate configuration file
        yaml_path = os.path.join(iter_dir, "config.yaml")
        programmer.write_yaml(config, yaml_path)
        
        print(f"YAML Config File: {yaml_path}")
        
        # Run experiment
        print("Training...")
        mse, run_time = programmer.run_exp(yaml_path)
        
        # Display results
        print(f"✓ Completed | MSE: {format_mse(mse)} | Time: {format_time(run_time)}s")
        
        # NEW: Update visit counts if using UCT
        if memory_tree and args.use_uct:
            memory_tree.update_visit_count(pde_name, config)
            print(f"📈 Updated visit counts")
        
        # Record experiment history
        iteration_record = {
            "iter_id": iter_id,
            "config": config,
            "mse": mse,
            "run_time": run_time,
            "pde_name": pde_name,
            "run_id": run_id
        }
        iteration_history.append(iteration_record)
        
        # Add to knowledge base
        record = dict(config)
        record["task"] = pde_name
        record["mse"] = mse
        record["run_time"] = run_time
        kb.add_record(record)
        
        # Display current best result
        best_iteration = min(iteration_history, key=lambda x: x["mse"])
        print(f"Best So Far | MSE: {format_mse(best_iteration['mse'])} | run_time: {format_time(best_iteration['run_time'])}s")
    
    # Save single run experiment summary
    logger.save_run_summary(iteration_history, run_output_dir, pde_name, run_id)
    
    # Summarize this round of experiment
    best_iteration = min(iteration_history, key=lambda x: x["mse"])
    print(f"\n{'='*80}")
    print(f"Run {run_id} Summary for {pde_name}:")
    print(f"  Best MSE: {format_mse(best_iteration['mse'])}")
    print(f"  Best Time: {format_time(best_iteration['run_time'])}s")
    print(f"  Best Config: {best_iteration['config']}")
    print(f"{'='*80}")
    
    return iteration_history

def main():
    args = parse_args()
    
    # Set up experiment
    config_loader, output_dir = setup_experiment(args)
    
    # Initialize agents
    kb, pgkr, memory_tree, planner, programmer, logger = initialize_agents(args, config_loader, output_dir)
    
    # Get PDE list
    if args.pde_name:
        pde_list = [args.pde_name]
    else:
        pde_list = config_loader.get_pde_list(args.pde_type)
    
    print(f"\n{'='*120}")
    print(f"Created Output Directory: {output_dir}")
    print(f"PDEs to run: {pde_list}")
    print(f"{'='*120}")
    
    # Run experiments
    all_results = {}
    
    for pde_name in pde_list:
        pde_results = []
        
        for run_id in range(1, args.num_runs + 1):
            iteration_history = run_single_pde_experiment(
                pde_name, args, config_loader, 
                kb, pgkr, memory_tree, planner, programmer, logger,
                output_dir, run_id
            )
            pde_results.extend(iteration_history)
            all_results[pde_name] = pde_results
            
            # Update overall results in real-time after each run completion
            logger.save_experiment_summary(all_results, args, completed_runs=run_id)
        
        # After all runs for this PDE are completed, save PDE-level summary
        logger.save_pde_summary(pde_name, all_results)
        print(f"\n✅ Completed all {args.num_runs} runs for {pde_name}\n")
    
    # Save knowledge base (controlled by argument)
    if args.save_kb:
        save_path = args.kb_save_path if args.kb_save_path else None
        kb.save(save_path)
        saved_to = save_path if save_path else args.csv_path
        print(f"\n Knowledge base saved to: {saved_to}")
    else:
        print(f"\n Knowledge base not saved (use --save_kb to enable saving)")
    
    # Final summary
    logger.save_experiment_summary(all_results, args, completed_runs=args.num_runs)
    
    # Print concise summary table for all PDEs
    logger.print_all_pdes_summary(all_results)

if __name__ == "__main__":
    main()
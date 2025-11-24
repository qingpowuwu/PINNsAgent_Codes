# database/memory_tree.py
from pathlib import Path
import os
import sys
current_dir = Path(__file__).parent.absolute()
pinnsagent_root = current_dir.parent
os.chdir(pinnsagent_root)
sys.path.insert(0, str(pinnsagent_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from database.knowledge_base import KnowledgeBase
from collections import defaultdict
import json
import copy


class MemoryTree:
    """
    Offline Memory Tree for managing exploration scores based on historical data.
    
    This class maintains:
    1. Exploration scores (based on performance variance)
    2. Visit counts N(param, value) for each hyperparameter value
    3. Success rates (fraction of trials that achieved good performance)
    4. UCT exploration bonuses for online search
    
    Usage:
        - Offline: Precompute statistics from historical experiments
        - Online: Use global statistics to guide search on new PDEs
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize MemoryTree from a KnowledgeBase instance.
        
        Args:
            knowledge_base: KnowledgeBase instance with historical experiments
        """
        self.kb = knowledge_base
        self.df = self.kb.df  # Direct access to DataFrame
        
        # Precompute exploration scores for all PDEs
        self.scores_by_pde = self._precompute_scores_by_pde()
        
        # Compute global statistics (for simulate_new_pde mode)
        self.global_scores = self._compute_global_scores()
        
        # Precompute visit counts and success rates
        self.visit_counts_by_pde = self._precompute_visit_counts_by_pde()
        self.global_visit_counts = self._compute_global_visit_counts()
        self.global_success_rates = self._compute_global_success_rates()
        
        # NEW: Online visit counts for current tuning session (per PDE)
        # Format: {pde_name: {param_name: total_visits}}
        self.online_visit_counts = {}
    
    def _precompute_scores_by_pde(self) -> Dict[str, Dict[str, float]]:
        """
        Precompute exploration scores for each PDE separately.
        
        Returns:
            Dict of {pde_name: {param_name: score}}
        """
        scores_by_pde = {}
        
        # Get all available PDEs from KnowledgeBase
        all_pdes = self.kb.list_available_pdes()
        
        for pde_name in all_pdes:
            # Filter data for this PDE using KnowledgeBase's DataFrame
            pde_df = self.df[self.df['task'] == pde_name]
            
            # Convert to history format
            history = self._df_to_history(pde_df)
            
            # Calculate scores for this PDE
            scores = self.calculate_exploration_scores(history)
            scores_by_pde[pde_name] = scores
        
        return scores_by_pde
    
    def _precompute_visit_counts_by_pde(self) -> Dict[str, Dict[str, Dict[Any, int]]]:
        """
        Precompute visit counts N(param, value) for each PDE.
        
        Returns:
            Dict of {pde_name: {param_name: {value: count}}}
        """
        visit_counts_by_pde = {}
        all_pdes = self.kb.list_available_pdes()
        
        for pde_name in all_pdes:
            pde_df = self.df[self.df['task'] == pde_name]
            visit_counts = self._calculate_visit_counts(pde_df)
            visit_counts_by_pde[pde_name] = visit_counts
        
        return visit_counts_by_pde
    
    def _calculate_visit_counts(self, df: pd.DataFrame) -> Dict[str, Dict[Any, int]]:
        """
        Calculate visit counts for each (param, value) pair.
        
        Args:
            df: DataFrame with experiment records
            
        Returns:
            Dict of {param_name: {value: count}}
        """
        visit_counts = defaultdict(lambda: defaultdict(int))
        
        params = ["activation", "net", "optimizer", "initializer", 
                 "lr", "width", "depth", "num_domain_points", 
                 "num_boundary_points", "num_initial_points"]
        
        for _, row in df.iterrows():
            for param in params:
                value = row.get(param)
                if value is not None:
                    # For continuous params, bin them
                    if param in ["lr", "width", "depth", "num_domain_points", 
                                "num_boundary_points", "num_initial_points"]:
                        value = self._bin_continuous_value(param, value)
                    visit_counts[param][value] += 1
        
        # Convert defaultdict to regular dict for JSON serialization
        return {k: dict(v) for k, v in visit_counts.items()}
    
    def _bin_continuous_value(self, param: str, value: Any) -> str:
        """
        Bin continuous parameter values into discrete ranges.
        
        Args:
            param: Parameter name
            value: Parameter value
            
        Returns:
            Binned value as string (e.g., "lr_1e-4", "width_128")
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if param == "lr":
            # Log-scale binning for learning rate
            if value >= 1e-2:
                return "lr_1e-2+"
            elif value >= 1e-3:
                return "lr_1e-3"
            elif value >= 1e-4:
                return "lr_1e-4"
            else:
                return "lr_1e-5-"
        
        elif param in ["width", "depth"]:
            # Fixed bins
            bins = {
                "width": [32, 64, 128, 256, 512],
                "depth": [2, 3, 4, 5, 6, 8]
            }
            for threshold in sorted(bins[param]):
                if value <= threshold:
                    return f"{param}_{threshold}"
            return f"{param}_{bins[param][-1]}+"
        
        elif param in ["num_domain_points", "num_boundary_points", "num_initial_points"]:
            # Log-scale binning for point counts
            if value >= 10000:
                return f"{param}_10k+"
            elif value >= 5000:
                return f"{param}_5k"
            elif value >= 2000:
                return f"{param}_2k"
            else:
                return f"{param}_1k-"
        
        return str(value)
    
    def _compute_global_visit_counts(self) -> Dict[str, Dict[Any, int]]:
        """
        Compute global visit counts across all PDEs.
        
        Returns:
            Dict of {param_name: {value: total_count}}
        """
        global_counts = defaultdict(lambda: defaultdict(int))
        
        for pde_name, pde_counts in self.visit_counts_by_pde.items():
            for param, value_counts in pde_counts.items():
                for value, count in value_counts.items():
                    global_counts[param][value] += count
        
        return {k: dict(v) for k, v in global_counts.items()}
    
    def _compute_global_success_rates(self, threshold_percentile: int = 25) -> Dict[str, Dict[Any, float]]:
        """
        Compute success rates for each (param, value) pair.
        
        Success = achieving MSE below threshold (e.g., 25th percentile)
        
        Args:
            threshold_percentile: Percentile for defining success (lower is better)
            
        Returns:
            Dict of {param_name: {value: success_rate}}
        """
        # Define success threshold based on all MSE values
        all_mse = self.df['mse'].dropna()
        all_mse = all_mse[all_mse > 0]  # Remove invalid values
        if len(all_mse) == 0:
            return {}
        
        threshold = np.percentile(all_mse, threshold_percentile)
        
        success_rates = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
        
        params = ["activation", "net", "optimizer", "initializer", 
                 "lr", "width", "depth", "num_domain_points", 
                 "num_boundary_points", "num_initial_points"]
        
        for _, row in self.df.iterrows():
            mse = row.get('mse')
            if mse is None or mse <= 0 or np.isnan(mse):
                continue
            
            is_success = mse < threshold
            
            for param in params:
                value = row.get(param)
                if value is not None:
                    # Bin continuous params
                    if param in ["lr", "width", "depth", "num_domain_points", 
                                "num_boundary_points", "num_initial_points"]:
                        value = self._bin_continuous_value(param, value)
                    
                    success_rates[param][value]["total"] += 1
                    if is_success:
                        success_rates[param][value]["success"] += 1
        
        # Calculate rates
        final_rates = {}
        for param, value_stats in success_rates.items():
            final_rates[param] = {}
            for value, stats in value_stats.items():
                if stats["total"] > 0:
                    final_rates[param][value] = stats["success"] / stats["total"]
                else:
                    final_rates[param][value] = 0.0
        
        return final_rates
    
    def _df_to_history(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame rows to history format.
        
        Args:
            df: DataFrame with experiment records
            
        Returns:
            List of history entries
        """
        history = []
        for _, row in df.iterrows():
            config = {
                "activation": row.get("activation"),
                "net": row.get("net"),
                "optimizer": row.get("optimizer"),
                "initializer": row.get("initializer"),
                "lr": row.get("lr"),
                "width": row.get("width"),
                "depth": row.get("depth"),
                "num_domain_points": row.get("num_domain_points"),
                "num_boundary_points": row.get("num_boundary_points"),
                "num_initial_points": row.get("num_initial_points")
            }
            history.append({
                "config": config,
                "mse": row.get("mse"),
                "run_time": row.get("run_time")
            })
        
        return history
    
    def _compute_global_scores(self) -> Dict[str, float]:
        """
        Compute global average exploration scores across all PDEs.
        
        Returns:
            Dict of {param_name: avg_score}
        """
        if not self.scores_by_pde:
            return {}
        
        # Get all parameter names
        all_params = set()
        for scores in self.scores_by_pde.values():
            all_params.update(scores.keys())
        
        # Average across all PDEs
        global_scores = {}
        for param in all_params:
            scores = [pde_scores.get(param, 0.0) 
                     for pde_scores in self.scores_by_pde.values()]
            global_scores[param] = np.mean(scores)
        
        return global_scores
    
    def _compute_global_scores_exclude_target(self, target_pde: str) -> Dict[str, float]:
        """
        Compute global scores excluding target PDE.
        Simulates scenario where target PDE is new/unseen.
        
        Args:
            target_pde: PDE name to exclude
            
        Returns:
            Dict of {param_name: avg_score} excluding target PDE
        """
        if not self.scores_by_pde:
            return {}
        
        # Get all parameter names
        all_params = set()
        for pde_name, scores in self.scores_by_pde.items():
            if pde_name != target_pde:  # Exclude target PDE
                all_params.update(scores.keys())
        
        # Average across all PDEs except target
        global_scores_exclude = {}
        for param in all_params:
            scores = [pde_scores.get(param, 0.0) 
                     for pde_name, pde_scores in self.scores_by_pde.items()
                     if pde_name != target_pde]  # Exclude target PDE
            if scores:  # Only compute if there are scores
                global_scores_exclude[param] = np.mean(scores)
            else:
                global_scores_exclude[param] = 0.0
        
        return global_scores_exclude
    
    def get_scores_for_pde(self, pde_name: str, simulate_new_pde: bool = False) -> Dict[str, float]:
        """
        Get exploration scores for a specific PDE.
        
        Args:
            pde_name: Target PDE name
            simulate_new_pde: If True, exclude target PDE (simulate unseen scenario)
            
        Returns:
            Dict of {param_name: score} sorted by score (descending)
        """
        if simulate_new_pde:
            scores = self._compute_global_scores_exclude_target(pde_name)
        else:
            if pde_name in self.scores_by_pde:
                scores = self.scores_by_pde[pde_name]
            else:
                raise ValueError(
                    f"PDE '{pde_name}' not found. "
                    f"Available: {list(self.scores_by_pde.keys())}. "
                    f"Use simulate_new_pde=True for global scores."
                )
        
        rounded_scores = {k: round(v, 3) for k, v in scores.items()}
        return dict(sorted(rounded_scores.items(), key=lambda x: x[1], reverse=True))
    
    def reset_online_visit_counts(self, pde_name: str):
        """
        Reset online visit counts for a new tuning session.
        
        Args:
            pde_name: PDE name to initialize
        """
        # FIXED: Track total visits per parameter (not per value)
        self.online_visit_counts[pde_name] = defaultdict(int)
    
    def update_visit_count(self, pde_name: str, config: Dict[str, Any]):
        """
        Update online visit counts after each trial.
        
        FIXED: Only increment the parameters that were actually tuned.
        
        Args:
            pde_name: PDE name
            config: Configuration dict with parameter values
        """
        if pde_name not in self.online_visit_counts:
            self.reset_online_visit_counts(pde_name)
        
        params = ["activation", "net", "optimizer", "initializer", 
                 "lr", "width", "depth", "num_domain_points", 
                 "num_boundary_points", "num_initial_points"]
        
        for param in params:
            value = config.get(param)
            if value is not None:
                # FIXED: Only increment parameter-level count
                self.online_visit_counts[pde_name][param] += 1
    
    def get_uct_scores(self, pde_name: str, simulate_new_pde: bool = False, 
                      lambda_val: float = 1.4) -> Dict[str, float]:
        """
        Calculate UCT exploration scores.
        
        FIXED: Only apply UCT bonus to visited parameters.
        
        UCT formula for visited params: 
            Score = ExploitationScore + λ × √(log(N_param + 1) / (N_param + 1))
        
        For unvisited params:
            Score = ExploitationScore (no UCT bonus)
        
        Args:
            pde_name: Target PDE name
            simulate_new_pde: If True, use global statistics
            lambda_val: Exploration weight (default 1.4)
            
        Returns:
            Dict of {param_name: uct_score} sorted by score (descending)
        """
        # Get base exploration scores (exploitation term)
        base_scores = self.get_scores_for_pde(pde_name, simulate_new_pde)
        
        # Get parameter-level visit counts
        if simulate_new_pde:
            # Use online counts only
            if pde_name in self.online_visit_counts:
                param_visits = dict(self.online_visit_counts[pde_name])
            else:
                # No visits yet, return base scores
                return base_scores
        else:
            # Use historical visit counts
            if pde_name in self.visit_counts_by_pde:
                # Calculate total visits per parameter from value counts
                param_visits = {}
                for param, value_counts in self.visit_counts_by_pde[pde_name].items():
                    param_visits[param] = sum(value_counts.values())
                
                # Add online visits
                if pde_name in self.online_visit_counts:
                    for param, count in self.online_visit_counts[pde_name].items():
                        param_visits[param] = param_visits.get(param, 0) + count
            else:
                # Use online only
                if pde_name in self.online_visit_counts:
                    param_visits = dict(self.online_visit_counts[pde_name])
                else:
                    return base_scores
        
        # If no parameters have been visited, return base scores
        if not param_visits or all(v == 0 for v in param_visits.values()):
            return base_scores
        
        # Calculate UCT scores
        uct_scores = {}
        for param, base_score in base_scores.items():
            N_param = param_visits.get(param, 0)
            
            if N_param > 0:
                # FIXED: Apply UCT bonus only to visited parameters
                # Use parameter's own visit count for exploration bonus
                exploration_bonus = lambda_val * np.sqrt(np.log(N_param + 1) / (N_param + 1))
                uct_scores[param] = base_score + exploration_bonus
            else:
                # FIXED: Unvisited parameters keep base score only
                uct_scores[param] = base_score
        
        # Sort by UCT score (descending)
        return dict(sorted(uct_scores.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_k_params(self, pde_name: str, k: int = 3, 
                        simulate_new_pde: bool = False,
                        use_uct: bool = True,
                        lambda_val: float = 1.4) -> List[str]:
        """
        Get top-k parameters to explore.
        
        Args:
            pde_name: Target PDE name
            k: Number of parameters to return
            simulate_new_pde: If True, use global statistics
            use_uct: If True, use UCT scores; else use base exploration scores
            lambda_val: UCT exploration weight
            
        Returns:
            List of parameter names sorted by score (descending)
        """
        if use_uct:
            scores = self.get_uct_scores(pde_name, simulate_new_pde, lambda_val)
        else:
            scores = self.get_scores_for_pde(pde_name, simulate_new_pde)
        
        return list(scores.keys())[:k]
    
    def get_statistics_summary(self, pde_name: str, simulate_new_pde: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a PDE.
        
        Returns:
            Dict with exploration scores, visit counts, success rates, and UCT scores
        """
        # Get parameter-level visit counts
        if simulate_new_pde:
            if pde_name in self.online_visit_counts:
                param_visits = dict(self.online_visit_counts[pde_name])
            else:
                param_visits = {}
        else:
            param_visits = {}
            if pde_name in self.visit_counts_by_pde:
                for param, value_counts in self.visit_counts_by_pde[pde_name].items():
                    param_visits[param] = sum(value_counts.values())
            
            # Add online visits
            if pde_name in self.online_visit_counts:
                for param, count in self.online_visit_counts[pde_name].items():
                    param_visits[param] = param_visits.get(param, 0) + count
        
        return {
            "exploration_scores": self.get_scores_for_pde(pde_name, simulate_new_pde),
            "uct_scores": self.get_uct_scores(pde_name, simulate_new_pde),
            "visit_counts": param_visits,
            "success_rates": self.global_success_rates,
            "top_3_params_base": self.get_top_k_params(pde_name, 3, simulate_new_pde, use_uct=False),
            "top_3_params_uct": self.get_top_k_params(pde_name, 3, simulate_new_pde, use_uct=True)
        }
    
    def calculate_exploration_scores(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate exploration scores based on historical performance variance.
        
        Args:
            history: Historical experiment records
            
        Returns:
            Dict of {param_name: exploration_score} normalized to [0, 1]
        """
        exploration_scores = {}
        
        # Categorize parameters
        categorical_params = ["activation", "net", "optimizer", "initializer"]
        continuous_params = ["lr", "width", "depth", "num_domain_points", 
                            "num_boundary_points", "num_initial_points"]
        
        # Extract and normalize MSE values
        all_mse = [exp.get("mse", float('inf')) for exp in history]
        all_mse = [m for m in all_mse if m != float('inf') and not np.isnan(m) and m > 0]
        
        if not all_mse:
            # No valid data, return uniform scores
            return {param: 1.0 for param in categorical_params + continuous_params}
        
        # Use log scale for MSE since it spans many orders of magnitude
        log_mse = np.log10(all_mse)
        
        # Process categorical parameters
        for param_name in categorical_params:
            score = self._calculate_categorical_score(history, param_name, log_mse)
            exploration_scores[param_name] = score
        
        # Process continuous parameters
        for param_name in continuous_params:
            score = self._calculate_continuous_score(history, param_name, log_mse)
            exploration_scores[param_name] = score
        
        # Normalize scores to [0, 1]
        if exploration_scores:
            max_score = max(exploration_scores.values())
            min_score = min(exploration_scores.values())
            if max_score > min_score:
                for param_name in exploration_scores:
                    exploration_scores[param_name] = (
                        (exploration_scores[param_name] - min_score) / (max_score - min_score)
                    )
            else:
                # All scores are the same, normalize to 0.5
                for param_name in exploration_scores:
                    exploration_scores[param_name] = 0.5
        
        return exploration_scores

    def _calculate_categorical_score(self, history: List[Dict[str, Any]], 
                                    param_name: str, log_mse_all: np.ndarray) -> float:
        """
        Calculate exploration score for categorical parameters.

            Score = Performance_Gap × min(1 + Success_Rate_Std, 2.0)
        
        - Performance_Gap: Performance gap between best and worst values (log scale)
        - Success_Rate_Std: Success rate variance (as a multiplier, capped at 2x)
        """
        # Group experiments by parameter value
        value_groups = {}
        for exp in history:
            config = exp.get("config", {})
            value = config.get(param_name)
            mse = exp.get("mse", float('inf'))
            
            if value is not None and mse != float('inf') and not np.isnan(mse) and mse > 0:
                if value not in value_groups:
                    value_groups[value] = []
                # Use log scale MSE
                value_groups[value].append(np.log10(mse))
        
        if len(value_groups) <= 1:
            return 0.0
        
        # (1) Performance gap
        value_avg_mse = {value: np.mean(mses) for value, mses in value_groups.items()}
        mse_values = np.array(list(value_avg_mse.values()))
        performance_gap = np.max(mse_values) - np.min(mse_values)
        
        # (2) Success rate variance
        threshold = np.percentile([m for mses in value_groups.values() for m in mses], 25)
        success_rates = {
            value: np.mean([m < threshold for m in mses])
            for value, mses in value_groups.items()
        }
        success_rate_std = np.std(list(success_rates.values()))
        
        # Combined score: Gap is primary, success_rate_std can boost up to 2x
        # Using (1 + std) ensures gap always dominates
        score = performance_gap * min(1.0 + success_rate_std, 2.0)
        
        return score

    def _calculate_continuous_score(self, history: List[Dict[str, Any]], 
                                param_name: str, log_mse_all: np.ndarray) -> float:
        """
        Calculate exploration score for continuous parameters.
        
            Score = Range_Gap × (1 + |Correlation|)
        
        - Range_Gap: measures the magnitude of impact (primary)
        - Correlation: measures monotonicity (multiplier, 1-2x boost)
        """
        param_values = []
        log_mse_values = []
        
        for exp in history:
            config = exp.get("config", {})
            value = config.get(param_name)
            mse = exp.get("mse", float('inf'))
            
            # Skip if value is None or MSE is invalid
            if value is None or mse == float('inf') or np.isnan(mse) or mse <= 0:
                continue
            
            # Try to convert to float, skip if it's not a numeric value
            try:
                numeric_value = float(value)
                param_values.append(numeric_value)
                log_mse_values.append(np.log10(mse))
            except (ValueError, TypeError):
                # Skip non-numeric values (e.g., "['Burgers1D']", "auto", etc.)
                continue
        
        if len(param_values) < 6:
            return 0.0
        
        # Convert to numpy arrays
        param_values = np.array(param_values)
        log_mse_values = np.array(log_mse_values)
        
        # (1) Range gap
        sorted_indices = np.argsort(param_values)
        sorted_mse = log_mse_values[sorted_indices]
        n = len(sorted_mse)
        
        low_third = sorted_mse[:n//3]
        high_third = sorted_mse[-n//3:]
        range_gap = abs(np.mean(high_third) - np.mean(low_third))
        
        # (2) Correlation
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(param_values, log_mse_values)
            correlation = abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            correlation = 0.0
        
        # Combined score
        score = range_gap * (1.0 + correlation)
        
        return score
    
    def save_scores(self, output_dir: str = "./data"):
        """
        Save all statistics to CSV files.
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Exploration scores by PDE
        scores_df = pd.DataFrame(self.scores_by_pde).T
        scores_df.index.name = 'Task'
        scores_df.to_csv(output_path / "exploration_scores_by_pde.csv")
        
        # 2. Global exploration scores
        summary_df = pd.DataFrame({
            'Parameter': list(self.global_scores.keys()),
            'Global_Score': list(self.global_scores.values())
        }).sort_values('Global_Score', ascending=False)
        summary_df.to_csv(output_path / "exploration_scores_global.csv", index=False)
        
        # 3. Visit counts
        visit_counts_flattened = []
        for param, value_counts in self.global_visit_counts.items():
            for value, count in value_counts.items():
                visit_counts_flattened.append({
                    'Parameter': param,
                    'Value': value,
                    'Count': count
                })
        pd.DataFrame(visit_counts_flattened).to_csv(
            output_path / "visit_counts_global.csv", index=False
        )
        
        # 4. Success rates
        success_rates_flattened = []
        for param, value_rates in self.global_success_rates.items():
            for value, rate in value_rates.items():
                success_rates_flattened.append({
                    'Parameter': param,
                    'Value': value,
                    'Success_Rate': rate
                })
        pd.DataFrame(success_rates_flattened).to_csv(
            output_path / "success_rates_global.csv", index=False
        )
        
        # 5. Save detailed statistics as JSON
        stats = {
            "global_exploration_scores": self.global_scores,
            "global_visit_counts": self.global_visit_counts,
            "global_success_rates": self.global_success_rates
        }
        with open(output_path / "global_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Saved all statistics to {output_dir}/")


if __name__ == "__main__":
    from database.knowledge_base import KnowledgeBase
    import matplotlib.pyplot as plt
    
    print("="*80)
    print("Testing MemoryTree with 5 Tuning Trials - UCT Evolution")
    print("="*80)
    
    # Initialize KnowledgeBase
    csv_path = "./data/dataset_for_retrieval.csv"
    kb = KnowledgeBase(csv_path)
    
    print("\n[Initialization] Creating MemoryTree...")
    memory_tree = MemoryTree(knowledge_base=kb)
    
    print(f"✓ Loaded {len(memory_tree.scores_by_pde)} PDEs")
    print(f"✓ Computed {len(memory_tree.global_scores)} exploration scores")
    
    # Test: Simulate 5 tuning trials
    print("\n" + "="*80)
    print("[Simulation] 5 Tuning Trials for Burgers1D (New PDE)")
    print("="*80)
    
    # Reset online counts
    memory_tree.reset_online_visit_counts("Burgers1D")
    
    # Define 5 trial configurations (simulating a realistic tuning process)
    trials = [
        {
            "name": "Trial 1: Explore activation & optimizer",
            "config": {
                "activation": "tanh",
                "optimizer": "adam",
            }
        },
        {
            "name": "Trial 2: Explore lr & width",
            "config": {
                "lr": 0.001,
                "width": 64,
            }
        },
        {
            "name": "Trial 3: Re-tune activation with different net",
            "config": {
                "activation": "sin",
                "net": "fnn",
            }
        },
        {
            "name": "Trial 4: Fine-tune depth & boundary points",
            "config": {
                "depth": 4,
                "num_boundary_points": 100,
            }
        },
        {
            "name": "Trial 5: Re-tune optimizer & lr",
            "config": {
                "optimizer": "lbfgs",
                "lr": 0.0001,
            }
        },
    ]
    
    # Track UCT evolution
    uct_history = []
    visit_count_history = []
    top3_history = []
    
    # Initial state (before any trials)
    initial_scores = memory_tree.get_uct_scores("Burgers1D", simulate_new_pde=True)
    uct_history.append(("Initial", initial_scores.copy()))
    visit_count_history.append(("Initial", memory_tree.online_visit_counts.get("Burgers1D", {}).copy()))
    top3_history.append(("Initial", memory_tree.get_top_k_params("Burgers1D", k=3, simulate_new_pde=True)))
    
    # Run 5 trials
    for i, trial in enumerate(trials, 1):
        print(f"\n{'-'*80}")
        print(f"[{trial['name']}]")
        print(f"{'-'*80}")
        
        # Update visit counts
        memory_tree.update_visit_count("Burgers1D", trial["config"])
        
        # Get updated UCT scores
        current_scores = memory_tree.get_uct_scores("Burgers1D", simulate_new_pde=True)
        current_visits = memory_tree.online_visit_counts["Burgers1D"].copy()
        current_top3 = memory_tree.get_top_k_params("Burgers1D", k=3, simulate_new_pde=True)
        
        # Store history
        uct_history.append((f"Trial {i}", current_scores.copy()))
        visit_count_history.append((f"Trial {i}", current_visits.copy()))
        top3_history.append((f"Trial {i}", current_top3))
        
        # Print tuned parameters
        print(f"\nTuned parameters:")
        for param, value in trial["config"].items():
            print(f"  • {param:25s} = {value}")
        
        # Print current visit counts
        print(f"\nCurrent visit counts:")
        for param in sorted(current_visits.keys(), key=lambda x: current_visits[x], reverse=True):
            count = current_visits[param]
            print(f"  [{count}x] {param}")
        
        # Print UCT scores with changes
        print(f"\nUCT Scores after Trial {i}:")
        prev_scores = uct_history[-2][1]  # Previous trial's scores
        for param, score in list(current_scores.items()):  # Top 5
            change = score - prev_scores.get(param, 0)
            visits = current_visits.get(param, 0)
            marker = "★" if param in trial["config"] else " "
            print(f"  {marker} [{visits}x] {param:22s}: {score:.4f} (Δ = {change:+.4f})")
        
        # Print top-3 params
        print(f"\nTop-3 parameters to explore next: {current_top3}")
    
    # ========================================================================
    # Summary: UCT Evolution Table
    # ========================================================================
    print("\n" + "="*80)
    print("[Summary] UCT Scores Evolution (Top-5 Parameters)")
    print("="*80)
    
    # Get top-5 most visited parameters
    final_visits = visit_count_history[-1][1]
    top_params = sorted(final_visits.keys(), key=lambda x: final_visits[x], reverse=True)
    
    # Print header
    print(f"\n{'Parameter':<20s}", end="")
    for stage, _ in uct_history:
        print(f" | {stage:^12s}", end="")
    print()
    print("-" * (20 + 15 * len(uct_history)))
    
    # Print each parameter's evolution
    for param in top_params:
        print(f"{param:<20s}", end="")
        for stage, scores in uct_history:
            score = scores.get(param, 0)
            print(f" | {score:>12.4f}", end="")
        print()
    
    # ========================================================================
    # Summary: Visit Count Evolution
    # ========================================================================
    print("\n" + "="*80)
    print("[Summary] Visit Count Evolution")
    print("="*80)
    
    print(f"\n{'Parameter':<20s}", end="")
    for stage, _ in visit_count_history:
        print(f" | {stage:^12s}", end="")
    print()
    print("-" * (20 + 15 * len(visit_count_history)))
    
    for param in top_params:
        print(f"{param:<20s}", end="")
        for stage, visits in visit_count_history:
            count = visits.get(param, 0)
            print(f" | {count:>12d}", end="")
        print()
    
    # ========================================================================
    # Summary: Top-3 Evolution
    # ========================================================================
    print("\n" + "="*80)
    print("[Summary] Top-3 Parameters Evolution")
    print("="*80)
    
    for stage, top3 in top3_history:
        print(f"\n{stage:12s}: {top3}")
    
    # ========================================================================
    # Visualization: UCT Evolution Plot
    # ========================================================================
    print("\n" + "="*80)
    print("[Visualization] Generating UCT Evolution Plot...")
    print("="*80)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: UCT Scores Evolution
        ax1 = axes[0]
        stages = [stage for stage, _ in uct_history]
        for param in top_params:
            scores = [scores_dict.get(param, 0) for _, scores_dict in uct_history]
            ax1.plot(stages, scores, marker='o', label=param, linewidth=2)
        
        ax1.set_xlabel('Tuning Stage', fontsize=12)
        ax1.set_ylabel('UCT Score', fontsize=12)
        ax1.set_title('UCT Scores Evolution Over 5 Trials', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticklabels(stages, rotation=45, ha='right')
        
        # Plot 2: Visit Counts Evolution
        ax2 = axes[1]
        for param in top_params:
            counts = [visits_dict.get(param, 0) for _, visits_dict in visit_count_history]
            ax2.plot(stages, counts, marker='s', label=param, linewidth=2)
        
        ax2.set_xlabel('Tuning Stage', fontsize=12)
        ax2.set_ylabel('Visit Count', fontsize=12)
        ax2.set_title('Visit Counts Evolution Over 5 Trials', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticklabels(stages, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('./data/uct_evolution_5_trials.png', dpi=150, bbox_inches='tight')
        print("✓ Plot saved to ./data/uct_evolution_5_trials.png")
        
    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")
    
    # ========================================================================
    # Final Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("[Analysis] Key Observations")
    print("="*80)
    
    final_scores = uct_history[-1][1]
    final_visits = visit_count_history[-1][1]
    
    print("\n1. Most Visited Parameters:")
    for param in sorted(final_visits.keys(), key=lambda x: final_visits[x], reverse=True)[:3]:
        count = final_visits[param]
        print(f"   • {param:25s}: {count} visits")
    
    print("\n2. Highest UCT Scores (Final):")
    for param, score in list(final_scores.items())[:3]:
        visits = final_visits.get(param, 0)
        print(f"   • {param:25s}: {score:.4f} (visited {visits}x)")
    
    print("\n3. Unexplored Parameters:")
    unexplored = [param for param in final_scores.keys() if final_visits.get(param, 0) == 0]
    if unexplored:
        for param in unexplored[:3]:
            score = final_scores[param]
            print(f"   • {param:25s}: {score:.4f} (not visited yet)")
    else:
        print("   (All parameters have been explored)")
    
    print("\n" + "="*80)
    print("All tests completed! ✓")
    print("="*80)
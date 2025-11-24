# prompts/pinnsagent_prompt.py

import json
from typing import Dict, List, Any, Optional
from prompts.base_prompt import BasePrompt
from prompts.pgkr_prompt import PGKRPrompt
from prompts.memory_tree_prompt import MemoryTreePrompt


class PINNsAgentPrompt(BasePrompt):
    """
    PINNsAgent prompt strategy with iteration-aware composition
    
    This strategy dynamically combines PGKR and MemoryTree based on iteration:
    - Iteration 1: PGKR + MemoryTree (cold start with prior knowledge)
    - Iteration 2+: MemoryTree only (focus on exploration guided by scores)
    
    Args:
        search_space: Hyperparameter search space dictionary
        use_pgkr: Whether to use PGKR retrieval (default: False)
        use_memory_tree: Whether to use MemoryTree UCT scores (default: False)
        max_iterations: Maximum number of iterations (default: 5)
    """
    
    def __init__(
        self, 
        search_space: Dict[str, List], 
        use_pgkr: bool = False, 
        use_memory_tree: bool = False,
        max_iterations: int = 5
    ):
        """
        Initialize PINNsAgent prompt
        
        Args:
            search_space: Hyperparameter search space dictionary
            use_pgkr: Whether to use PGKR retrieval
            use_memory_tree: Whether to use MemoryTree UCT scores
            max_iterations: Maximum number of iterations
        """
        super().__init__(search_space)
        self.use_pgkr = use_pgkr
        self.use_memory_tree = use_memory_tree
        self.max_iterations = max_iterations
        
        # Initialize sub-prompt generators
        self.pgkr_prompt = PGKRPrompt(search_space) if use_pgkr else None
        self.mtrs_prompt = MemoryTreePrompt(search_space) if use_memory_tree else None
        
        # Track current iteration
        self.current_iteration = 1
    
    def set_iteration(self, iteration: int):
        """Set current iteration number"""
        self.current_iteration = iteration
    
    def get_system_prompt(self) -> str:
        """Generate iteration-aware system prompt"""
        
        # Iteration 1: Use PGKR if enabled
        if self.current_iteration == 1 and self.use_pgkr:
            return self._get_system_prompt_iter1()
        
        # Iteration 2+: Use MemoryTree if enabled
        elif self.current_iteration >= 2 and self.use_memory_tree:
            return self._get_system_prompt_iter2plus()
        
        # Fallback: basic prompt
        else:
            return self._get_basic_system_prompt()
    
    def _get_system_prompt_iter1(self) -> str:
        """System prompt for iteration 1 (PGKR + MemoryTree)"""
        components = []
        
        if self.use_pgkr:
            components.append("1. Best configurations from similar PDEs (PGKR - Physics-Guided Knowledge Retrieval)")
        
        if self.use_memory_tree:
            components.append(f"{len(components) + 1}. UCT scores from MemoryTree (parameter importance ranking)")
        
        components_str = "\n".join(components)
        
        prompt = f"""You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.

🎯 TASK: Recommend the FIRST experiment configuration (Iteration {self.current_iteration}/{self.max_iterations})

You will receive:
{components_str}

"""
        prompt += self.get_hyperparameter_description()
        prompt += f"""

📋 GUIDELINES:
• PGKR provides proven configurations from PDEs with similar physics - use them as strong references
• MemoryTree UCT scores show which parameters have the most impact historically
• This is a cold start - leverage both prior knowledge sources
• You have {self.max_iterations} iterations total to find the best configuration

⚠️ CRITICAL: Return ONLY valid JSON configuration, no explanations."""
        
        return prompt
    
    def _get_system_prompt_iter2plus(self) -> str:
        """System prompt for iteration 2+ (MemoryTree only)"""
        remaining_iterations = self.max_iterations - self.current_iteration + 1
        
        prompt = f"""You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.

🎯 TASK: Recommend the NEXT experiment configuration (Iteration {self.current_iteration}/{self.max_iterations})

You will receive:
1. UCT scores from MemoryTree (parameter importance ranking)
2. Historical experimental results from previous iterations

"""
        prompt += self.get_hyperparameter_description()
        prompt += f"""

📋 GUIDELINES:
• MemoryTree UCT scores indicate which parameters have the most impact on performance
• Higher scores = higher priority for tuning
• FOCUS on adjusting high-score parameters with different values
• Keep low-score parameters relatively stable
• You have {remaining_iterations} iterations remaining

⚠️ CRITICAL RULES:
1. Do NOT repeat configurations from previous experiments
2. Prioritize exploring top-3 high-score parameters
3. Return ONLY valid JSON configuration, no explanations"""
        
        return prompt
    
    def _get_basic_system_prompt(self) -> str:
        """Basic system prompt without PGKR or MemoryTree"""
        remaining_iterations = self.max_iterations - self.current_iteration + 1
        
        prompt = f"""You are a PINNs (Physics-Informed Neural Networks) hyperparameter optimization expert.

🎯 TASK: Recommend experiment configuration (Iteration {self.current_iteration}/{self.max_iterations})

"""
        prompt += self.get_hyperparameter_description()
        prompt += f"""

📋 GUIDELINES:
• Analyze historical experimental trends
• Recommend configurations likely to achieve lower MSE
• You have {remaining_iterations} iterations remaining

⚠️ CRITICAL: Do NOT repeat previous configurations. Return ONLY valid JSON."""
        
        return prompt
    
    def build_user_prompt(
        self, 
        history: List[Dict[str, Any]] = None,
        similar_pdes_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        exploration_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Build iteration-aware user prompt
        
        Args:
            history: Historical experiment records for current PDE
            similar_pdes_configs: Dictionary of similar PDEs' best configurations (only used in iter 1)
            exploration_scores: Dictionary of {param_name: uct_score} from MemoryTree
            
        Returns:
            User prompt string
        """
        
        # Iteration 1: Use PGKR + MemoryTree
        if self.current_iteration == 1:
            return self._build_user_prompt_iter1(
                history=history,
                similar_pdes_configs=similar_pdes_configs,
                exploration_scores=exploration_scores
            )
        
        # Iteration 2+: Use MemoryTree only
        else:
            return self._build_user_prompt_iter2plus(
                history=history,
                exploration_scores=exploration_scores
            )
    
    def _build_user_prompt_iter1(
        self,
        history: List[Dict[str, Any]] = None,
        similar_pdes_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        exploration_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """Build user prompt for iteration 1"""
        prompt = ""
        part_num = 1
        
        # Part 1: PGKR similar PDEs (if enabled)
        if self.use_pgkr and similar_pdes_configs:
            prompt += "=" * 80 + "\n"
            prompt += f"PART {part_num}: Best Configurations from Similar PDEs (PGKR)\n"
            prompt += "=" * 80 + "\n\n"
            
            for pde_name, pde_info in similar_pdes_configs.items():
                similarity = pde_info['similarity']
                best_config = pde_info['best_config']
                best_mse = pde_info['best_mse']
                
                prompt += f"📌 Similar PDE: {pde_name} (Similarity: {similarity:.4f})\n"
                prompt += f"   Best MSE: {best_mse:.2e}\n"
                prompt += f"   Best Config: {json.dumps(best_config, indent=2)}\n\n"
            
            part_num += 1
        
        # Part 2: MemoryTree UCT scores (if enabled)
        if self.use_memory_tree and exploration_scores:
            prompt += "=" * 80 + "\n"
            prompt += f"PART {part_num}: UCT Scores (Parameter Importance)\n"
            prompt += "=" * 80 + "\n\n"
            
            # Find max score for scaling
            max_score = max(exploration_scores.values()) if exploration_scores else 1.0
            
            # Show top-5 most important parameters prominently
            top_k = 5
            top_params = list(exploration_scores.items())[:top_k]
            
            prompt += f"🔥 TOP-{top_k} MOST IMPACTFUL PARAMETERS:\n"
            for param_name, score in top_params:
                bar_length = int((score / max_score) * 20) if max_score > 0 else 0
                bar = "█" * bar_length + "░" * (20 - bar_length)
                prompt += f"   {param_name:20s}: {score:.4f} {bar}\n"
            
            prompt += "\n📊 ALL PARAMETERS:\n"
            for param_name, score in exploration_scores.items():
                bar_length = int((score / max_score) * 20) if max_score > 0 else 0
                bar = "█" * bar_length + "░" * (20 - bar_length)
                prompt += f"   {param_name:20s}: {score:.4f} {bar}\n"
            
            prompt += "\n"
            part_num += 1
        
        # Task for iteration 1
        prompt += "=" * 80 + "\n"
        prompt += f"🎯 TASK: Recommend FIRST Configuration (Iter {self.current_iteration}/{self.max_iterations})\n"
        prompt += "=" * 80 + "\n\n"
        
        if self.use_pgkr and similar_pdes_configs:
            prompt += "Strategy:\n"
            prompt += "1. Start from configurations proven successful on similar PDEs (Part 1)\n"
        
        if self.use_memory_tree and exploration_scores:
            part_ref = 2 if (self.use_pgkr and similar_pdes_configs) else 1
            prompt += f"{'2' if self.use_pgkr else '1'}. Prioritize exploring high-score parameters (Part {part_ref})\n"
        
        prompt += f"{part_num}. Apply PINNs best practices\n\n"
        prompt += "Return configuration in JSON format:\n"
        prompt += self.get_json_format_example()
        
        return prompt
    
    def _build_user_prompt_iter2plus(
        self,
        history: List[Dict[str, Any]] = None,
        exploration_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """Build user prompt for iteration 2+"""
        prompt = ""
        
        # Part 1: MemoryTree UCT scores
        if self.use_memory_tree and exploration_scores:
            prompt += "=" * 80 + "\n"
            prompt += "PART 1: UCT Scores (Parameter Importance)\n"
            prompt += "=" * 80 + "\n\n"
            
            # Find max score for scaling
            max_score = max(exploration_scores.values()) if exploration_scores else 1.0
            
            # Identify top parameters to explore
            top_k = 4
            top_params = list(exploration_scores.items())[:top_k]
            top_param_names = [p[0] for p in top_params]
            
            prompt += f"🔥 TOP-{top_k} PARAMETERS TO EXPLORE:\n"
            for param_name, score in top_params:
                bar_length = int((score / max_score) * 20) if max_score > 0 else 0
                bar = "█" * bar_length + "░" * (20 - bar_length)
                prompt += f"   {param_name:20s}: {score:.4f} {bar}\n"
            
            prompt += f"\n💡 Focus on exploring: {', '.join(top_param_names)}\n"
            prompt += "   Try different values for these while keeping others relatively stable.\n\n"
            
            prompt += "📊 ALL PARAMETERS:\n"
            for param_name, score in exploration_scores.items():
                bar_length = int((score / max_score) * 20) if max_score > 0 else 0
                bar = "█" * bar_length + "░" * (20 - bar_length)
                prompt += f"   {param_name:20s}: {score:.4f} {bar}\n"
            
            prompt += "\n"
        
        # Part 2: Historical experiments
        if not history:
            raise ValueError("History should not be empty in iteration 2+")
        
        sorted_history = sorted(history, key=lambda x: x.get("mse", float('inf')))
        
        prompt += "=" * 80 + "\n"
        if exploration_scores:
            prompt += "PART 2: Historical Experimental Results\n"
        else:
            prompt += "Historical Experimental Results\n"
        prompt += "=" * 80 + "\n\n"
        
        prompt += f"📈 PREVIOUS EXPERIMENTS (Total: {len(sorted_history)}):\n\n"
        
        for i, exp in enumerate(sorted_history):
            mse = exp.get("mse", "N/A")
            config = exp.get("config", {})
            rank_emoji = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
            prompt += f"{rank_emoji} Experiment {i+1}: MSE = {mse:.2e}\n"
            prompt += f"Config: {json.dumps(config, indent=2)}\n\n"
        
        best_mse = sorted_history[0].get("mse", float('inf'))
        best_config = sorted_history[0].get("config", {})
        
        prompt += f"🎯 CURRENT BEST:\n"
        prompt += f"   MSE: {best_mse:.2e}\n"
        prompt += f"   Config: {json.dumps(best_config, indent=2)}\n\n"
        
        # Task description
        remaining_iterations = self.max_iterations - self.current_iteration + 1
        
        prompt += "=" * 80 + "\n"
        prompt += f"🎯 TASK: Recommend NEXT Configuration (Iter {self.current_iteration}/{self.max_iterations})\n"
        prompt += "=" * 80 + "\n\n"
        
        if self.use_memory_tree and exploration_scores:
            top_k = 3
            top_param_names = [p[0] for p in list(exploration_scores.items())[:top_k]]
            
            prompt += "Strategy:\n"
            prompt += f"1. 🔥 PRIORITIZE exploring: {', '.join(top_param_names)}\n"
            prompt += "2. 📊 Start from current best config and modify high-score parameters\n"
            prompt += "3. ✓ Keep low-score parameters relatively stable\n\n"
            
            prompt += f"💡 Suggestion: Take the best config and try different values for {top_param_names[0]}, {top_param_names[1]}, or {top_param_names[2]}\n\n"
        else:
            prompt += "Strategy:\n"
            prompt += "1. Analyze trends and recommend a novel configuration\n\n"
        
        prompt += f"⏱️ Remaining iterations: {remaining_iterations}\n\n"
        prompt += "Return ONLY the configuration in JSON format:"
        
        return prompt
# prompts/prompt_factory.py

from typing import Dict, List
from .base_prompt import BasePrompt
from .zero_shot_prompt import ZeroShotPrompt
from .full_history_prompt import FullHistoryPrompt
from .memory_tree_prompt import MemoryTreePrompt
from .pgkr_prompt import PGKRPrompt
from .pinns_agent_prompt import PINNsAgentPrompt


class PromptFactory:
    """
    Factory class for creating prompt strategy instances
    
    Supports:
    - zero_shot: Zero-shot prompt (no history)
    - full_history: Full history prompt (full history)
    - memory_tree: Memory tree prompt (exploration scores + history)
    - pgkr: PGKR prompt (similar PDEs' best configs + history)
    - pinns_agent: PINNsAgent prompt (PGKR + Memory Tree + full history)
    """
    
    # Registry of available prompt strategies
    _strategies = {
        "zero_shot": ZeroShotPrompt,
        "full_history": FullHistoryPrompt,
        "memory_tree": MemoryTreePrompt,
        "pgkr": PGKRPrompt,
        "pinns_agent": PINNsAgentPrompt,
    }
    
    @classmethod
    def create_prompt(cls, strategy: str, search_space: Dict[str, List], **kwargs) -> BasePrompt:
        """
        Create a prompt strategy instance
        
        Args:
            strategy: Strategy name ("zero_shot", "full_history", "memory_tree", "pgkr", "pinns_agent")
            search_space: Hyperparameter search space dictionary
            **kwargs: Additional arguments for specific strategies
                     - pgkr_top_k: for pgkr
                     - use_pgkr, - pgkr_top_k: for pinns_agent
        
        Returns:
            Prompt strategy instance
            
        Raises:
            ValueError: If strategy name is not recognized
        """
        if strategy not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown prompt strategy: {strategy}. "
                f"Available strategies: {available}"
            )
        
        prompt_class = cls._strategies[strategy]
        return prompt_class(search_space, **kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all available prompt strategies
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """
        Register a new prompt strategy
        
        Args:
            name: Strategy name
            strategy_class: Prompt strategy class (must inherit from BasePrompt)
        """
        if not issubclass(strategy_class, BasePrompt):
            raise TypeError("Strategy class must inherit from BasePrompt")
        cls._strategies[name] = strategy_class
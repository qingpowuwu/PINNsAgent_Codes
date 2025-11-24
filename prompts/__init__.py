# prompts/__init__.py

from .base_prompt import BasePrompt
from .zero_shot_prompt import ZeroShotPrompt
from .full_history_prompt import FullHistoryPrompt
from .memory_tree_prompt import MemoryTreePrompt
from .pinns_agent_prompt import PINNsAgentPrompt
from .prompt_factory import PromptFactory

__all__ = [
    "BasePrompt",
    "ZeroShotPrompt",
    "FullHistoryPrompt",
    "MemoryTreePrompt",
    "PINNsAgentPrompt",
    "PromptFactory",
]

# pinnsAgent/
# ├── prompts/
# │   ├── __init__.py
# │   ├── base_prompt.py           # Base Prompt Class
# │   ├── zero_shot_prompt.py      # Zero-shot 
# │   ├── full_history_prompt.py   # Full History
# │   ├── memory_tree_prompt.py    # Memory Tree
# │   ├── pgkr_prompt.py           # PGKR
# │   ├── pinnsagent_prompt.py     # PINNsAgent
# │   └── prompt_factory.py        # Prompt Factory
# └── ...
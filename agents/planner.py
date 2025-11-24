# agents/planner.py

import random
import json
import os
import time
from typing import Dict, List, Any, Optional
from prompts import PromptFactory
from .config_utils import fix_config, is_duplicate_config


class Planner:
    """
    Configuration generator, supports random baseline and LLM mode with various prompt strategies
    """
    
    def __init__(self, mode: str = "random", llm_client=None, log_dir=None, 
                 search_space: Dict[str, List] = None, verbose: bool = False,
                 prompt_strategy: str = "full_history", max_iterations: int = 5,
                 **prompt_kwargs):
        """
        Initialize Planner
        
        Args:
            mode: Planning mode ["random", "llm"]
            llm_client: LLM client (if using llm mode)
            log_dir: Log save directory
            search_space: Hyperparameter search space dictionary (loaded from config file)
            verbose: Whether to print detailed LLM interaction information
            prompt_strategy: Prompt strategy name ["zero_shot", "full_history", "memory_tree", "pgkr", "pinns_agent"]
            max_iterations: Maximum number of iterations (default: 5)
            **prompt_kwargs: Additional arguments for prompt strategy (e.g., use_pgkr, use_memory_tree for pinns_agent)
        """
        self.mode = mode
        self.llm_client = llm_client
        self.log_dir = log_dir
        self.verbose = verbose
        self.prompt_strategy_name = prompt_strategy
        self.max_iterations = max_iterations
        
        # Get search space from external input
        if search_space is None:
            raise ValueError("search_space must be provided, please load from config file")
        
        # Map config file keys to internal keys
        self.search_space = {
            "activation": search_space["activation"],
            "net": search_space["net"],
            "optimizer": search_space["optimizer"],
            "width": search_space["width"],
            "depth": search_space["depth"],
            "lr": search_space["lr"],
            "num_domain_points": search_space["num_domain_points"],
            "num_boundary_points": search_space["num_boundary_points"],
            "num_initial_points": search_space["num_initial_points"],
            "initializer": search_space["initializer"]
        }
        
        # Create prompt strategy instance
        if mode == "llm":
            # Pass max_iterations directly to prompt strategy
            self.prompt_strategy = PromptFactory.create_prompt(
                prompt_strategy, 
                self.search_space,
                max_iterations=self.max_iterations,
                **prompt_kwargs
            )
    
    def generate_config(self, history: List[Dict[str, Any]] = None, pde_name=None, run_id=None, iter_id=None, 
                       similar_pdes_configs: Dict[str, Dict[str, Any]] = None,
                       exploration_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate a single configuration
        
        Args:
            history: Historical experiment records, used for LLM mode decision making
            pde_name: PDE name, used for logging
            run_id: Run ID, used for logging
            iter_id: Iteration ID, used for logging
            similar_pdes_configs: Retrieved similar PDEs' configurations (for pgkr and pinns_agent strategies with PGKR)
            exploration_scores: Exploration scores from MemoryTree (for memory_tree and pinns_agent strategies)
            
        Returns:
            Single configuration dictionary
        """
        if self.mode == "random":
            return self._generate_random_config()
        elif self.mode == "llm":
            return self._generate_llm_config(history, pde_name, run_id, iter_id, 
                                            similar_pdes_configs, exploration_scores)
        else:
            raise ValueError(f"Unsupported planning mode: {self.mode}")
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """Traditional random search as baseline"""
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return config

    def _generate_llm_config(self, history: List[Dict[str, Any]] = None, pde_name=None, run_id=None, iter_id=None,
                            similar_pdes_configs: Dict[str, Dict[str, Any]] = None,
                            exploration_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate configuration based on LLM and historical records"""
        if not self.llm_client:
            raise ValueError("LLM client not configured, cannot use llm mode")
        
        # Set iteration for PINNsAgent strategy
        if self.prompt_strategy_name == "pinns_agent" and hasattr(self.prompt_strategy, 'set_iteration'):
            self.prompt_strategy.set_iteration(iter_id)
        
        max_retries = 30
        
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    print(f"LLM config generation attempt {attempt + 1}/{max_retries}")
                
                # Build prompt using strategy
                system_prompt = self.prompt_strategy.get_system_prompt()
                
                # Build user prompt - pass different parameters based on strategy and iteration
                if self.prompt_strategy_name == "pinns_agent":
                    # PINNsAgent: iteration-aware, may use PGKR (iter 1 only) and MemoryTree
                    # For iter 1: pass similar_pdes_configs if use_pgkr is enabled
                    # For iter 2+: don't pass similar_pdes_configs (set to None)
                    user_prompt = self.prompt_strategy.build_user_prompt(
                        history=history,
                        similar_pdes_configs=similar_pdes_configs if iter_id == 1 else None,
                        exploration_scores=exploration_scores
                    )
                elif self.prompt_strategy_name == "pgkr":
                    # PGKR: only uses PGKR
                    user_prompt = self.prompt_strategy.build_user_prompt(
                        history=history,
                        similar_pdes_configs=similar_pdes_configs
                    )
                elif self.prompt_strategy_name == "memory_tree":
                    # MemoryTree: only uses exploration scores
                    user_prompt = self.prompt_strategy.build_user_prompt(
                        history=history,
                        exploration_scores=exploration_scores
                    )
                else:
                    # Other strategies: only use history
                    user_prompt = self.prompt_strategy.build_user_prompt(
                        history=history
                    )
                
                if attempt > 0:
                    user_prompt += f"\n\nNote: Please ensure all parameter values are within the specified valid ranges. This is attempt {attempt + 1}."
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Call LLM
                response = self.llm_client.chat_completion(messages, temperature=0.7)
                
                try:
                    # Parse LLM response
                    config = self._parse_llm_response(response)
                    
                    # Validate configuration
                    if self._validate_config(config):
                        # Check for duplicate configurations in history using the utility function
                        if is_duplicate_config(config, history, self.search_space):
                            if self.verbose or attempt > 5:
                                print(f"⚠️ LLM generated duplicate config (attempt {attempt + 1}/{max_retries}), retrying...")
                            self._save_llm_conversation(messages, response, pde_name, run_id, iter_id, attempt + 1, success=False, reason="duplicate")
                            continue
                        
                        if self.verbose:
                            print(f"✓ LLM config generation successful (attempt {attempt + 1}/{max_retries})")
                        else:
                            # Non-verbose mode: only print success message if not first attempt
                            if attempt > 0:
                                print(f"✓ LLM config generation successful (attempt {attempt + 1}/{max_retries})")
                        
                        # Save successful conversation
                        self._save_llm_conversation(messages, response, pde_name, run_id, iter_id, attempt + 1, success=True)
                        return config
                    else:
                        # Try to fix the configuration
                        fixed_config = fix_config(config, self.search_space, verbose=self.verbose)
                        
                        # Validate fixed configuration
                        if self._validate_config(fixed_config):
                            # Check for duplicate after fix using the utility function
                            if is_duplicate_config(fixed_config, history, self.search_space):
                                if self.verbose or attempt > 5:
                                    print(f"⚠️ Fixed config is duplicate (attempt {attempt + 1}/{max_retries}), retrying...")
                                self._save_llm_conversation(messages, response, pde_name, run_id, iter_id, attempt + 1, success=False, reason="duplicate_after_fix")
                                continue
                            
                            if self.verbose or attempt > 0:
                                print(f"✓ LLM config auto-fixed and validated (attempt {attempt + 1}/{max_retries})")
                            
                            # Save successful conversation (with fix note)
                            self._save_llm_conversation(messages, response, pde_name, run_id, iter_id, attempt + 1, success=True)
                            return fixed_config
                        else:
                            if self.verbose:
                                print(f"✗ LLM generated invalid config and fix failed (attempt {attempt + 1}/{max_retries}), retrying...")
                            # Save failed conversation
                            self._save_llm_conversation(messages, response, pde_name, run_id, iter_id, attempt + 1, success=False, reason="invalid_after_fix")
                            continue
                except Exception as parse_error:
                    if self.verbose:
                        print(f"✗ Failed to parse LLM response (attempt {attempt + 1}/{max_retries}): {parse_error}, retrying...")
                    # Save failed conversation
                    self._save_llm_conversation(messages, response, pde_name, run_id, iter_id, attempt + 1, success=False, reason=f"parse_error: {parse_error}")
                    continue
                        
            except Exception as e:
                if self.verbose:
                    print(f"✗ LLM config generation failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                # If even response doesn't exist, create an error record
                error_response = f"Error: {str(e)}"
                self._save_llm_conversation(messages if 'messages' in locals() else [], error_response, pde_name, run_id, iter_id, attempt + 1, success=False, reason=f"exception: {e}")
                continue
        
        # If all retries fail, raise exception instead of fallback
        raise RuntimeError(f"LLM failed to generate valid config after {max_retries} attempts")
    
    def _save_llm_conversation(self, messages, response, pde_name, run_id, iter_id, attempt, success=True, reason=None):
        """Save LLM conversation to file - organized by iter folders"""
        if not self.log_dir:
            return
        
        # Create conversation log directory - organized by iter folders
        iter_conversation_dir = os.path.join(
            self.log_dir, 
            f"{pde_name}_run_{run_id}", 
            "llm_conversations",
            f"iter_{iter_id}"
        )
        os.makedirs(iter_conversation_dir, exist_ok=True)
        
        # Save conversation
        conversation_log = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pde_name": pde_name,
            "run_id": run_id,
            "iter_id": iter_id,
            "attempt": attempt,
            "prompt_strategy": self.prompt_strategy_name,
            "success": success,
            "messages": messages,
            "response": response
        }
        
        if not success and reason:
            conversation_log["failure_reason"] = reason
        
        # Simplified filename: attempt_{attempt}.json or attempt_{attempt}_failed.json
        if success:
            log_file = os.path.join(iter_conversation_dir, f"attempt_{attempt}.json")
        else:
            log_file = os.path.join(iter_conversation_dir, f"attempt_{attempt}_failed.json")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract configuration"""
        # Clean markdown code block markers
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]  # Remove '```json'
        if response.startswith('```'):
            response = response[3:]   # Remove '```'
        if response.endswith('```'):
            response = response[:-3]  # Remove ending '```'
        response = response.strip()
        
        try:
            # Try to parse JSON directly
            config = json.loads(response)
            return config
        except:
            # If direct parsing fails, try to extract JSON part
            try:
                # Find JSON block
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                    config = json.loads(json_str)
                    return config
            except:
                pass
            
            # If still fails, try line-by-line parsing
            try:
                config = {}
                lines = response.split('\n')
                for line in lines:
                    if ':' in line and any(key in line for key in self.search_space.keys()):
                        # Try to extract key-value pairs
                        parts = line.split(':')
                        if len(parts) >= 2:
                            key = parts[0].strip().strip('"').strip("'")
                            value_str = parts[1].strip().rstrip(',').strip()
                            
                            # Try to parse value
                            if value_str.startswith('"') and value_str.endswith('"'):
                                value = value_str[1:-1]
                            elif value_str.startswith("'") and value_str.endswith("'"):
                                value = value_str[1:-1]
                            else:
                                try:
                                    value = float(value_str)
                                except:
                                    value = value_str
                            
                            if key in self.search_space:
                                config[key] = value
                
                if config:
                    return config
            except:
                pass
            
            raise ValueError("Unable to parse LLM response")
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate if configuration is valid"""
        if not isinstance(config, dict):
            return False
        
        # Check if necessary keys exist and values are valid
        for key, valid_values in self.search_space.items():
            if key not in config:
                if self.verbose:
                    print(f"Missing required parameter: {key}")
                return False
            
            value = config[key]
            
            # Numeric parameters list (need numeric comparison instead of type matching)
            numeric_params = ["lr", "width", "depth", "num_domain_points", 
                            "num_boundary_points", "num_initial_points"]
            
            if key in numeric_params:
                # Numeric parameters: convert to float for comparison
                try:
                    # Convert value to float
                    if isinstance(value, str):
                        float_value = float(value)
                    elif isinstance(value, (int, float)):
                        float_value = float(value)
                    else:
                        if self.verbose:
                            print(f"Parameter {key} value {value} has incorrect type, should be numeric")
                        return False
                    
                    # Convert valid values list to float for comparison
                    valid_floats = [float(v) for v in valid_values]
                    
                    # Use numeric comparison (considering floating point precision)
                    if not any(abs(float_value - v) < 1e-10 for v in valid_floats):
                        if self.verbose:
                            print(f"Parameter {key} value {value} (={float_value}) not in valid range: {valid_values}")
                        return False
                        
                except (ValueError, TypeError) as e:
                    if self.verbose:
                        print(f"Parameter {key} value {value} cannot be parsed as numeric: {e}")
                    return False
            else:
                # String parameters: strict matching
                if value not in valid_values:
                    if self.verbose:
                        print(f"Parameter {key} value {value} not in valid range: {valid_values}")
                    return False
        
        return True
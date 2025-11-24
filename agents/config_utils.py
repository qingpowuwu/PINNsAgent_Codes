# agents/config_utils.py

from typing import Dict, List, Any


def fix_config(config: Dict[str, Any], search_space: Dict[str, List], verbose: bool = False) -> Dict[str, Any]:
    """
    Fix invalid configuration by replacing invalid values with nearest valid values
    
    - For numeric parameters: find the closest valid value
    - For string parameters:  use the first valid value if invalid
    - Missing parameters: keep unchanged
    
    Args:
        config: Configuration to fix
        search_space: Valid search space
        verbose: Whether to print fix information
        
    Returns:
        Fixed configuration
    """
    fixed_config = config.copy()
    
    numeric_params = ["lr", "width", "depth", "num_domain_points", 
                     "num_boundary_points", "num_initial_points"]
    
    for key, valid_values in search_space.items():
        if key not in fixed_config:
            continue
            
        value = fixed_config[key]
        
        if key in numeric_params:
            # Numeric parameters: find nearest valid value
            try:
                if isinstance(value, str):
                    float_value = float(value)
                elif isinstance(value, (int, float)):
                    float_value = float(value)
                else:
                    if verbose:
                        print(f"⚠ Parameter {key} has invalid type, using first valid value")
                    fixed_config[key] = valid_values[0]
                    continue
                
                valid_floats = [float(v) for v in valid_values]
                
                # Check if already valid
                if any(abs(float_value - v) < 1e-10 for v in valid_floats):
                    # Ensure type consistency
                    if isinstance(valid_values[0], int):
                        fixed_config[key] = int(float_value)
                    else:
                        fixed_config[key] = float_value
                    continue
                
                # Find nearest valid value
                nearest_value = min(valid_floats, key=lambda x: abs(x - float_value))
                
                if verbose:
                    print(f"⚠ Parameter {key}: {value} → {nearest_value} (nearest valid value)")
                
                # Maintain original type
                if isinstance(valid_values[0], int):
                    fixed_config[key] = int(nearest_value)
                else:
                    fixed_config[key] = nearest_value
                    
            except (ValueError, TypeError) as e:
                if verbose:
                    print(f"⚠ Parameter {key} cannot be parsed, using first valid value: {e}")
                fixed_config[key] = valid_values[0]
        else:
            # String parameters: check if valid, use first valid value if not
            if value not in valid_values:
                if verbose:
                    print(f"⚠ Parameter {key}: {value} → {valid_values[0]} (not in valid range)")
                fixed_config[key] = valid_values[0]
    
    return fixed_config


def is_duplicate_config(config: Dict[str, Any], history: List[Dict[str, Any]], 
                       search_space: Dict[str, List]) -> bool:
    """
    Check if config is duplicate of any in history
    
    Args:
        config: Configuration to check
        history: Historical experiment records
        search_space: Valid search space (used to filter hyperparameters)
        
    Returns:
        True if duplicate, False otherwise
    """
    if not history:
        return False
    
    # Get only the hyperparameter keys (exclude task, pde_list, etc.)
    config_params = {k: v for k, v in config.items() if k in search_space}
    
    for past_exp in history:
        past_config = past_exp.get("config", {})
        past_params = {k: v for k, v in past_config.items() if k in search_space}
        
        # Compare configurations
        if configs_equal(config_params, past_params):
            return True
    
    return False


def configs_equal(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Compare two configurations for equality (handling numeric precision)
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        True if equal, False otherwise
    """
    if set(config1.keys()) != set(config2.keys()):
        return False
    
    numeric_params = ["lr", "width", "depth", "num_domain_points", 
                     "num_boundary_points", "num_initial_points"]
    
    for key in config1.keys():
        val1 = config1[key]
        val2 = config2[key]
        
        if key in numeric_params:
            # Numeric comparison with tolerance
            try:
                if abs(float(val1) - float(val2)) > 1e-10:
                    return False
            except:
                if val1 != val2:
                    return False
        else:
            # String comparison
            if val1 != val2:
                return False
    
    return True
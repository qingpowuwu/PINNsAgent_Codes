# agents/programmer.py

import os
import subprocess
import yaml

class Programmer:
    def __init__(self, template_fixed: dict, train_dir: str, conda_python: str, verbose: bool = False):
        """
        Initialize Programmer
        
        Args:
            template_fixed: Fixed parameter configuration dictionary (must be provided)
            train_dir:      Training code directory (where benchmark.py is located, must be provided)
            conda_python:   Conda environment's Python executable path (must be provided)
            verbose:        Whether to print detailed training process output
        """
        self.fixed = template_fixed
        self.train_dir = train_dir
        self.conda_python = conda_python
        self.verbose = verbose

    def write_yaml(self, config: dict, out_path: str):
        """Merge self.fixed and config, write as yaml configuration file"""
        yaml_dict = dict(self.fixed)
        yaml_dict.update(config) # config = {'task': 'Burgers1D', '0_run_time': 2310.0, '2_mse': 6.31e-05, '1_train_loss': 2.51e-05, '4_l2rel': 0.013, '3_mxe': 0.0853, '5_crmse': 7.87e-05, '2_activation': 'gaussian', '3_net': 'fnn', '4_optimizer': 'lbfgs', '6_loss_weight': 'none', '7_1_width': 128, '7_2_depth': 3, '8_lr': 0.001, '9_1_domain_points': 8192, '9_2_boundary_points': 2048, '9_3_initial_points': 2024, '10_initializer': 'Glorot normal', 'iter': 20000}
        with open(out_path, "w") as f:
            yaml.dump(yaml_dict, f)
        return out_path

    def run_exp(self, yaml_path: str):
        import time
        import os
        import re
        
        yaml_path_abs = os.path.abspath(yaml_path)
        cmd = [
            self.conda_python, "benchmark.py",
            "--yaml_path", yaml_path_abs
        ]
        
        if self.verbose:
            print(f"Executing command: {' '.join(cmd)}")
            print(f"Working directory: {self.train_dir}")
            print(f"Config file path: {yaml_path_abs}")
        
        # Create log file
        log_dir = os.path.dirname(yaml_path_abs)
        log_file = os.path.join(log_dir, "training_log.txt")
        
        try:
            with open(log_file, 'w') as log_f:
                # Use Popen to get real-time output
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.train_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                mse = None
                run_time = None
                line_count = 0
                
                if self.verbose:
                    print("=== Starting real-time output (also saving to log file) ===")
                    print(f"Log file: {log_file}")
                
                # Read output in real-time
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        line = line.rstrip()
                        line_count += 1
                        
                        # Write to log file
                        log_f.write(f"{line}\n")
                        log_f.flush()  # Flush to file immediately
                        
                        # Verbose mode: print more information
                        if self.verbose:
                            # Only skip pure numeric training step lines
                            should_print = True
                            if line.strip() and line.strip()[0].isdigit() and ":" not in line:
                                should_print = False
                            
                            if should_print:
                                print(f"[{line_count:4d}] {line}")
                        else:
                            # Non-verbose mode: only show key information
                            should_print = (
                                "2_mse" in line or 
                                "0_run_time" in line or
                                "Error" in line or 
                                "Exception" in line or
                                "failed" in line or
                                "success" in line
                            )
                            
                            if should_print:
                                print(f"{line}")
                        
                        # Extract 2_mse - from wandb output (final test MSE)
                        if "2_mse" in line:
                            try:
                                # Format: wandb:            2_mse 0.33541
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "2_mse" and i + 1 < len(parts):
                                        mse_found = float(parts[i + 1])
                                        mse = mse_found
                                        if self.verbose:
                                            print(f"[Found 2_mse] {mse}")
                                        break
                            except Exception as e:
                                if self.verbose:
                                    print(f"[2_mse parsing failed] {e}")
                        
                        # Extract 0_run_time - from wandb output
                        if "0_run_time" in line:
                            try:
                                # Format: wandb:       0_run_time 9.67896
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "0_run_time" and i + 1 < len(parts):
                                        run_time_found = float(parts[i + 1])
                                        run_time = run_time_found
                                        if self.verbose:
                                            print(f"[Found 0_run_time] {run_time}")
                                        break
                            except Exception as e:
                                if self.verbose:
                                    print(f"[0_run_time parsing failed] {e}")
                
                # Wait for process to complete
                return_code = proc.wait()
                
                if self.verbose:
                    print("=== Real-time output ended ===")
                    print(f"Process return code: {return_code}")
                    print(f"Total output lines: {line_count}")
                    print(f"Final 2_mse: {mse}")
                    print(f"Final 0_run_time: {run_time}")
                    print(f"Complete log saved to: {log_file}")
                
                if return_code != 0:
                    print("✗ Training script execution failed! Please check log file for details.")
                else:
                    if self.verbose:
                        print("✓ Training script executed successfully!")
                    
                return mse if mse is not None else 1e10, run_time if run_time is not None else 0.0
                
        except Exception as e:
            print(f"✗ Exception occurred while executing training script: {e}")
            return 1e10, 0.0
# database/knowledge_base.py
from pathlib import Path
import os
import sys
current_dir = Path(__file__).parent.absolute()
pinnsagent_root = current_dir.parent
os.chdir(pinnsagent_root)
sys.path.append(str(pinnsagent_root))
import pandas as pd
from typing import Optional, Dict, Any, List

class KnowledgeBase:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # Can pre-index or organize as needed

    def get_topk_by_mse(self, pde_name: str, k: int = 3, mse_col: str = 'mse') -> List[Dict[str, Any]]:  # Updated column name
        """Return top k optimal configurations for given PDE name"""
        records = self.df[self.df['task'] == pde_name]
        if len(records) == 0:
            return []
        sorted_rows = records.sort_values(by=mse_col).head(k)
        return [row.to_dict() for _, row in sorted_rows.iterrows()]

    def get_topk_by_composite_score(self, pde_name: str, k: int = 3, 
                                     mse_weight: float = 0.7, time_weight: float = 0.3,
                                     mse_col: str = 'mse', time_col: str = 'run_time',
                                     mse_magnitude_filter: float = 3.16) -> List[Dict[str, Any]]:
        """
        Return top k configurations based on composite score of MSE and runtime
        
        Args:
            pde_name: Target PDE name
            k: Number of top configurations to return
            mse_weight: Weight for MSE (default 0.7, prioritize accuracy)
            time_weight: Weight for runtime (default 0.3)
            mse_col: Column name for MSE
            time_col: Column name for runtime
            mse_magnitude_filter: Filter out configs with MSE > best_mse * this factor (default 10^0.5 ≈ 3.16)
            
        Returns:
            List of top-k configuration dictionaries
        """
        records = self.df[self.df['task'] == pde_name].copy()
        if len(records) == 0:
            return []
        
        # Find the best (minimum) MSE
        best_mse = records[mse_col].min()
        
        # Filter: keep only configs with MSE <= best_mse * mse_magnitude_filter
        mse_threshold = best_mse * mse_magnitude_filter
        records = records[records[mse_col] <= mse_threshold].copy()
        
        if len(records) == 0:
            return []
        
        # Normalize MSE and runtime to [0, 1] range
        # Use min-max normalization
        mse_min = records[mse_col].min()
        mse_max = records[mse_col].max()
        time_min = records[time_col].min()
        time_max = records[time_col].max()
        
        # Avoid division by zero
        if mse_max - mse_min > 0:
            records['mse_normalized'] = (records[mse_col] - mse_min) / (mse_max - mse_min)
        else:
            records['mse_normalized'] = 0.0
        
        if time_max - time_min > 0:
            records['time_normalized'] = (records[time_col] - time_min) / (time_max - time_min)
        else:
            records['time_normalized'] = 0.0
        
        # Calculate composite score (lower is better)
        # Score = mse_weight * normalized_mse + time_weight * normalized_time
        records['composite_score'] = (mse_weight * records['mse_normalized'] + 
                                      time_weight * records['time_normalized'])
        
        # Sort by composite score (ascending)
        sorted_rows = records.sort_values(by='composite_score').head(k)
        
        # Convert to dict and add score information
        results = []
        for _, row in sorted_rows.iterrows():
            row_dict = row.to_dict()
            row_dict['composite_score'] = row['composite_score']
            row_dict['mse_normalized'] = row['mse_normalized']
            row_dict['time_normalized'] = row['time_normalized']
            results.append(row_dict)
        
        return results

    def list_available_pdes(self) -> List[str]:
        """Return all PDE names in the database"""
        return sorted(self.df['task'].unique())

    def add_record(self, record: Dict[str, Any]) -> None:
        """Add new experiment record (only in memory, not automatically written back to csv)"""
        # Use concat instead of append, compatible with new pandas version
        new_row = pd.DataFrame([record])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def save(self, path: Optional[str] = None):
        """Save to csv"""
        out_path = path if path is not None else self.csv_path
        self.df.to_csv(out_path, index=False)


if __name__ == "__main__":
    print("="*80)
    print("Testing KnowledgeBase")
    print("="*80)
    
    # Initialize knowledge base
    csv_path = "./data/dataset_for_retrieval.csv"
    kb = KnowledgeBase(csv_path)
    
    # Test 1: List available PDEs
    print("\n[Test 1] Available PDEs in knowledge base:")
    pdes = kb.list_available_pdes()
    print(f"Total PDEs: {len(pdes)}")
    print(f"PDE names: {pdes}")
    
    # Test 2: Get top-k configurations for a specific PDE (by MSE only)
    test_pde = "Burgers1D"
    k = 5
    print(f"\n[Test 2] Top-{k} configurations for {test_pde} (by MSE only):")
    top_configs = kb.get_topk_by_mse(pde_name=test_pde, k=k)
    
    if top_configs:
        for i, config in enumerate(top_configs, 1):
            print(f"\nRank {i}:")
            print(f"  MSE: {config['mse']:.2e}")
            print(f"  Run time: {config['run_time']:.1f}s ({config['run_time']/3600:.2f}h)")
            print(f"  Activation: {config['activation']}")
            print(f"  Network: {config['net']}")
            print(f"  Optimizer: {config['optimizer']}")
            print(f"  Width: {config['width']}, Depth: {config['depth']}")
            print(f"  Learning rate: {config['lr']}")
    else:
        print(f"No configurations found for {test_pde}")
    
    # Test 3: Get top-k configurations by composite score with MSE magnitude filter
    print(f"\n[Test 3] Top-{k} configurations for {test_pde} (by composite score with MSE filter):")
    print(f"  MSE weight: 0.85, Time weight: 0.15, MSE magnitude filter: 10^0.5x")
    top_configs_composite = kb.get_topk_by_composite_score(
        pde_name=test_pde, 
        k=k,
        mse_weight=0.85,
        time_weight=0.15,
        mse_magnitude_filter=3.16
    )
    
    if top_configs_composite:
        best_mse = top_configs_composite[0]['mse']
        print(f"  Best MSE: {best_mse:.2e}")
        print(f"  MSE threshold (10^0.5x best): {best_mse * 10:.2e}")
        
        for i, config in enumerate(top_configs_composite, 1):
            print(f"\nRank {i}:")
            print(f"  Composite Score: {config['composite_score']:.4f}")
            print(f"  MSE: {config['mse']:.2e} (normalized: {config['mse_normalized']:.4f})")
            print(f"  Run time: {config['run_time']:.1f}s ({config['run_time']/3600:.2f}h) (normalized: {config['time_normalized']:.4f})")
            print(f"  Activation: {config['activation']}")
            print(f"  Network: {config['net']}")
            print(f"  Optimizer: {config['optimizer']}")
            print(f"  Width: {config['width']}, Depth: {config['depth']}")
            print(f"  Learning rate: {config['lr']}")
    else:
        print(f"No configurations found for {test_pde}")
    
    # Test 4: Compare different weight settings
    print(f"\n[Test 4] Comparing different weight settings for {test_pde}:")
    
    weight_configs = [
        (1.0, 0.0, "MSE only"),
        (0.7, 0.3, "Balanced (MSE priority)"),
        (0.5, 0.5, "Equal weight"),
        (0.3, 0.7, "Time priority"),
        (0.0, 1.0, "Time only")
    ]
    
    for mse_w, time_w, desc in weight_configs:
        print(f"\n  {desc} (MSE: {mse_w}, Time: {time_w}):")
        results = kb.get_topk_by_composite_score(
            pde_name=test_pde,
            k=1,
            mse_weight=mse_w,
            time_weight=time_w,
            mse_magnitude_filter=3.16
        )
        if results:
            config = results[0]
            print(f"    Best: MSE={config['mse']:.2e}, Time={config['run_time']:.1f}s, Score={config['composite_score']:.4f}")
    
    # Test 5: Add a new record
    print("\n[Test 5] Adding a new record...")
    new_record = {
        'task': 'TestPDE',
        'mse': 1.23e-5,
        'run_time': 100.0,
        'activation': 'relu',
        'net': 'fnn',
        'optimizer': 'adam',
        'lr': 0.001,
        'width': 64,
        'depth': 4
    }
    
    original_size = len(kb.df)
    kb.add_record(new_record)
    new_size = len(kb.df)
    
    print(f"Original size: {original_size}")
    print(f"New size: {new_size}")
    print(f"Record added successfully: {new_size == original_size + 1}")
    
    # Test 6: Query non-existent PDE
    print("\n[Test 6] Query non-existent PDE:")
    result = kb.get_topk_by_mse("NonExistentPDE", k=1)
    print(f"Result for non-existent PDE: {result}")
    print(f"Returns empty list: {result == []}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
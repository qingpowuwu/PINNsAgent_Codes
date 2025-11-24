# agents/retriever.py
from pathlib import Path
import os
import sys
current_dir = Path(__file__).parent.absolute()
pinnsagent_root = current_dir.parent
os.chdir(pinnsagent_root)
sys.path.append(str(pinnsagent_root))
from database.pde_encoder import PDE_LABELS, feature_encoding
from database.similarity import cosine_sim

class PGKR:
    def __init__(self):
        """
        Initialize PGKR (Physics-Guided Knowledge Retrieval)
        """
        self.pde_labels = PDE_LABELS
        self.feature_encoding = feature_encoding
        self.pde_names = list(PDE_LABELS.keys())
        self.encoded_pdes, self.max_length = self._encode_pdes()
        self.X = list(self.encoded_pdes.values())
        # MSE threshold for filtering retrieved configurations.

        self.mse_threshold = 1e-0 # filter out poorly fitted PDEs
    
    def _encode_pdes(self):
        """Encode historical PDE labels as vectors (called during internal initialization)"""
        encoded_pdes = {}
        max_length = 0
        for pde, labels in self.pde_labels.items():
            encoding = []
            for label in labels:
                encoding.extend(self.feature_encoding.get(label, []))
            max_length = max(max_length, len(encoding))
            encoded_pdes[pde] = encoding
        # Pad
        for pde, encoding in encoded_pdes.items():
            encoded_pdes[pde] = encoding + [0] * (max_length - len(encoding))
        return encoded_pdes, max_length

    def encode_labels(self, labels):
        """Encode new PDE labels as vectors"""
        encoding = []
        for label in labels: # ['Burgers', 'parabolic', '1d', 'nonlinear', 'time-dependent', 'dirichlet', 'initial-condition', 'constant-coefficient', 'short-time', 'simple-geometry']
            encoding.extend(self.feature_encoding.get(label, []))
        return encoding + [0] * (self.max_length - len(encoding)) # len(encoding + [0] * (self.max_length - len(encoding))) = 34 
    
    def topk_similar_pdes(self, new_pde_labels, k=5):
        new_encoding = self.encode_labels(new_pde_labels)
            # new_pde_labels = ['Burgers', 'parabolic', '1d', 'nonlinear', 'time-dependent', 'dirichlet', 'initial-condition', 'constant-coefficient', 'short-time', 'simple-geometry']
            # new_encoding      = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
            # len(new_encoding) = 34
        sims = [cosine_sim(new_encoding, x) for x in self.X]
            # self.X = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, ...], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, ...], ...]
            # len(self.X) = 21
            # len(self.X[0]) = 34   
        idxs = sorted(range(len(sims)), key=lambda i: -sims[i]) # [0, 1, 15, 18, 7, 11, 14, 17, 6, 20, 8, 9, 13, 10, 16, 2, 12, 19, 3, 4, 5]
            # len(idxs) = 21
        return [(self.pde_names[i], sims[i]) for i in idxs[:k]] # [('Burgers1D', 1.0), ('Burgers2D', 0.8), ('KuramotoSivashinskyEquation', 0.8)]

    def topk_similar_pdes_exclude_target(self, new_pde_labels, target_pde, k=5):
        """
        Retrieve similar PDEs, but exclude target PDE (simulate new PDE scenario)
        
        Args:
            new_pde_labels: Label list of new PDE
            target_pde: Target PDE name to exclude
            k: Number of top-k results to return
            
        Returns:
            [(pde_name, similarity), ...] List of similar PDEs excluding target PDE
        """
        new_encoding = self.encode_labels(new_pde_labels)
        sims = [cosine_sim(new_encoding, x) for x in self.X]
        
        # Create list of (index, similarity, PDE name) and exclude target PDE
        candidates = []
        for i, sim in enumerate(sims):
            pde_name = self.pde_names[i]
            if pde_name != target_pde:  # Exclude target PDE
                candidates.append((i, sim, pde_name))
        
        # Sort by similarity in descending order
        candidates.sort(key=lambda x: -x[1])
        
        # Return top-k results
        return [(candidate[2], candidate[1]) for candidate in candidates[:k]]
    
    def retrieve_similar_pdes_configs(self, target_pde, kb, pgkr_top_k=3, simulate_new_pde=False, 
                                     use_composite_score=False, mse_weight=0.7, time_weight=0.3,
                                     mse_magnitude_filter=3.16):
        """
        Retrieve best configurations from pgkr_top_k most similar PDEs
        
        Args:
            target_pde: Target PDE name
            kb: KnowledgeBase instance
            pgkr_top_k: Number of similar PDEs to retrieve
            simulate_new_pde: Whether to exclude target PDE from retrieval
            use_composite_score: Whether to use composite score (MSE + time) (default False)
            mse_weight: Weight for MSE in composite score (default 0.7)
            time_weight: Weight for runtime in composite score (default 0.3)
            mse_magnitude_filter: Filter out configs with MSE > best_mse * this factor (default 10^0.5 ≈ 3.16)
            
        Returns:
            Dictionary of {pde_name: {'similarity': float, 'best_config': dict, 'best_mse': float, 'run_time': float, 'composite_score': float}}
            Returns None if target PDE not found in labels
        """
        if target_pde not in PDE_LABELS:
            return None
        
        # Get labels for target PDE
        target_labels = PDE_LABELS[target_pde]
        
        # Find top-pgkr_top_k similar PDEs
        if simulate_new_pde:
            similar_pdes = self.topk_similar_pdes_exclude_target(target_labels, target_pde, k=pgkr_top_k)
        else:
            similar_pdes = self.topk_similar_pdes(target_labels, k=pgkr_top_k)
        
        # Retrieve best configuration for each similar PDE
        similar_configs = {}
        for pde_name, similarity in similar_pdes:
            # Get top-1 best configuration (lowest MSE & Time)
            if use_composite_score:
                top_configs = kb.get_topk_by_composite_score(
                    pde_name, 
                    k=1,
                    mse_weight=mse_weight,
                    time_weight=time_weight,
                    mse_magnitude_filter=mse_magnitude_filter
                )
            else:
                # Get top-1 best configuration (lowest MSE)
                top_configs = kb.get_topk_by_mse(pde_name, k=1)
            
            if top_configs:
                best_record = top_configs[0]
                best_mse = best_record.get('mse', float('inf'))
                
                # Filter by MSE threshold
                if best_mse > self.mse_threshold:
                    print(f"  ⚠ Skipping {pde_name}: MSE {best_mse:.2e} exceeds threshold {self.mse_threshold:.2e}")
                    continue
                
                # Extract only hyperparameters
                hyperparams = ['activation', 'net', 'optimizer', 'lr', 'width', 'depth',
                             'num_domain_points', 'num_boundary_points', 'num_initial_points', 'initializer']
                best_config = {k: v for k, v in best_record.items() if k in hyperparams}
                
                # Extract metrics
                run_time = best_record.get('run_time', None)
                composite_score = best_record.get('composite_score', None)
                
                result_dict = {
                    'similarity': similarity,
                    'best_config': best_config,
                    'best_mse': best_mse,
                    'run_time': run_time
                }
                
                # Add composite score if available
                if composite_score is not None:
                    result_dict['composite_score'] = composite_score
                    result_dict['mse_normalized'] = best_record.get('mse_normalized', None)
                    result_dict['time_normalized'] = best_record.get('time_normalized', None)
                
                similar_configs[pde_name] = result_dict
        
        return similar_configs if similar_configs else None


if __name__ == "__main__":
    from database.knowledge_base import KnowledgeBase
    
    print("="*80)
    print("Testing PGKR")
    print("="*80)
    
    # Initialize PGKR
    pgkr = PGKR()
    
    # Test 1: Encode PDE labels
    print("\n[Test 1] Encoding PDE labels:")
    test_labels = ['Burgers', 'parabolic', '1d', 'nonlinear', 'time-dependent', 
                   'dirichlet', 'initial-condition', 'constant-coefficient', 
                   'short-time', 'simple-geometry']
    encoded = pgkr.encode_labels(test_labels)
    print(f"Test labels: {test_labels}")
    print(f"Encoded vector length: {len(encoded)}")
    print(f"First 10 values: {encoded[:10]}")
    
    # Test 2: Find top-k similar PDEs (include target)
    print("\n[Test 2] Find top-5 similar PDEs (including target):")
    target_pde = "Burgers1D"
    pgkr_top_k = 3
    target_labels = PDE_LABELS[target_pde]
    similar_pdes = pgkr.topk_similar_pdes(target_labels, k=pgkr_top_k)
    
    print(f"Target PDE: {target_pde}")
    print(f"Similar PDEs:")
    for i, (pde_name, similarity) in enumerate(similar_pdes, 1):
        print(f"  {i}. {pde_name:30s} | Similarity: {similarity:.4f}")
    
    # Test 3: Find top-k similar PDEs (exclude target - simulate new PDE)
    print("\n[Test 3] Find top-5 similar PDEs (excluding target - simulate new PDE):")
    similar_pdes_excl = pgkr.topk_similar_pdes_exclude_target(target_labels, target_pde, k=5)
    
    print(f"Target PDE: {target_pde} (excluded)")
    print(f"Similar PDEs:")
    for i, (pde_name, similarity) in enumerate(similar_pdes_excl, 1):
        print(f"  {i}. {pde_name:30s} | Similarity: {similarity:.4f}")
    
    # Test 4: Retrieve similar PDEs with configurations from knowledge base
    print("\n[Test 4] Retrieve configurations from similar PDEs:")
    csv_path = "./data/dataset_for_retrieval.csv"
    kb = KnowledgeBase(csv_path)
    
    # Test 4A: Without excluding target (use_composite_score=False, default)
    print(f"\nScenario A: Include target PDE ({target_pde}) - Use MSE only (default):")
    similar_configs = pgkr.retrieve_similar_pdes_configs(
        target_pde=target_pde,
        kb=kb,
        pgkr_top_k=pgkr_top_k,
        simulate_new_pde=False
    )
    
    if similar_configs:
        for pde_name, info in similar_configs.items():
            print(f"\n  PDE: {pde_name}")
            print(f"    Similarity: {info['similarity']:.4f}")
            print(f"    Best MSE: {info['best_mse']:.2e}")
            print(f"    Run Time: {info.get('run_time', 'N/A')}")
            print(f"    Best Config: {info['best_config']}")
    else:
        print("  No configurations found (all exceeded MSE threshold)")
    
    # Test 4B: With excluding target (simulate new PDE, use_composite_score=False)
    print(f"\nScenario B: Exclude target PDE - simulate new PDE ({target_pde}) - Use MSE only:")
    similar_configs_new = pgkr.retrieve_similar_pdes_configs(
        target_pde=target_pde,
        kb=kb,
        pgkr_top_k=pgkr_top_k,
        simulate_new_pde=True,
        use_composite_score=True,
    )
    
    if similar_configs_new:
        for pde_name, info in similar_configs_new.items():
            print(f"\n  PDE: {pde_name}")
            print(f"    Similarity: {info['similarity']:.4f}")
            print(f"    Best MSE: {info['best_mse']:.2e}")
            print(f"    Run Time: {info.get('run_time', 'N/A')}")
            print(f"    Best Config: {info['best_config']}")
    else:
        print("  No configurations found (all exceeded MSE threshold)")
    
    
    # Test 4C: Exclude target with composite score
    print(f"\nScenario C: Exclude target PDE - simulate new PDE ({target_pde}) - Use composite score:")
    similar_configs_composite_new = pgkr.retrieve_similar_pdes_configs(
        target_pde=target_pde,
        kb=kb,
        pgkr_top_k=pgkr_top_k,
        simulate_new_pde=True,
        use_composite_score=True,
        mse_weight=0.7,
        time_weight=0.3
    )
    
    if similar_configs_composite_new:
        for pde_name, info in similar_configs_composite_new.items():
            print(f"\n  PDE: {pde_name}")
            print(f"    Similarity: {info['similarity']:.4f}")
            print(f"    Best MSE: {info['best_mse']:.2e}")
            print(f"    Run Time: {info.get('run_time', 'N/A')}")
            if 'composite_score' in info:
                print(f"    Composite Score: {info['composite_score']:.4f}")
                print(f"    MSE Normalized: {info.get('mse_normalized', 'N/A')}")
                print(f"    Time Normalized: {info.get('time_normalized', 'N/A')}")
            print(f"    Best Config: {info['best_config']}")
    else:
        print("  No configurations found (all exceeded MSE threshold)")
    
    # Test 5: Test with PDE not in labels
    print("\n[Test 5] Test with non-existent PDE:")
    result = pgkr.retrieve_similar_pdes_configs(
        target_pde="NonExistentPDE",
        kb=kb,
        pgkr_top_k=3,
        simulate_new_pde=True
    )
    print(f"Result: {result}")
    print(f"Returns None: {result is None}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
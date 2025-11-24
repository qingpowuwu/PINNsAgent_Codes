# 1d (by parsing --pde_type)
cd /mnt/pfs/world_foundational_model/qingpo.wuwu/Project_0_PINNsAgent/1_Ours/PINNsAgent/pinnsagent_progress-open_source-official
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
python run_experiments.py \
    --mode llm \
    --prompt_strategy pinns_agent \
    --use_pgkr \
    --pgkr_top_k 3 \
    --use_memory_tree \
    --use_uct \
    --pde_type 1d \
    --num_iters 5 \
    --num_runs 3 \
    --device 1 \
    --iter 10000 \
    --output_dir "./output/2025_5_30" \
    --run_name "5_pinns_agent/1d" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/mnt/pfs/world_foundational_model/qingpo.wuwu/Project_0_PINNsAgent/1_Ours/PINNsAgent/pinnsagent_progress-open_source-official/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
# 1d (by parsing --pde_type)
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
python run_experiments.py \
    --mode llm \
    --prompt_strategy full_history \
    --pde_type 1d \
    --num_iters 5 \
    --num_runs 3 \
    --device 1 \
    --iter 10000 \
    --output_dir "./output/2025_5_30" \
    --run_name "2_full_history/1d" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
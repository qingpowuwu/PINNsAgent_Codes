# (1) Generate the configuration YAML files for random search experiments
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
conda activate pinnsagent
bash ./scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/1_generate_random_search_yaml.sh

# (2) Generate the batch shell scripts to run experiments
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
conda activate pinnsagent
python scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/2_generate_batches_shell.py

# (3) Run multiple batches experiments
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
cd scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/2_run_experiments_per_batchs
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/0607-cluster/miniconda3/envs/pinnsagent
clear

# Parallel execution (all batches simultaneously)
./run_all_batches.sh 1-2 --parallel   # run batches 1 and 2 in parallel
./run_all_batches.sh 3-4 --parallel   # run batches 3 and 4 in parallel
./run_all_batches.sh 5-6 --parallel   # run batches 5 and 6 in parallel

./run_all_batches.sh 1 # h20-9 terminal
./run_all_batches.sh 2 # h20-6 tmux agent-20 windows 0
./run_all_batches.sh 3 # h20-6 tmux agent-20 windows 1
./run_all_batches.sh 4 # h20-6 tmux agent-20 windows 2
./run_all_batches.sh 5
./run_all_batches.sh 6
./run_all_batches.sh 7
./run_all_batches.sh 8
./run_all_batches.sh 9
./run_all_batches.sh 10
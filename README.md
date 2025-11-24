# pinnsagent_progress

# Environment Setup (Server cuda121-torch231)

Setting up the environment for PINNsAgent:

```bash
conda create -n pinnsagent python=3.10
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
# Install pytorch
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
pip install numpy==1.23.5
pip install setuptools==69.5.1
# Install CUDA Toolkit 11.8
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
python -c "import tensorflow as tf; print(tf.__version__)"
```

Then you need to follow [PINNacle's README](./pinnacle/README.md) to generate the structured dataset for retrieval (this will take around 2-3 weeks on a single 8xA100 setup, so we have already provided it for you at `./data/dataset_for_retrieval.csv`)

# Experiment Running Instructions

Before running experiments, you need to configure the LLM API settings:

1. Open `configs/default_config.yaml`
2. Update the LLM configuration with your API credentials:

```yaml
llm_config:
  api_key: "your-api-key-here"  # Replace with your OpenAI API key
  base_url: "https://api.openai.com/v1"  # OpenAI official endpoint or your custom endpoint
```

**Important:** You must specify either `--pde_type` OR `--pde_name`, but not both.

## (1) Run LLM experiments for a single PDE

You can parse `--pde_name` to run experiments for a specific PDE:

```bash
# Run a single PDE experiment (by parsing --pde_name)
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
python run_experiments.py \
    --mode llm \
    --prompt_strategy zero_shot \
    --pde_name "Burgers1D" \
    --num_iters 5 \
    --num_runs 10 \
    --device 0 \
    --iter 20000 \
    --output_dir "./output/2025_5_30" \
    --run_name "1_zero_shot-test" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagenet_progress/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
```

## (2) Run LLM experiments for all PDEs of a specific dimension

You can parse `--pde_type` to run experiments for all PDEs of a specific dimension

```bash
# 1d (by parsing --pde_type)
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
python run_experiments.py \
    --mode llm \
    --prompt_strategy zero_shot \
    --pde_type 1d \
    --num_iters 5 \
    --num_runs 10 \
    --device 1 \
    --iter 200 \
    --output_dir "./output/2025_5_30" \
    --run_name "1_zero_shot/1d" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagenet_progress/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
```

```bash
# 2d (by parsing --pde_type)
python run_experiments.py \
    --mode llm \
    --prompt_strategy zero_shot \
    --pde_type 2d \
    --num_iters 5 \
    --num_runs 10 \
    --device 2 \
    --iter 20000 \
    --output_dir "./output/2025_5_30" \
    --run_name "1_zero_shot/2d" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagenet_progress/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
```

```bash
# 3d (by parsing --pde_type)
python run_experiments.py \
    --mode llm \
    --prompt_strategy zero_shot \
    --pde_type 3d \
    --num_iters 5 \
    --num_runs 10 \
    --device 3 \
    --iter 20000 \
    --output_dir "./output/2025_5_30" \
    --run_name "1_zero_shot/3d" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagenet_progress/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
```

```bash
# nd (by parsing --pde_type)
python run_experiments.py \
    --mode llm \
    --prompt_strategy zero_shot \
    --pde_type nd \
    --num_iters 5 \
    --num_runs 10 \
    --device 4 \
    --iter 20000 \
    --output_dir "./output/2025_5_30" \
    --run_name "1_zero_shot/nd" \
    --csv_path "./data/dataset_for_retrieval.csv" \
    --train_code_dir "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagenet_progress/pinnacle" \
    --conda_python "/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python" \
    --simulate_new_pde
```

You can also specify different prompt strategies by parsing `--prompt_strategy`, such as `zero_shot`, `full_history`, `memory_tree`, `pgkr` and `pinns_agent`.

## (3) Run all experiments using shell scripts

You can run scripts to run PINNsAgent experiments over PDEs from `./src/2025_6_01`:

```bash
# Grant execution permission
chmod +x ./src/2025_6_01/1_pinns_agent-pinnsagent_prompt-use_pkgr-use_memory_tree--use_uct-10000iters.sh
# Run the experiment script
./src/1_pinns_agent-pinnsagent_prompt-use_pkgr-use_memory_tree--use_uct-10000iters.sh
```
# PINNacle Qing

This project is based on [PINNacle](https://github.com/i207M/PINNacle.git).

# Experiment Scripts

First, you need to replace all occurrences of `/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source` with your own pinnsagent path.

## To run a single experiment (for training a single PDE)

For simplicity, we `only` use config file to run experiments, if `--yaml_path` is specified, the arguments in the config file will be used, otherwise the default arguments defined in `benchmark.py` will be used.

To run a simple experiment, you can use the following command: `python benchmark.py [--yaml_path yaml_path]`

```bash
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
python benchmark.py \
 --yaml_path "./config/1_1-test_single_pde.yaml"
```

* The results of the experiment will be saved in the `./{args.output_dir}/{args.name}` directory
* where `args.output_dir` and `args.name` are defined in the .yaml config file.

## Database Preparation for PINNsAgent

### (1) Generate Random Search .yaml files

To create database for PINNsAgent, we first generate a large number of PDEs with different hyperparameters. This is done by running a random search over the hyperparameter space. 

Generate random search configuration files

```bash
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
bash ./scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/1_generate_random_search_yaml.sh
```

This script generates 10 batches (30 configs per PDE per batch) and saves them to `scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/1_ICML_2025_configs/` directory, structured as follows:

```
scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/1_ICML_2025_configs/
├── Batch_1/
│   ├── 1d/
│   │   ├── Burgers1D-30/
│   │   │   ├── train_1.yaml
│   │   │   ├── train_2.yaml
│   │   │   └── ...
│   │   └── Wave1D-30/
│   ├── 2d/
│   │   ├── Burgers2D-30/
│   │   ├── Wave2D_Heterogeneous-30/
│   │   └── ...
│   ├── 3d/
│   │   └── Poisson3D_ComplexGeometry-30/
│   └── nd/
│       ├── PoissonND-30/
│       └── HeatND-30/
├── Batch_2/
│   └── ...
└── Batch_10/
    └── ...
```

### (2) Generate Batch Training .sh files

Then you can generate training shell `.sh` scripts for all batches

```bash
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
python scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/2_generate_batches_shell.py
```

* Do remember to update `config_base`, `script_base`, `work_dir`, and `log_base` in the script to your paths before running it, as it will generate the .sh files in the same directory.

This creates a complete training script system:
```
scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/2_run_experiments_per_batchs/
├── Batch_1/
│   ├── 1_train_1d.sh        # Train 1D PDEs (GPU 0)
│   ├── 2_train_2d.sh        # Train 2D PDEs (GPU 0-7, parallel)
│   ├── 3_train_3d.sh        # Train 3D PDEs (GPU 1)
│   ├── 4_train_nd.sh        # Train ND PDEs (GPU 7)
│   └── run_batch_1.sh       # Run all 4 scripts in parallel
├── Batch_2/ ... Batch_10/
├── generate_summary.py      # Generate training summary
└── run_all_batches.sh       # Master control script
```

**Features:**
- Automatic GPU allocation (1D: GPU0, 2D: GPU0-7, 3D: GPU1, ND: GPU7)
- Resume from checkpoint (skips `*_ok.yaml` files)
- Detailed logging in `log/2025_05_15/RandomSearch_Datasets_high_quality/`

### (3) Run the Batch Training .sh files

After generating the .sh files, you can run them to start the training.

#### (3).1 Run multiple batches experiments

You can run the following command to run multiple batches experiments:

```bash
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
cd scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/2_run_experiments_per_batchs
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent

# Serial execution (one batch after another)
./run_all_batches.sh 1-5     # Run batches 1 to 5
./run_all_batches.sh 6-10    # Run batches 6 to 10
```

#### (3).2 Run single batch experiments

Or you can run the following command to run single batch experiments:

```bash
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle
cd scripts/2025_05_15/1_RandomSearch_Datasets_high_quality/2_run_experiments_per_batchs
source /gpfs/0607-cluster/miniconda3/etc/profile.d/conda.sh
conda activate pinnsagent
# run single batch experiments
./run_all_batches.sh 1
./run_all_batches.sh 2
./run_all_batches.sh 3
./run_all_batches.sh 4
./run_all_batches.sh 5
./run_all_batches.sh 6
./run_all_batches.sh 7
./run_all_batches.sh 8
./run_all_batches.sh 9
./run_all_batches.sh 10
```

#### Monitor progress:**

```bash
# View master log
tail -f log/2025_05_15/RandomSearch_Datasets_high_quality/master.log

# View specific batch logs
tail -f log/2025_05_15/RandomSearch_Datasets_high_quality/Batch_1/1d.log
tail -f log/2025_05_15/RandomSearch_Datasets_high_quality/Batch_1/2d.log
```

or you can also track them by running `3_1_get_statistics_for-1-batch.py` or `3_1_get_statistics_for-10-batch.py`

**Results:**
- Training outputs: `./output/`
- Completed configs are renamed to `*_ok.yaml`
- Summary reports: `log/2025_05_15/RandomSearch_Datasets_high_quality/Batch_X/summary_report.txt`
```

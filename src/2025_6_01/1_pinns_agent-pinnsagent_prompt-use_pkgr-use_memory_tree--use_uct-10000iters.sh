#!/bin/bash
# parallel_run_experiments.sh
# Script to run PDE experiments in parallel across multiple GPUs

# Configuration
BASE_DIR="/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source"
CONDA_PATH="/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python"
CONDA_ENV="pinnsagent"
OUTPUT_DIR="./output/2025_6_01"
CSV_PATH="./data/dataset_for_retrieval.csv"
TRAIN_CODE_DIR="/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle"
CONDA_PYTHON="/gpfs/0607-cluster/miniconda3/envs/pinnsagent/bin/python"
EXPERIMENT_BASE_DIR="/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/src/2025_6_01/1_pinns_agent-pinnsagent_prompt-use_pkgr-use_memory_tree--use_uct-10000iters"
RUN_NAME_PREFIX="1_pinns_agent-pinnsagent_prompt-use_pkgr-use_memory_tree--use_uct-10000iters"
LOG_BASE_DIR="${EXPERIMENT_BASE_DIR}/logs"
LOG_SUMMARY_DIR="${EXPERIMENT_BASE_DIR}/logs_summary"

# Path to run_experiments.py (FIXED)
RUN_EXPERIMENTS_SCRIPT="/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/run_experiments.py"

# Create log directories
mkdir -p "${LOG_BASE_DIR}"
mkdir -p "${LOG_SUMMARY_DIR}"

# Verify that the script exists
if [ ! -f "${RUN_EXPERIMENTS_SCRIPT}" ]; then
    echo "❌ ERROR: run_experiments.py not found at ${RUN_EXPERIMENTS_SCRIPT}"
    exit 1
fi

echo "✅ Found run_experiments.py at ${RUN_EXPERIMENTS_SCRIPT}"

# Function to run a single PDE experiment
run_pde() {
    local pde_name=$1
    local pde_type=$2
    local gpu_id=$3
    local run_name=$4
    
    # Create log file name
    local log_file="${LOG_BASE_DIR}/${pde_type}_${pde_name}_gpu${gpu_id}.log"
    local summary_file="${LOG_SUMMARY_DIR}/${pde_type}_${pde_name}_summary.txt"
    
    echo "Starting ${pde_name} on GPU ${gpu_id}" | tee -a "${summary_file}"
    echo "Log file: ${log_file}" | tee -a "${summary_file}"
    echo "Start time: $(date)" | tee -a "${summary_file}"
    echo "----------------------------------------" | tee -a "${summary_file}"
    
    # Change to base directory and activate conda in a subshell
    (
        cd "${BASE_DIR}" || exit 1
        source "${CONDA_PATH}"
        conda activate "${CONDA_ENV}"
        
        # Run the experiment with absolute path to script
        python "${RUN_EXPERIMENTS_SCRIPT}" \
            --mode llm \
            --prompt_strategy pinns_agent \
            --use_pgkr \
            --pgkr_top_k 3 \
            --use_memory_tree \
            --use_uct \
            --pde_name "${pde_name}" \
            --num_iters 5 \
            --num_runs 3 \
            --device "${gpu_id}" \
            --iter 10000 \
            --output_dir "${OUTPUT_DIR}" \
            --run_name "${run_name}" \
            --csv_path "${CSV_PATH}" \
            --train_code_dir "${TRAIN_CODE_DIR}" \
            --conda_python "${CONDA_PYTHON}" \
            --simulate_new_pde \
            2>&1
    ) | tee "${log_file}"
    
    # Capture exit status
    local exit_status=${PIPESTATUS[0]}
    
    # Write summary
    echo "----------------------------------------" | tee -a "${summary_file}"
    echo "End time: $(date)" | tee -a "${summary_file}"
    echo "Exit status: ${exit_status}" | tee -a "${summary_file}"
    
    if [ ${exit_status} -eq 0 ]; then
        echo "✅ SUCCESS: ${pde_name} completed successfully" | tee -a "${summary_file}"
    else
        echo "❌ FAILED: ${pde_name} failed with exit code ${exit_status}" | tee -a "${summary_file}"
    fi
    
    echo "" | tee -a "${summary_file}"
}

# Export variables and function for parallel execution
export -f run_pde
export BASE_DIR CONDA_PATH CONDA_ENV OUTPUT_DIR CSV_PATH TRAIN_CODE_DIR CONDA_PYTHON 
export LOG_BASE_DIR LOG_SUMMARY_DIR RUN_EXPERIMENTS_SCRIPT

# Create master summary file
MASTER_SUMMARY="${LOG_SUMMARY_DIR}/master_summary.txt"
echo "==================================================" > "${MASTER_SUMMARY}"
echo "PINNs Agent Parallel Experiment Run" >> "${MASTER_SUMMARY}"
echo "Start time: $(date)" >> "${MASTER_SUMMARY}"
echo "==================================================" >> "${MASTER_SUMMARY}"
echo "" >> "${MASTER_SUMMARY}"
echo "Configuration:" >> "${MASTER_SUMMARY}"
echo "  Base directory: ${BASE_DIR}" >> "${MASTER_SUMMARY}"
echo "  Script location: ${RUN_EXPERIMENTS_SCRIPT}" >> "${MASTER_SUMMARY}"
echo "  Output directory: ${OUTPUT_DIR}" >> "${MASTER_SUMMARY}"
echo "  Log directory: ${LOG_BASE_DIR}" >> "${MASTER_SUMMARY}"
echo "  Summary directory: ${LOG_SUMMARY_DIR}" >> "${MASTER_SUMMARY}"
echo "" >> "${MASTER_SUMMARY}"

# 1D PDEs - GPUs 0, 1, 2 (3 PDEs, 3 GPUs)
echo "Starting 1D PDEs..." | tee -a "${MASTER_SUMMARY}"
run_pde "Burgers1D" "1d" 0 "${RUN_NAME_PREFIX}/1d/Burgers1D" &
run_pde "Wave1D" "1d" 1 "${RUN_NAME_PREFIX}/1d/Wave1D" &
run_pde "KuramotoSivashinskyEquation" "1d" 2 "${RUN_NAME_PREFIX}/1d/KuramotoSivashinskyEquation" &

# 2D PDEs - GPUs 0, 1, 2, 3, 4, 5, 6, 7 (8 PDEs, 8 GPUs)
echo "Starting 2D PDEs..." | tee -a "${MASTER_SUMMARY}"
run_pde "Burgers2D" "2d" 0 "${RUN_NAME_PREFIX}/2d/Burgers2D" &
run_pde "Wave2D_Heterogeneous" "2d" 1 "${RUN_NAME_PREFIX}/2d/Wave2D_Heterogeneous" &
run_pde "Heat2D_ComplexGeometry" "2d" 2 "${RUN_NAME_PREFIX}/2d/Heat2D_ComplexGeometry" &
run_pde "NS2D_LidDriven" "2d" 3 "${RUN_NAME_PREFIX}/2d/NS2D_LidDriven" &
run_pde "GrayScottEquation" "2d" 4 "${RUN_NAME_PREFIX}/2d/GrayScottEquation" &
run_pde "Heat2D_Multiscale" "2d" 5 "${RUN_NAME_PREFIX}/2d/Heat2D_Multiscale" &
run_pde "Heat2D_VaryingCoef" "2d" 6 "${RUN_NAME_PREFIX}/2d/Heat2D_VaryingCoef" &
run_pde "Poisson2D_ManyArea" "2d" 7 "${RUN_NAME_PREFIX}/2d/Poisson2D_ManyArea" &

# 3D PDEs - GPU 3 (1 PDE)
echo "Starting 3D PDEs..." | tee -a "${MASTER_SUMMARY}"
run_pde "Poisson3D_ComplexGeometry" "3d" 3 "${RUN_NAME_PREFIX}/3d/Poisson3D_ComplexGeometry" &

# ND PDEs - GPUs 4, 5 (2 PDEs, 2 GPUs)
echo "Starting ND PDEs..." | tee -a "${MASTER_SUMMARY}"
run_pde "PoissonND" "nd" 4 "${RUN_NAME_PREFIX}/nd/PoissonND" &
run_pde "HeatND" "nd" 5 "${RUN_NAME_PREFIX}/nd/HeatND" &

# Wait for all background jobs to complete
echo "" | tee -a "${MASTER_SUMMARY}"
echo "Waiting for all experiments to complete..." | tee -a "${MASTER_SUMMARY}"
echo "Total background jobs: $(jobs -p | wc -l)" | tee -a "${MASTER_SUMMARY}"
wait

# Generate final summary
echo "" >> "${MASTER_SUMMARY}"
echo "==================================================" >> "${MASTER_SUMMARY}"
echo "All experiments completed!" >> "${MASTER_SUMMARY}"
echo "End time: $(date)" >> "${MASTER_SUMMARY}"
echo "==================================================" >> "${MASTER_SUMMARY}"
echo "" >> "${MASTER_SUMMARY}"

# Count successes and failures
echo "Summary of results:" >> "${MASTER_SUMMARY}"
echo "-------------------" >> "${MASTER_SUMMARY}"
if [ -d "${LOG_SUMMARY_DIR}" ] && [ "$(ls -A ${LOG_SUMMARY_DIR}/*_summary.txt 2>/dev/null)" ]; then
    grep -h "SUCCESS\|FAILED" "${LOG_SUMMARY_DIR}"/*_summary.txt 2>/dev/null | sort >> "${MASTER_SUMMARY}"
else
    echo "No summary files found" >> "${MASTER_SUMMARY}"
fi

# Calculate statistics
total_experiments=$(ls "${LOG_SUMMARY_DIR}"/*_summary.txt 2>/dev/null | wc -l)
successful=$(grep -h "SUCCESS" "${LOG_SUMMARY_DIR}"/*_summary.txt 2>/dev/null | wc -l)
failed=$(grep -h "FAILED" "${LOG_SUMMARY_DIR}"/*_summary.txt 2>/dev/null | wc -l)

echo "" >> "${MASTER_SUMMARY}"
echo "Statistics:" >> "${MASTER_SUMMARY}"
echo "  Total experiments: ${total_experiments}" >> "${MASTER_SUMMARY}"
echo "  Successful: ${successful}" >> "${MASTER_SUMMARY}"
echo "  Failed: ${failed}" >> "${MASTER_SUMMARY}"

# Display final summary
echo ""
echo "========================================="
echo "Experiment run completed!"
echo "========================================="
cat "${MASTER_SUMMARY}"

echo ""
echo "All logs saved to: ${LOG_BASE_DIR}"
echo "Summary files saved to: ${LOG_SUMMARY_DIR}"
echo "Master summary: ${MASTER_SUMMARY}"
import os
import glob
from datetime import datetime

def generate_training_scripts():
    # 基础路径
    config_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/config/ICML_2025/RandomSearch_Table1/2025-05-25"
    script_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/scripts/2025_5_25/RandomSearch_Table1/2_run_experiments_per_batchs"
    work_dir = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle"
    log_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/log/RandomSearch_Table1"
    
    # 创建脚本目录和日志目录
    os.makedirs(script_base, exist_ok=True)
    os.makedirs(log_base, exist_ok=True)
    
    # 为每个batch生成脚本
    for batch_num in range(1, 11):  # Batch_1 到 Batch_10
        batch_dir = f"{script_base}/Batch_{batch_num}"
        batch_log_dir = f"{log_base}/Batch_{batch_num}"
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(batch_log_dir, exist_ok=True)
        
        print(f"生成 Batch_{batch_num} 的训练脚本...")
        
        # 1. 生成 1d 训练脚本
        generate_1d_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir)
        
        # 2. 生成 2d 训练脚本
        generate_2d_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir)
        
        # 3. 生成 3d 训练脚本
        generate_3d_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir)
        
        # 4. 生成 nd 训练脚本
        generate_nd_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir)
        
        # 5. 生成批量运行脚本
        generate_batch_runner(batch_num, batch_dir, batch_log_dir)

def get_conda_init():
    """返回conda初始化代码"""
    return '''
# 初始化conda环境
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]; then
    source /usr/local/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    # 如果conda命令存在，尝试初始化
    eval "$(conda shell.bash hook)" 2>/dev/null || true
else
    echo "警告: 未找到conda安装，尝试使用系统Python环境"
fi
'''

def get_log_functions():
    """返回通用的日志记录函数"""
    return '''
# 日志记录函数
log_start() {
    local yaml_file="$1"
    local gpu_id="$2"
    local log_file="$3"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] START | GPU:$gpu_id | $yaml_file" >> "$log_file"
}

log_success() {
    local yaml_file="$1"
    local gpu_id="$2"
    local log_file="$3"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] SUCCESS | GPU:$gpu_id | $yaml_file" >> "$log_file"
}

log_failure() {
    local yaml_file="$1"
    local gpu_id="$2" 
    local log_file="$3"
    local error_msg="$4"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] FAILED | GPU:$gpu_id | $yaml_file | Error: $error_msg" >> "$log_file"
}

log_skip() {
    local yaml_file="$1"
    local log_file="$2"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] SKIPPED | Already completed: $yaml_file" >> "$log_file"
}
'''

def generate_1d_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir):
    conda_init = get_conda_init()
    log_functions = get_log_functions()
    script_content = f"""#!/bin/bash

# 1D PDE 训练脚本 - Batch_{batch_num}

# 确保日志目录存在
mkdir -p "{batch_log_dir}"

cd {work_dir}

{conda_init}

# 尝试激活conda环境
if command -v conda >/dev/null 2>&1; then
    conda activate pinnsagent 2>/dev/null || {{
        echo "警告: 无法激活pinnacle环境，使用当前环境"
    }}
else
    echo "警告: conda不可用，使用当前Python环境"
fi

# 检查并安装依赖包
python -c "import dill" 2>/dev/null || {{
    echo "安装缺失的依赖包..."
    pip install dill
}}

# 设置GPU和日志文件（使用绝对路径）
export CUDA_VISIBLE_DEVICES=0
LOG_FILE="{batch_log_dir}/1d.log"

# 确保日志文件存在
touch "$LOG_FILE"

{log_functions}

echo "开始训练 Batch_{batch_num} 的 1D PDEs..."
echo "日志保存到: $LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始 Batch_{batch_num} 1D 训练 ==========" >> "$LOG_FILE"

# 获取所有1D PDE的yaml文件
CONFIG_DIR="{config_base}/Batch_{batch_num}/1d"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误: 配置目录不存在: $CONFIG_DIR" >> "$LOG_FILE"
    exit 1
fi

# 收集所有需要训练的yaml文件
echo "收集配置文件..."
yaml_files=()
for pde_dir in "$CONFIG_DIR"/*; do
    if [ -d "$pde_dir" ]; then
        for yaml_file in "$pde_dir"/train_*.yaml; do
            if [ -f "$yaml_file" ] && [[ "$yaml_file" != *"_ok.yaml" ]]; then
                yaml_files+=("$yaml_file")
            fi
        done
    fi
done

total_configs=${{#yaml_files[@]}}
completed_configs=0
failed_configs=0

echo "找到 $total_configs 个待训练的配置文件"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 找到 $total_configs 个待训练的配置文件" >> "$LOG_FILE"

# 如果没有待训练文件，统计已完成的并退出
if [ $total_configs -eq 0 ]; then
    echo "没有找到需要训练的配置文件，统计已完成的..."
    completed_configs=$(find "$CONFIG_DIR" -name "*_ok.yaml" | wc -l)
    echo "已完成配置: $completed_configs"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 没有待训练配置，已完成: $completed_configs" >> "$LOG_FILE"
    exit 0
fi

# 遍历所有配置文件进行训练
for ((i=0; i<${{#yaml_files[@]}}; i++)); do
    yaml_file="${{yaml_files[$i]}}"
    current_progress=$((i+1))
    
    pde_name=$(basename $(dirname "$yaml_file"))
    config_name=$(basename "$yaml_file")
    
    echo "进度: $current_progress/$total_configs - 训练 $pde_name/$config_name"
    log_start "$yaml_file" "0" "$LOG_FILE"
    
    # 运行训练
    if python benchmark.py --name "2025_5_25_1d_batch{batch_num}" --yaml_path "$yaml_file" 2>&1; then
        # 训练成功，重命名文件
        new_name="${{yaml_file%.yaml}}_ok.yaml"
        mv "$yaml_file" "$new_name"
        echo "训练完成，已重命名: $new_name"
        log_success "$new_name" "0" "$LOG_FILE"
        ((completed_configs++))
    else
        echo "训练失败: $yaml_file"
        log_failure "$yaml_file" "0" "$LOG_FILE" "Training failed"
        ((failed_configs++))
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进度: $current_progress/$total_configs" >> "$LOG_FILE"
done

# 记录统计信息
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== Batch_{batch_num} 1D 训练完成 ==========" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs" >> "$LOG_FILE"
echo "Batch_{batch_num} 1D PDEs 训练完成！总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs"
"""
    
    with open(f"{batch_dir}/1_train_1d.sh", "w") as f:
        f.write(script_content)
    os.chmod(f"{batch_dir}/1_train_1d.sh", 0o755)

def generate_2d_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir):
    conda_init = get_conda_init()
    log_functions = get_log_functions()
    script_content = f"""#!/bin/bash

# 2D PDE 训练脚本 - Batch_{batch_num} (8个GPU轮流训练)

# 确保日志目录存在
mkdir -p "{batch_log_dir}"

cd {work_dir}

{conda_init}

# 尝试激活conda环境
if command -v conda >/dev/null 2>&1; then
    conda activate pinnsagent 2>/dev/null || {{
        echo "警告: 无法激活pinnacle环境，使用当前环境"
    }}
else
    echo "警告: conda不可用，使用当前Python环境"
fi

# 检查并安装依赖包
python -c "import dill" 2>/dev/null || {{
    echo "安装缺失的依赖包..."
    pip install dill
}}

# 日志文件（使用绝对路径）
LOG_FILE="{batch_log_dir}/2d.log"

# 确保日志文件存在
touch "$LOG_FILE"

{log_functions}

echo "开始训练 Batch_{batch_num} 的 2D PDEs (8个GPU轮流模式)..."
echo "日志保存到: $LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始 Batch_{batch_num} 2D 训练 ==========" >> "$LOG_FILE"

# 获取所有2D PDE的yaml文件
CONFIG_DIR="{config_base}/Batch_{batch_num}/2d"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误: 配置目录不存在: $CONFIG_DIR" >> "$LOG_FILE"
    exit 1
fi

# 收集所有需要训练的yaml文件，按PDE轮流排序
echo "收集配置文件并按PDE轮流排序..."
yaml_files=()

# 首先获取所有PDE目录
pde_dirs=()
for pde_dir in "$CONFIG_DIR"/*; do
    if [ -d "$pde_dir" ]; then
        pde_dirs+=("$pde_dir")
    fi
done

echo "找到 ${{#pde_dirs[@]}} 个PDE类型: ${{pde_dirs[@]}}"

# 按轮流方式收集配置文件：每个PDE轮流取一个配置
max_configs_per_pde=0
declare -A pde_files

# 先收集每个PDE的所有配置文件
for pde_dir in "${{pde_dirs[@]}}"; do
    pde_name=$(basename "$pde_dir")
    pde_configs=()
    
    while IFS= read -r -d '' yaml_file; do
        if [[ "$yaml_file" != *"_ok.yaml" ]]; then
            pde_configs+=("$yaml_file")
        fi
    done < <(find "$pde_dir" -name "train_*.yaml" -print0 | sort -z)
    
    pde_files["$pde_name"]="${{pde_configs[*]}}"
    
    if [ ${{#pde_configs[@]}} -gt $max_configs_per_pde ]; then
        max_configs_per_pde=${{#pde_configs[@]}}
    fi
    
    echo "  $pde_name: ${{#pde_configs[@]}} 个配置"
done

echo "最大配置数: $max_configs_per_pde"

# 按轮流方式组织文件：第1轮每个PDE的第1个配置，第2轮每个PDE的第2个配置...
for ((round=1; round<=max_configs_per_pde; round++)); do
    echo "处理第 $round 轮配置..."
    for pde_dir in "${{pde_dirs[@]}}"; do
        pde_name=$(basename "$pde_dir")
        
        # 将该PDE的配置文件字符串转换为数组
        IFS=' ' read -ra configs <<< "${{pde_files[$pde_name]}}"
        
        # 获取第round个配置（从0开始索引）
        config_index=$((round-1))
        if [ $config_index -lt ${{#configs[@]}} ]; then
            yaml_files+=("${{configs[$config_index]}}")
            echo "  添加: ${{configs[$config_index]}}"
        fi
    done
done

echo "最终收集到 ${{#yaml_files[@]}} 个配置文件，按PDE轮流排序"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 找到 ${{#yaml_files[@]}} 个待训练的配置文件（按PDE轮流排序）" >> "$LOG_FILE"

# 显示前10个文件的排序情况以验证
echo "前10个配置文件排序："
for ((i=0; i<10 && i<${{#yaml_files[@]}}; i++)); do
    echo "  $((i+1)). $(basename $(dirname "${{yaml_files[$i]}}")))/$(basename "${{yaml_files[$i]}}")"
done

# 如果没有待训练文件，直接退出
if [ ${{#yaml_files[@]}} -eq 0 ]; then
    echo "没有找到需要训练的配置文件，退出"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 没有找到需要训练的配置文件" >> "$LOG_FILE"
    exit 0
fi

# 单个训练函数
train_config() {{
    local yaml_file="$1"
    local gpu_id="$2"
    local log_file="$3"
    
    echo "GPU $gpu_id 开始训练: $yaml_file"
    log_start "$yaml_file" "$gpu_id" "$log_file"
    
    if CUDA_VISIBLE_DEVICES=$gpu_id python benchmark.py --name "2025_5_25_2d_batch{batch_num}_gpu$gpu_id" --yaml_path "$yaml_file" 2>&1; then
        # 训练成功，重命名文件
        new_name="${{yaml_file%.yaml}}_ok.yaml"
        mv "$yaml_file" "$new_name"
        echo "GPU $gpu_id 训练完成: $new_name"
        log_success "$new_name" "$gpu_id" "$log_file"
        return 0
    else
        echo "GPU $gpu_id 训练失败: $yaml_file"
        log_failure "$yaml_file" "$gpu_id" "$log_file" "Training failed"
        return 1
    fi
}}

# 使用8个GPU (0-7) 轮流训练
available_gpus=(0 1 2 3 4 5 6 7)
declare -A gpu_status  # 关联数组记录GPU状态: 0=空闲, PID=忙碌
declare -A gpu_jobs    # 关联数组记录GPU对应的任务文件

# 初始化GPU状态
for gpu in "${{available_gpus[@]}}"; do
    gpu_status[$gpu]=0
    gpu_jobs[$gpu]=""
done

# 启动训练任务的函数
start_job() {{
    local yaml_file="$1"
    local gpu_id="$2"
    
    train_config "$yaml_file" "$gpu_id" "$LOG_FILE" &
    local pid=$!
    gpu_status[$gpu_id]=$pid
    gpu_jobs[$gpu_id]="$yaml_file"
    echo "在GPU $gpu_id 上启动任务 PID=$pid: $(basename $(dirname "$yaml_file"))/$(basename "$yaml_file")"
}}

# 检查并清理完成的任务（改进版）
check_and_clean_jobs() {{
    for gpu in "${{available_gpus[@]}}"; do
        local pid=${{gpu_status[$gpu]}}
        if [[ $pid -ne 0 ]]; then
            # 多重检查进程状态
            if ! kill -0 "$pid" 2>/dev/null; then
                # 进程已结束
                wait "$pid" 2>/dev/null
                echo "GPU $gpu 上的任务 PID=$pid 已完成: $(basename $(dirname "${{gpu_jobs[$gpu]}}"))/$(basename "${{gpu_jobs[$gpu]}}")"
                gpu_status[$gpu]=0
                gpu_jobs[$gpu]=""
                
                # 额外清理GPU显存
                CUDA_VISIBLE_DEVICES=$gpu python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            else
                # 进程仍在运行，但检查是否卡住超时
                local job_file="${{gpu_jobs[$gpu]}}"
                if [ -n "$job_file" ]; then
                    # 检查对应的_ok文件是否已生成（说明训练完成但进程未退出）
                    local ok_file="${{job_file%.yaml}}_ok.yaml"
                    if [ -f "$ok_file" ]; then
                        echo "警告: GPU $gpu 训练已完成但进程未退出，强制清理 PID=$pid"
                        kill -TERM "$pid" 2>/dev/null || true
                        sleep 2
                        kill -KILL "$pid" 2>/dev/null || true
                        wait "$pid" 2>/dev/null || true
                        gpu_status[$gpu]=0
                        gpu_jobs[$gpu]=""
                        
                        # 清理GPU显存
                        CUDA_VISIBLE_DEVICES=$gpu python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                    fi
                fi
            fi
        fi
    done
}}

# 获取一个空闲的GPU
get_free_gpu() {{
    for gpu in "${{available_gpus[@]}}"; do
        if [[ ${{gpu_status[$gpu]}} -eq 0 ]]; then
            echo "$gpu"
            return 0
        fi
    done
    return 1
}}

# 获取当前运行中的任务数量
get_running_jobs_count() {{
    local count=0
    for gpu in "${{available_gpus[@]}}"; do
        if [[ ${{gpu_status[$gpu]}} -ne 0 ]]; then
            ((count++))
        fi
    done
    echo $count
}}

# 主训练循环
yaml_index=0
total_yamls=${{#yaml_files[@]}}

echo "开始主训练循环，总共 $total_yamls 个配置文件"

while [[ $yaml_index -lt $total_yamls ]]; do
    # 清理已完成的任务
    check_and_clean_jobs
    
    # 尝试启动新任务
    if free_gpu=$(get_free_gpu); then
        yaml_file="${{yaml_files[$yaml_index]}}"
        start_job "$yaml_file" "$free_gpu"
        ((yaml_index++))
        echo "进度: $yaml_index/$total_yamls (已启动: $(basename $(dirname "$yaml_file"))/$(basename "$yaml_file"))"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进度: $yaml_index/$total_yamls" >> "$LOG_FILE"
    else
        # 没有空闲GPU，等待一会儿
        echo "所有GPU忙碌中，等待..."
        sleep 10
    fi
    
    # 防止无限循环的安全检查
    running_count=$(get_running_jobs_count)
    if [[ $running_count -eq 0 ]] && [[ $yaml_index -lt $total_yamls ]]; then
        echo "警告: 没有运行中的任务但仍有待处理文件，强制重试..."
        sleep 5
    fi
done

# 等待所有剩余任务完成
echo "等待所有剩余任务完成..."
while true; do
    check_and_clean_jobs
    running_count=$(get_running_jobs_count)
    if [[ $running_count -eq 0 ]]; then
        break
    fi
    echo "还有 $running_count 个任务在运行中..."
    sleep 10
done

# 重新统计最终结果（重新扫描目录）
total_configs=0
completed_configs=0

# 统计总配置数（包括已完成的）
for pde_dir in "$CONFIG_DIR"/*; do
    if [ -d "$pde_dir" ]; then
        config_count=$(find "$pde_dir" -name "train_*.yaml" | wc -l)
        total_configs=$((total_configs + config_count))
    fi
done

# 统计已完成的配置
completed_configs=$(find "$CONFIG_DIR" -name "*_ok.yaml" | wc -l)
failed_configs=$((total_configs - completed_configs))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== Batch_{batch_num} 2D 训练完成 ==========" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs" >> "$LOG_FILE"
echo "Batch_{batch_num} 2D PDEs 训练完成！总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs"
"""
    
    with open(f"{batch_dir}/2_train_2d.sh", "w") as f:
        f.write(script_content)
    os.chmod(f"{batch_dir}/2_train_2d.sh", 0o755)

def generate_3d_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir):
    conda_init = get_conda_init()
    log_functions = get_log_functions()
    script_content = f"""#!/bin/bash

# 3D PDE 训练脚本 - Batch_{batch_num}

# 确保日志目录存在
mkdir -p "{batch_log_dir}"

cd {work_dir}

{conda_init}

# 尝试激活conda环境
if command -v conda >/dev/null 2>&1; then
    conda activate pinnsagent 2>/dev/null || {{
        echo "警告: 无法激活pinnacle环境，使用当前环境"
    }}
else
    echo "警告: conda不可用，使用当前Python环境"
fi

# 检查并安装依赖包
python -c "import dill" 2>/dev/null || {{
    echo "安装缺失的依赖包..."
    pip install dill
}}

# 设置GPU和日志文件（使用绝对路径）
export CUDA_VISIBLE_DEVICES=1
LOG_FILE="{batch_log_dir}/3d.log"

# 确保日志文件存在
touch "$LOG_FILE"

{log_functions}

echo "开始训练 Batch_{batch_num} 的 3D PDEs..."
echo "日志保存到: $LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始 Batch_{batch_num} 3D 训练 ==========" >> "$LOG_FILE"

# 获取所有3D PDE的yaml文件
CONFIG_DIR="{config_base}/Batch_{batch_num}/3d"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误: 配置目录不存在: $CONFIG_DIR" >> "$LOG_FILE"
    exit 1
fi

# 收集所有需要训练的yaml文件
echo "收集配置文件..."
yaml_files=()
for pde_dir in "$CONFIG_DIR"/*; do
    if [ -d "$pde_dir" ]; then
        for yaml_file in "$pde_dir"/train_*.yaml; do
            if [ -f "$yaml_file" ] && [[ "$yaml_file" != *"_ok.yaml" ]]; then
                yaml_files+=("$yaml_file")
            fi
        done
    fi
done

total_configs=${{#yaml_files[@]}}
completed_configs=0
failed_configs=0

echo "找到 $total_configs 个待训练的配置文件"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 找到 $total_configs 个待训练的配置文件" >> "$LOG_FILE"

# 如果没有待训练文件，统计已完成的并退出
if [ $total_configs -eq 0 ]; then
    echo "没有找到需要训练的配置文件，统计已完成的..."
    completed_configs=$(find "$CONFIG_DIR" -name "*_ok.yaml" | wc -l)
    echo "已完成配置: $completed_configs"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 没有待训练配置，已完成: $completed_configs" >> "$LOG_FILE"
    exit 0
fi

# 遍历所有配置文件进行训练
for ((i=0; i<${{#yaml_files[@]}}; i++)); do
    yaml_file="${{yaml_files[$i]}}"
    current_progress=$((i+1))
    
    pde_name=$(basename $(dirname "$yaml_file"))
    config_name=$(basename "$yaml_file")
    
    echo "进度: $current_progress/$total_configs - 训练 $pde_name/$config_name"
    log_start "$yaml_file" "1" "$LOG_FILE"
    
    # 运行训练
    if python benchmark.py --name "2025_5_25_3d_batch{batch_num}" --yaml_path "$yaml_file" 2>&1; then
        # 训练成功，重命名文件
        new_name="${{yaml_file%.yaml}}_ok.yaml"
        mv "$yaml_file" "$new_name"
        echo "训练完成，已重命名: $new_name"
        log_success "$new_name" "1" "$LOG_FILE"
        ((completed_configs++))
    else
        echo "训练失败: $yaml_file"
        log_failure "$yaml_file" "1" "$LOG_FILE" "Training failed"
        ((failed_configs++))
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进度: $current_progress/$total_configs" >> "$LOG_FILE"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== Batch_{batch_num} 3D 训练完成 ==========" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs" >> "$LOG_FILE"
echo "Batch_{batch_num} 3D PDEs 训练完成！总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs"
"""
    
    with open(f"{batch_dir}/3_train_3d.sh", "w") as f:
        f.write(script_content)
    os.chmod(f"{batch_dir}/3_train_3d.sh", 0o755)

def generate_nd_script(batch_num, config_base, batch_dir, work_dir, batch_log_dir):
    conda_init = get_conda_init()
    log_functions = get_log_functions()
    script_content = f"""#!/bin/bash

# ND PDE 训练脚本 - Batch_{batch_num}

# 确保日志目录存在
mkdir -p "{batch_log_dir}"

cd {work_dir}

{conda_init}

# 尝试激活conda环境
if command -v conda >/dev/null 2>&1; then
    conda activate pinnsagent 2>/dev/null || {{
        echo "警告: 无法激活pinnacle环境，使用当前环境"
    }}
else
    echo "警告: conda不可用，使用当前Python环境"
fi

# 检查并安装依赖包
python -c "import dill" 2>/dev/null || {{
    echo "安装缺失的依赖包..."
    pip install dill
}}

# 设置GPU和日志文件（使用绝对路径）
export CUDA_VISIBLE_DEVICES=2
LOG_FILE="{batch_log_dir}/nd.log"

# 确保日志文件存在
touch "$LOG_FILE"

{log_functions}

echo "开始训练 Batch_{batch_num} 的 ND PDEs..."
echo "日志保存到: $LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始 Batch_{batch_num} ND 训练 ==========" >> "$LOG_FILE"

# 获取所有ND PDE的yaml文件
CONFIG_DIR="{config_base}/Batch_{batch_num}/nd"

# 检查配置目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "错误: 配置目录不存在: $CONFIG_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误: 配置目录不存在: $CONFIG_DIR" >> "$LOG_FILE"
    exit 1
fi

# 统计信息
total_configs=0
completed_configs=0
failed_configs=0

# 遍历所有ND PDE类型
for pde_dir in "$CONFIG_DIR"/*; do
    if [ -d "$pde_dir" ]; then
        pde_name=$(basename "$pde_dir" | sed 's/-[0-9]*$//')
        echo "开始训练 $pde_name..."
        
        # 遍历该PDE的所有配置文件
        for yaml_file in "$pde_dir"/train_*.yaml; do
            if [ -f "$yaml_file" ]; then
                ((total_configs++))
                
                # 检查是否已经完成
                if [[ "$yaml_file" == *"_ok.yaml" ]]; then
                    echo "跳过已完成的配置: $yaml_file"
                    log_skip "$yaml_file" "$LOG_FILE"
                    ((completed_configs++))
                    continue
                fi
                
                echo "训练配置: $yaml_file"
                log_start "$yaml_file" "2" "$LOG_FILE"
                
                # 运行训练
                if python benchmark.py --name "2025_5_25_nd_batch{batch_num}" --yaml_path "$yaml_file" 2>&1; then
                    # 训练成功，重命名文件
                    new_name="${{yaml_file%.yaml}}_ok.yaml"
                    mv "$yaml_file" "$new_name"
                    echo "训练完成，已重命名: $new_name"
                    log_success "$new_name" "2" "$LOG_FILE"
                    ((completed_configs++))
                else
                    echo "训练失败: $yaml_file"
                    log_failure "$yaml_file" "2" "$LOG_FILE" "Training failed"
                    ((failed_configs++))
                fi
            fi
        done
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== Batch_{batch_num} ND 训练完成 ==========" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs" >> "$LOG_FILE"
echo "Batch_{batch_num} ND PDEs 训练完成！总配置: $total_configs | 完成: $completed_configs | 失败: $failed_configs"
"""
    
    with open(f"{batch_dir}/4_train_nd.sh", "w") as f:
        f.write(script_content)
    os.chmod(f"{batch_dir}/4_train_nd.sh", 0o755)

def generate_batch_runner(batch_num, batch_dir, batch_log_dir):
    script_content = f"""#!/bin/bash

# Batch_{batch_num} 总控制脚本

# 确保日志目录存在
mkdir -p "{batch_log_dir}"

LOG_FILE="{batch_log_dir}/summary.log"

# 确保日志文件存在
touch "$LOG_FILE"

echo "开始运行 Batch_{batch_num} 的所有训练任务..."
echo "日志保存到: {batch_log_dir}/"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== 开始 Batch_{batch_num} 总体训练 ==========" >> "$LOG_FILE"

# 并行运行不同维度的训练（使用不同GPU）
echo "启动并行训练..."

# 1D, 3D, ND 可以同时运行（使用GPU 0,1,2）
./1_train_1d.sh &
PID_1D=$!

./3_train_3d.sh &
PID_3D=$!

./4_train_nd.sh &
PID_ND=$!

echo "1D, 3D, ND 训练已启动..."

# 同时启动2D训练（使用GPU 0-7）
echo "同时启动2D训练（使用GPU 0-7）..."
./2_train_2d.sh &
PID_2D=$!

# 等待所有任务完成
echo "等待所有训练完成..."

wait $PID_1D
echo "1D 训练完成"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 1D 训练完成" >> "$LOG_FILE"

wait $PID_3D  
echo "3D 训练完成"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 3D 训练完成" >> "$LOG_FILE"

wait $PID_ND
echo "ND 训练完成"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ND 训练完成" >> "$LOG_FILE"

wait $PID_2D
echo "2D 训练完成"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 2D 训练完成" >> "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ========== Batch_{batch_num} 所有训练完成 ==========" >> "$LOG_FILE"
echo "Batch_{batch_num} 所有训练任务完成！"

# 生成汇总报告
echo "生成训练汇总报告..."
cd "$(dirname "$(readlink -f "$0")")/.."
python generate_summary.py {batch_num}
"""
    
    with open(f"{batch_dir}/run_batch_{batch_num}.sh", "w") as f:
        f.write(script_content)
    os.chmod(f"{batch_dir}/run_batch_{batch_num}.sh", 0o755)

def generate_summary_script():
    """生成汇总统计脚本"""
    script_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/scripts/2025_5_25/RandomSearch_Datasets_high_quality/2_run_experiments_per_batchs"
    log_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/log/RandomSearch_Datasets_high_quality"
    
    summary_content = f'''#!/usr/bin/env python3
import sys
import os
import glob
from datetime import datetime

def generate_summary(batch_num):
    """生成指定batch的训练汇总报告"""
    log_dir = "{log_base}/Batch_{{batch_num}}"
    
    print(f"\\n========== Batch_{{batch_num}} 训练汇总报告 ==========")
    print(f"生成时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"日志目录: {{log_dir}}")
    
    dimensions = ['1d', '2d', '3d', 'nd']
    total_all = 0
    success_all = 0
    failed_all = 0
    
    for dim in dimensions:
        log_file = f"{{log_dir}}/{{dim}}.log"
        if os.path.exists(log_file):
            total, success, failed = parse_log_file(log_file)
            print(f"\\n{{dim.upper()}} PDEs:")
            print(f"  总配置: {{total}}")
            print(f"  成功: {{success}}")
            print(f"  失败: {{failed}}")
            print(f"  成功率: {{success/total*100:.1f}}%" if total > 0 else "  成功率: N/A")
            
            total_all += total
            success_all += success
            failed_all += failed
        else:
            print(f"\\n{{dim.upper()}} PDEs: 日志文件不存在 ({{log_file}})")
    
    print(f"\\n总体统计:")
    print(f"  总配置: {{total_all}}")
    print(f"  成功: {{success_all}}")
    print(f"  失败: {{failed_all}}")
    print(f"  成功率: {{success_all/total_all*100:.1f}}%" if total_all > 0 else "  成功率: N/A")
    
    # 确保目录存在并保存到日志目录
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{{log_dir}}/summary_report.txt", "w") as f:
        f.write(f"Batch_{{batch_num}} 训练汇总报告\\n")
        f.write(f"生成时间: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n\\n")
        f.write(f"总体统计: 总配置={{total_all}}, 成功={{success_all}}, 失败={{failed_all}}, 成功率={{success_all/total_all*100:.1f}}%\\n")

def parse_log_file(log_file):
    """解析日志文件，统计训练结果"""
    total = 0
    success = 0
    failed = 0
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if '| SUCCESS |' in line:
                    success += 1
                elif '| FAILED |' in line:
                    failed += 1
                elif '| START |' in line and '| SKIPPED |' not in line:
                    total += 1
    except Exception as e:
        print(f"读取日志文件失败: {{e}}")
    
    return total, success, failed

if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_num = sys.argv[1]
        generate_summary(batch_num)
    else:
        print("使用方法: python generate_summary.py <batch_number>")
'''
    
    with open(f"{script_base}/generate_summary.py", "w") as f:
        f.write(summary_content)
    os.chmod(f"{script_base}/generate_summary.py", 0o755)

def generate_master_script():
    script_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/scripts/2025_5_25/RandomSearch_Datasets_high_quality/2_run_experiments_per_batchs"
    log_base = "/gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/log/RandomSearch_Datasets_high_quality"
    
    master_content = f"""#!/bin/bash

# 主控制脚本 - 支持范围和并行运行
echo "PINNs Random Search 训练系统 (增强版)"
echo "使用方法:"
echo "  ./run_all_batches.sh                    # 运行所有batch (串行)"
echo "  ./run_all_batches.sh 3                  # 运行单个batch 3"
echo "  ./run_all_batches.sh 1-5                # 运行batch 1到5 (串行)"
echo "  ./run_all_batches.sh 1-5 --parallel     # 运行batch 1到5 (并行)"
echo "  ./run_all_batches.sh 6-10 --parallel    # 运行batch 6到10 (并行)"
echo "  ./run_all_batches.sh --parallel         # 运行所有batch (并行)"
echo ""
echo "日志保存位置: {log_base}"

# 创建主日志目录
mkdir -p "{log_base}"

# 解析参数
PARALLEL=false
BATCH_RANGE=""

for arg in "$@"; do
    case $arg in
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            if [ -z "$BATCH_RANGE" ]; then
                BATCH_RANGE="$arg"
            fi
            shift
            ;;
    esac
done

# 解析batch范围
parse_range() {{
    local range="$1"
    if [ -z "$range" ]; then
        # 默认运行所有batch
        echo "1 2 3 4 5 6 7 8 9 10"
    elif [[ "$range" =~ ^[0-9]+$ ]]; then
        # 单个数字
        if [ "$range" -ge 1 ] && [ "$range" -le 10 ]; then
            echo "$range"
        else
            echo "错误: batch数量必须在1-10之间" >&2
            exit 1
        fi
    elif [[ "$range" =~ ^[0-9]+-[0-9]+$ ]]; then
        # 范围格式 (如 1-5)
        local start=$(echo "$range" | cut -d'-' -f1)
        local end=$(echo "$range" | cut -d'-' -f2)
        
        if [ "$start" -ge 1 ] && [ "$end" -le 10 ] && [ "$start" -le "$end" ]; then
            seq "$start" "$end"
        else
            echo "错误: 无效的batch范围 $range，范围必须在1-10之间且start<=end" >&2
            exit 1
        fi
    else
        echo "错误: 无效的batch格式 $range" >&2
        echo "支持格式: 单个数字(如 3) 或范围(如 1-5)" >&2
        exit 1
    fi
}}

# 运行单个batch的函数
run_single_batch() {{
    local batch_num="$1"
    local mode="$2"  # "serial" 或 "parallel"
    
    echo "[$mode] 开始运行 Batch_$batch_num"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$mode] 开始 Batch_$batch_num" >> "{log_base}/master.log"
    
    if [ -d "Batch_$batch_num" ]; then
        cd "Batch_$batch_num"
        if [ "$mode" = "parallel" ]; then
            # 并行模式：后台运行
            (
                ./run_batch_$batch_num.sh
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] [parallel] 完成 Batch_$batch_num" >> "{log_base}/master.log"
                echo "[parallel] Batch_$batch_num 完成"
            ) &
            local pid=$!
            echo "Batch_$batch_num 已在后台启动 (PID: $pid)"
            echo "$pid:$batch_num"  # 返回PID和batch号
        else
            # 串行模式：前台运行
            ./run_batch_$batch_num.sh
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [serial] 完成 Batch_$batch_num" >> "{log_base}/master.log"
            echo "[serial] Batch_$batch_num 完成"
        fi
        cd ..
    else
        echo "错误: 目录 Batch_$batch_num 不存在"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误: 目录 Batch_$batch_num 不存在" >> "{log_base}/master.log"
    fi
}}

# 获取要运行的batch列表
batches=($(parse_range "$BATCH_RANGE"))

if [ "$PARALLEL" = true ]; then
    echo "=========================================="
    echo "并行模式: 同时运行 ${{batches[*]}}"
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 并行模式开始，batch: ${{batches[*]}}" >> "{log_base}/master.log"
    
    # 存储后台进程信息
    background_jobs=()
    
    # 启动所有batch
    for batch in "${{batches[@]}}"; do
        job_info=$(run_single_batch "$batch" "parallel")
        if [[ "$job_info" =~ ^[0-9]+:[0-9]+$ ]]; then
            background_jobs+=("$job_info")
        fi
    done
    
    echo ""
    echo "所有batch已启动，等待完成..."
    echo "后台任务: ${{#background_jobs[@]}} 个"
    
    # 等待所有后台任务完成
    for job in "${{background_jobs[@]}}"; do
        local pid=$(echo "$job" | cut -d':' -f1)
        local batch_num=$(echo "$job" | cut -d':' -f2)
        echo "等待 Batch_$batch_num (PID: $pid) 完成..."
        wait "$pid"
        echo "✓ Batch_$batch_num 已完成"
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 并行模式所有batch完成" >> "{log_base}/master.log"
    echo ""
    echo "🎉 所有batch并行执行完成！"
    
else
    echo "=========================================="
    echo "串行模式: 依次运行 ${{batches[*]}}"
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 串行模式开始，batch: ${{batches[*]}}" >> "{log_base}/master.log"
    
    # 依次运行每个batch
    for batch in "${{batches[@]}}"; do
        echo "========================================"
        run_single_batch "$batch" "serial"
        echo "========================================"
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 串行模式所有batch完成" >> "{log_base}/master.log"
    echo ""
    echo "🎉 所有batch串行执行完成！"
fi

echo ""
echo "📊 查看结果:"
echo "  各batch日志: {log_base}/Batch_X/"
echo "  主控日志: {log_base}/master.log"

# 显示快速状态检查命令
echo ""
echo "💡 快速检查命令:"
echo "  tail -f {log_base}/master.log"
for batch in "${{batches[@]}}"; do
    echo "  tail -f {log_base}/Batch_$batch/summary.log"
done
"""
    
    with open(f"{script_base}/run_all_batches.sh", "w") as f:
        f.write(master_content)
    os.chmod(f"{script_base}/run_all_batches.sh", 0o755)

if __name__ == "__main__":
    generate_training_scripts()
    generate_summary_script()
    generate_master_script()
    print("所有训练脚本生成完成！")
    print("\n本次更新:")
    print("1. 为1D脚本添加了进度显示 (进度: X/Y)")
    print("2. 为3D脚本添加了进度显示 (进度: X/Y)")
    print("3. 为2D脚本的check_and_clean_jobs函数添加了改进的进程检测和GPU清理机制")
    print("4. 保持ND脚本和其他部分不变")
    print("5. 现在可以检测并清理训练完成但进程未退出的情况")
    print("\n改进解决了GPU资源释放问题！")
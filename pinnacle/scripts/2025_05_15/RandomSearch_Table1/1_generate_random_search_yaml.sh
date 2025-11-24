#!/bin/bash

# 批量生成随机搜索配置
cd /gpfs/0607-cluster/qingpowuwu/Project_4_PINNsAgent/1_Ours/PINNsAgent_Unified/pinnsagent_progress-open_source/pinnacle/config

# 直接设置参数，不需要外部输入
BATCH_COUNT=10   # 生成5个batch
NEXP=5         # 每个PDE生成10个配置

# 自动获取当前日期
CURRENT_DATE=$(date +%Y-%m-%d)

echo "开始生成 $BATCH_COUNT 个batch，每个PDE生成 $NEXP 个配置..."
echo "当前日期: $CURRENT_DATE"

for ((i=1; i<=BATCH_COUNT; i++))
do
    BATCH_NAME="Batch_$i"
    echo "正在生成 $BATCH_NAME ..."
    
    # 为每个batch创建不同的输出目录，包含日期
    OUT_DIR="2025_05_15/ICML_2025/RandomSearch_Table1/$CURRENT_DATE/$BATCH_NAME"
    
    # 生成1D PDE配置
    echo "  生成1D PDE配置..."
    python 1_random_search-yaml-argsparser.py --pde_type 1d --device 0 --nexp $NEXP --out_dir $OUT_DIR
    
    # 生成2D PDE配置
    echo "  生成2D PDE配置..."
    python 1_random_search-yaml-argsparser.py --pde_type 2d --device 0 --nexp $NEXP --out_dir $OUT_DIR
    
    # 生成3D PDE配置
    echo "  生成3D PDE配置..."
    python 1_random_search-yaml-argsparser.py --pde_type 3d --device 0 --nexp $NEXP --out_dir $OUT_DIR
    
    # 生成ND PDE配置
    echo "  生成ND PDE配置..."
    python 1_random_search-yaml-argsparser.py --pde_type nd --device 0 --nexp $NEXP --out_dir $OUT_DIR
    
    echo "$BATCH_NAME 生成完成！"
    echo "------------------------"
done

echo "所有 $BATCH_COUNT 个batch生成完成！"
echo "目录结构："
echo "./2025_05_15/ICML_2025/RandomSearch_Table1/"
echo "└── $CURRENT_DATE/"
for ((i=1; i<=BATCH_COUNT; i++))
do
    echo "    ├── Batch_$i/"
    echo "    │   ├── 1d/"
    echo "    │   ├── 2d/"
    echo "    │   ├── 3d/"
    echo "    │   └── nd/"
done
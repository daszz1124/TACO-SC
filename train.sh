#!/bin/bash

# 训练脚本 - 用于调用train.py进行模型训练

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # 无颜色

# 帮助函数
show_help() {
    echo "使用方法: $0 [选项...]"
    echo ""
    echo "选项:"
    echo "  -h, --help                 显示此帮助信息并退出"
    echo "  --dist_port PORT           分布式训练端口 (默认: 6411)"
    echo "  --train_dataset_root_path PATH  训练数据集根路径 (默认: /home/iisc/zsd/project/VG2SC/datasets/MSCOCO)"
    echo "  --lpips_coefficient VALUE  LPIPS损失系数 (默认: 3.50)"
    echo "  --joint_image_text_loss_coefficient VALUE  联合图像文本损失系数 (默认: 0.0025)"
    echo "  -e EPOCHS                  训练轮数 (默认: 50)"
    echo "  -lr LEARNING_RATE          学习率 (默认: 1e-4)"
    echo "  --aux-learning-rate AUX_LR 辅助学习率 (默认: 1e-3)"
    echo "  -n NUM_WORKERS             数据加载工作线程数 (默认: 8)"
    echo "  --lambda LAMBDA            Lambda参数 (默认: 0.0004)"
    echo "  --batch-size BATCH_SIZE    批次大小 (默认: 8)"
    echo "  --patch-size WIDTH HEIGHT  图像块大小 (宽 高，默认: 256 256)"
    echo "  --seed SEED                随机种子 (默认: 100)"
    echo "  --clip_max_norm NORM       梯度裁剪最大范数 (默认: 1.0)"
    echo "  --lr_epoch EPOCH1 EPOCH2   学习率调整轮次 (默认: 45 48)"
    echo ""
    echo "示例:"
    echo "  $0 --lambda 0.0008 --batch-size 16"
    echo "  $0 --help"
}

# 设置默认参数
DIST_PORT=6411
TRAIN_DATASET_ROOT_PATH=/home/iisc/zsd/project/VG2SC/datasets/MSCOCO
LPIPS_COEFFICIENT=3.50
JOINT_IMAGE_TEXT_LOSS_COEFFICIENT=0.0025
EPOCHS=50
LEARNING_RATE=1e-4
AUX_LEARNING_RATE=1e-3
NUM_WORKERS=8
LAMBDA=0.0004
BATCH_SIZE=32
PATCH_SIZE=("256" "256")
SEED=100
CLIP_MAX_NORM=1.0
LR_EPOCH=("45" "48")

# 解析命令行参数
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -lr)
            LEARNING_RATE=$(echo "$2" | bc -l)  # 使用 bc 转换为浮点数
            shift 2
            ;;
        --aux-learning-rate)
            AUX_LEARNING_RATE=$(echo "$2" | bc -l)  # 转换为浮点数
            shift 2
            ;;
        --lpips_coefficient)
            LPIPS_COEFFICIENT=$(echo "$2" | bc -l)  # 转换为浮点数
            shift 2
            ;;
        --joint_image_text_loss_coefficient)
            JOINT_IMAGE_TEXT_LOSS_COEFFICIENT=$(echo "$2" | bc -l)  # 转换为浮点数
            shift 2
            ;;
        -e)
            EPOCHS=$2  # 整数无需转换，但确保输入为数字
            shift 2
            ;;
        -n)
            NUM_WORKERS=$2  # 整数无需转换，但确保输入为数字
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2  # 整数无需转换，但确保输入为数字
            shift 2
            ;;
        --seed)
            SEED=$2  # 整数无需转换，但确保输入为数字
            shift 2
            ;;
        --clip_max_norm)
            CLIP_MAX_NORM=$(echo "$2" | bc -l)  # 转换为浮点数
            shift 2
            ;;
        --lambda)
            LAMBDA=$(echo "$2" | bc -l)  # 转换为浮点数
            shift 2
            ;;
        # 其他参数解析...
    esac
done

# 恢复位置参数
set -- "${POSITIONAL_ARGS[@]}"

# 验证参数
if [[ ${#POSITIONAL_ARGS[@]} -ne 0 ]]; then
    echo -e "${RED}错误: 不支持的参数 '${POSITIONAL_ARGS[@]}'${NC}" >&2
    show_help >&2
    exit 1
fi

# 验证图像块大小参数
if [[ ${#PATCH_SIZE[@]} -ne 2 ]]; then
    echo -e "${RED}错误: patch-size参数需要两个值 (宽 高)${NC}" >&2
    show_help >&2
    exit 1
fi

# 验证学习率调整轮次参数
if [[ ${#LR_EPOCH[@]} -ne 2 ]]; then
    echo -e "${RED}错误: lr_epoch参数需要两个值${NC}" >&2
    show_help >&2
    exit 1
fi

# 显示训练配置
echo -e "${GREEN}===== TRAINING CONFIGURATION =====${NC}"
echo -e "${YELLOW}Distributed Port:${NC} $DIST_PORT"
echo -e "${YELLOW}Dataset Path:${NC} $TRAIN_DATASET_ROOT_PATH"
echo -e "${YELLOW}LPIPS Coefficient:${NC} $LPIPS_COEFFICIENT"
echo -e "${YELLOW}Joint Loss Coefficient:${NC} $JOINT_IMAGE_TEXT_LOSS_COEFFICIENT"
echo -e "${YELLOW}Epochs:${NC} $EPOCHS"
echo -e "${YELLOW}Learning Rate:${NC} $LEARNING_RATE"
echo -e "${YELLOW}Aux Learning Rate:${NC} $AUX_LEARNING_RATE"
echo -e "${YELLOW}Workers:${NC} $NUM_WORKERS"
echo -e "${YELLOW}Lambda:${NC} $LAMBDA"
echo -e "${YELLOW}Batch Size:${NC} $BATCH_SIZE"
echo -e "${YELLOW}Patch Size:${NC} ${PATCH_SIZE[0]} ${PATCH_SIZE[1]}"
echo -e "${YELLOW}Seed:${NC} $SEED"
echo -e "${YELLOW}Clip Max Norm:${NC} $CLIP_MAX_NORM"
echo -e "${YELLOW}LR Epochs:${NC} ${LR_EPOCH[0]} ${LR_EPOCH[1]}"
echo -e "${GREEN}===============================${NC}"

# 确认是否开始训练
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training canceled${NC}"
    exit 0
fi

# 设置日志文件
LOG_FILE="training_log/train_log_$(date +%Y%m%d_%H%M%S)_lambda_${LAMBDA}.log"
echo -e "${GREEN}Training log saved to:${NC} $LOG_FILE"

# 构建并执行训练命令 - 修改参数名称以匹配train.py
TRAIN_CMD="python -u train.py \
    --dist_port $DIST_PORT \
    --train_dataset_root_path $TRAIN_DATASET_ROOT_PATH \
    --lpips_coefficient $LPIPS_COEFFICIENT \
    --joint_image_text_loss_coefficient $JOINT_IMAGE_TEXT_LOSS_COEFFICIENT \
    -e $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --aux_learning_rate $AUX_LEARNING_RATE \
    -n $NUM_WORKERS \
    --lambda $LAMBDA \
    --batch-size $BATCH_SIZE \
    --patch-size ${PATCH_SIZE[0]} ${PATCH_SIZE[1]} \
    --seed $SEED \
    --clip_max_norm $CLIP_MAX_NORM \
    --lr_epoch ${LR_EPOCH[0]} ${LR_EPOCH[1]}"

echo -e "${GREEN}Executing command:${NC}\n$TRAIN_CMD"

# 执行训练并记录日志
echo "Training started at: $(date)" > "$LOG_FILE"
echo "Command: $TRAIN_CMD" >> "$LOG_FILE"
echo -e "\n" >> "$LOG_FILE"

if $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"; then
    echo -e "\n${GREEN}Training completed successfully!${NC}" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "\n${RED}Training failed!${NC}" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi
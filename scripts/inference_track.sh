#!/bin/bash
# set -x

CONFIG=${1:-"configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml"}
CKPT=${2:-"pretrained_models/fast_421_res152_256x192.pth"}
IMAGE_DIR=${3:-"/mnt/d/data/posetrack18/images/val"}
OUTDIR=${4:-"/mnt/d/data/alphapose/results"}
ANNO_DIR=${5:-"/mnt/d/data/posetrack18/annotations/val"}

# 计算文件总数
total_files=$(find "${ANNO_DIR}" -maxdepth 1 -name "*.json" | wc -l)
current_file=0

# source ~/softwares/miniconda/bin/activate alphapose

# 使用find命令获取所有.json文件
for file in $(find "${ANNO_DIR}" -maxdepth 1 -name "*.json"); do
    # 更新当前文件计数
    ((current_file++))
    # 打印进度条
    echo -ne "处理进度：$current_file/$total_files\n"

    # 获取文件名前缀（去掉.json和路径）
    seq_name=$(basename "$file" .json)

    /home/jimmy/softwares/miniconda/envs/alphapose/bin/python scripts/demo_inference.py \
        --cfg ${CONFIG} \
        --checkpoint ${CKPT} \
        --indir "${IMAGE_DIR}/${seq_name}"\
        --outdir ${OUTDIR} \
        --detector yolox-x \
        --save_img --pose_track --vis_fast --eval --profile
done

echo -ne '\n完成！\n'
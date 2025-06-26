LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/infer_mscoco_$(date +%Y%m%d_%H%M%S).log"

run_mscoco_inference() {
    local image_folder=$1
    local checkpoint=$2
    local save_folder=$3
    local image_list_json=${4:-"./materials/mscoco_30k_list.json"}
    local cap_pair_json=${5:-"./materials/mscoco_val41k_img_cap_pair.json"}
    local clip_model_path=${6:-"./clip/clip-vit-base-patch32"}

    local checkpoint_name=$(basename "$checkpoint")

    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Running MSCOCO inference with parameters:" | tee -a "${LOG_FILE}"
    echo "Image folder:      ${image_folder}" | tee -a "${LOG_FILE}"
    echo "Checkpoint:        ${checkpoint_name}" | tee -a "${LOG_FILE}"
    echo "Save path:         ${save_folder}" | tee -a "${LOG_FILE}"
    echo "Image list JSON:   ${image_list_json}" | tee -a "${LOG_FILE}"
    echo "Caption pair JSON: ${cap_pair_json}" | tee -a "${LOG_FILE}"
    echo "CLIP model path:   ${clip_model_path}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"

    python -u inference_mscoco30k_taco.py \
        --image_folder_root "${image_folder}" \
        --checkpoint "${checkpoint}" \
        --save_folder "${save_folder}" \
        --image_list_json "${image_list_json}" \
        --cap_pair_json "${cap_pair_json}" \
        --clip_model_name "${clip_model_path}" 2>&1 | tee -a "${LOG_FILE}"

    echo "✅ MSCOCO inference completed for checkpoint ${checkpoint_name}" | tee -a "${LOG_FILE}"
}

run_mscoco_inference \
    "/home/iisc/zsd/project/VG2SC/datasets/MSCOCO/val2014" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.004.pth.tar" \
    "./compression_mscoco_result"

run_mscoco_inference \
    "/home/iisc/zsd/project/VG2SC/datasets/MSCOCO/val2014" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.0004.pth.tar" \
    "./compression_mscoco_result"

run_mscoco_inference \
    "/home/iisc/zsd/project/VG2SC/datasets/MSCOCO/val2014" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.0008.pth.tar" \
    "./compression_mscoco_result"

run_mscoco_inference \
    "/home/iisc/zsd/project/VG2SC/datasets/MSCOCO/val2014" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.0016.pth.tar" \
    "./compression_mscoco_result"

run_mscoco_inference \
    "/home/iisc/zsd/project/VG2SC/datasets/MSCOCO/val2014" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.009.pth.tar" \
    "./compression_mscoco_result"

run_mscoco_inference \
    "/home/iisc/zsd/project/VG2SC/datasets/MSCOCO/val2014" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.015.pth.tar" \
    "./compression_mscoco_result"

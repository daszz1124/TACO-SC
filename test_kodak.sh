LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/infer_kodak_$(date +%Y%m%d_%H%M%S).log"

run_kodak_inference() {
    local image_folder=$1
    local checkpoint=$2
    local save_path=$3
    local caption_json=${4:-"./materials/kodak_ofa.json"}
    local clip_model_path=${5:-"./clip/clip-vit-base-patch32"}
    local checkpoint_name=$(basename "$checkpoint")

    echo "========================================" | tee -a "${LOG_FILE}"
    echo "Running Kodak inference with parameters:" | tee -a "${LOG_FILE}"
    echo "Image folder:     ${image_folder}" | tee -a "${LOG_FILE}"
    echo "Checkpoint:       ${checkpoint_name}" | tee -a "${LOG_FILE}"
    echo "Save path:        ${save_path}" | tee -a "${LOG_FILE}"
    echo "Caption JSON:     ${caption_json}" | tee -a "${LOG_FILE}"
    echo "CLIP model path:  ${clip_model_path}" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"

    python -u inference_kodak_taco.py \
        --image_folder_root "${image_folder}" \
        --checkpoint "${checkpoint}" \
        --save_path "${save_path}" \
        --caption_json "${caption_json}" \
        --clip_model_name "${clip_model_path}" 2>&1 | tee -a "${LOG_FILE}"

    echo "✅ Inference completed for checkpoint ${checkpoint_name}" | tee -a "${LOG_FILE}"
}

run_kodak_inference \
    "/home/iisc/zsd/project/VG2SC/TACO/kodak" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.004.pth.tar" \
    "./compression_kodak_results"

run_kodak_inference \
    "/home/iisc/zsd/project/VG2SC/TACO/kodak" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.0004.pth.tar" \
    "./compression_kodak_results"

run_kodak_inference \
    "/home/iisc/zsd/project/VG2SC/TACO/kodak" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.0008.pth.tar" \
    "./compression_kodak_results"

run_kodak_inference \
    "/home/iisc/zsd/project/VG2SC/TACO/kodak" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.0016.pth.tar" \
    "./compression_kodak_results"

run_kodak_inference \
    "/home/iisc/zsd/project/VG2SC/TACO/kodak" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.009.pth.tar" \
    "./compression_kodak_results"

run_kodak_inference \
    "/home/iisc/zsd/project/VG2SC/TACO/kodak" \
    "/home/iisc/zsd/project/VG2SC/TACO/checkpoint/lambda_0.015.pth.tar" \
    "./compression_kodak_results"

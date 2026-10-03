#!/usr/bin/env bash
# ==============================================================================
# download_dataset.sh
# 
# Automated Downloader for 5G-Advanced LTM Channel Datasets
# Publication: "Risk-Aware Distributional Reinforcement Learning for 5G-Advanced
#               Handover Decisions" (IEEE TMLCN 2026)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

ZENODO_RECORD_ID="23124881" # Published Zenodo Record ID
ZENODO_BASE_URL="https://zenodo.org/api/records/${ZENODO_RECORD_ID}/files"

echo "======================================================================"
echo "5G LTM Distributional RL - Channel Gain & Precomputed Dataset Setup"
echo "Target Directory: ${DATA_DIR}"
echo "======================================================================"

mkdir -p "${DATA_DIR}"

download_multipart_dataset() {
    local base_name="5g_advanced_ltm_channel_trajectories.zip"
    if [ -d "${DATA_DIR}/train" ] && [ -d "${DATA_DIR}/test" ]; then
        echo "[INFO] Folders train/ and test/ already exist in ${DATA_DIR}. Skipping download."
        return 0
    fi

    echo "[INFO] Downloading 5G LTM dataset from Zenodo (Record ${ZENODO_RECORD_ID})..."
    echo "[INFO] Downloading README.txt..."
    if command -v curl >/dev/null 2>&1; then
        curl -f -s -L --retry 3 "${ZENODO_BASE_URL}/README.txt/content" -o "${DATA_DIR}/README.txt" || true
    elif command -v wget >/dev/null 2>&1; then
        wget -q -c --tries=3 "${ZENODO_BASE_URL}/README.txt/content" -O "${DATA_DIR}/README.txt" || true
    fi

    for part_idx in 01 02 03 04 05 06; do
        local part_file="${base_name}.part${part_idx}"
        echo "[INFO] Downloading ${part_file}..."
        if command -v curl >/dev/null 2>&1; then
            curl -f -L --retry 3 "${ZENODO_BASE_URL}/${part_file}/content" -o "${DATA_DIR}/${part_file}"
        elif command -v wget >/dev/null 2>&1; then
            wget -c --tries=3 "${ZENODO_BASE_URL}/${part_file}/content" -O "${DATA_DIR}/${part_file}"
        else
            echo "[ERROR] Neither wget nor curl found. Please install one to continue." >&2
            exit 1
        fi
    done

    echo "[INFO] Concatenating 6 volumes into ${base_name}..."
    cat "${DATA_DIR}/${base_name}.part"* > "${DATA_DIR}/${base_name}"
    
    echo "[INFO] Extracting ${base_name}..."
    unzip -q "${DATA_DIR}/${base_name}" -d "${DATA_DIR}"
    
    # Clean up split archives
    rm -f "${DATA_DIR}/${base_name}" "${DATA_DIR}/${base_name}.part"*
    echo "[SUCCESS] Dataset successfully extracted to ${DATA_DIR}/train and ${DATA_DIR}/test."
}

cat << 'EOF'
Datasets needed for full replication:
1. 5g_advanced_ltm_channel_trajectories.zip (contains train/ with 2,000 UEs and test/ with 1,000 UEs)
2. Precomputed / Precomputed_v2 (Optional: generated in <2 min via preprocess_dataset.py)

If running in local offline mode, ensure train and test folders (or ChannelGains / ChannelGains_v2)
are placed in data/
EOF

# Download and extract dataset from Zenodo
download_multipart_dataset

echo "======================================================================"
echo "Dataset preparation script ready."
echo "======================================================================"

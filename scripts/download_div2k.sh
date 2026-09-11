#!/usr/bin/env bash
set -euo pipefail

# Download the public DIV2K HR images and arrange them for the original MBRS
# dataloader. The default target is the requested mounted data disk.

MOUNT_ROOT="${WMCONTENT_ROOT:-/mnt/wmcontent}"
DATASET_ROOT="${DATASET_ROOT:-${MOUNT_ROOT}/GLX/icassp/MBRS/datasets}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${DATASET_ROOT}/downloads}"
RAW_ROOT="${DATASET_ROOT}/raw"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
VALID_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

if [[ ! -d "${MOUNT_ROOT}" ]]; then
  mkdir -p "${MOUNT_ROOT}"
fi
if [[ ! -w "${MOUNT_ROOT}" ]]; then
  echo "ERROR: ${MOUNT_ROOT} is not writable. Remount it read-write, then rerun." >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_ROOT}" "${RAW_ROOT}"

download() {
  local url="$1"
  local output="$2"
  echo "Downloading $(basename "${output}")"
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "${output}" "${url}"
}

download "${TRAIN_URL}" "${DOWNLOAD_ROOT}/DIV2K_train_HR.zip"
download "${VALID_URL}" "${DOWNLOAD_ROOT}/DIV2K_valid_HR.zip"

unzip -t "${DOWNLOAD_ROOT}/DIV2K_train_HR.zip" >/dev/null
unzip -t "${DOWNLOAD_ROOT}/DIV2K_valid_HR.zip" >/dev/null

if [[ ! -d "${RAW_ROOT}/DIV2K_train_HR" ]]; then
  unzip -q "${DOWNLOAD_ROOT}/DIV2K_train_HR.zip" -d "${RAW_ROOT}"
fi
if [[ ! -d "${RAW_ROOT}/DIV2K_valid_HR" ]]; then
  unzip -q "${DOWNLOAD_ROOT}/DIV2K_valid_HR.zip" -d "${RAW_ROOT}"
fi

link_image() {
  local split="$1"
  local relative_source="$2"
  local image_name="$3"
  local destination="${DATASET_ROOT}/${split}/${image_name}"
  mkdir -p "${DATASET_ROOT}/${split}"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    return 0
  fi
  ln -s "${relative_source}" "${destination}"
}

for number in $(seq 1 800); do
  printf -v id '%04d' "${number}"
  link_image train "../raw/DIV2K_train_HR/${id}.png" "${id}.png"
done

# DIV2K exposes 800 train HR and 100 validation HR images. MBRS expects three
# directories, so reserve 0801-0850 as validation and 0851-0900 as test.
for number in $(seq 801 850); do
  printf -v id '%04d' "${number}"
  link_image validation "../raw/DIV2K_valid_HR/${id}.png" "${id}.png"
done
for number in $(seq 851 900); do
  printf -v id '%04d' "${number}"
  link_image test "../raw/DIV2K_valid_HR/${id}.png" "${id}.png"
done

# The upstream scripts use a relative datasets/ path. Point it at the same
# mounted dataset when the link is absent; never replace a real directory.
DATASET_LINK="${PROJECT_ROOT}/datasets"
if [[ -L "${DATASET_LINK}" ]]; then
  :
elif [[ -e "${DATASET_LINK}" ]]; then
  echo "WARNING: leaving existing non-link at ${DATASET_LINK}" >&2
else
  ln -s "${DATASET_ROOT}" "${DATASET_LINK}"
fi

train_count=$(find -L "${DATASET_ROOT}/train" -maxdepth 1 -type f -name '*.png' | wc -l)
validation_count=$(find -L "${DATASET_ROOT}/validation" -maxdepth 1 -type f -name '*.png' | wc -l)
test_count=$(find -L "${DATASET_ROOT}/test" -maxdepth 1 -type f -name '*.png' | wc -l)
if [[ "${train_count}" -ne 800 || "${validation_count}" -ne 50 || "${test_count}" -ne 50 ]]; then
  echo "ERROR: unexpected image counts: train=${train_count}, validation=${validation_count}, test=${test_count}" >&2
  exit 1
fi

echo "DIV2K is ready at ${DATASET_ROOT} (train=${train_count}, validation=${validation_count}, test=${test_count})"

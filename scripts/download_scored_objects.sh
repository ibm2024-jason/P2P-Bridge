#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="${1:-data/objects}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-data/_scoredenoise_downloads}"
GDRIVE_URL="${GDRIVE_URL:-https://drive.google.com/drive/folders/1--MvLnP7dsBgBZiu46H0S32Y1eBa_j6P?usp=sharing}"

mkdir -p "${DEST_ROOT}" "${DOWNLOAD_ROOT}"

if ! python -c "import gdown" >/dev/null 2>&1; then
  python -m pip install gdown
fi

python -m gdown --folder "${GDRIVE_URL}" -O "${DOWNLOAD_ROOT}" --remaining-ok

find "${DOWNLOAD_ROOT}" -type f \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.tgz' \) | while read -r archive; do
  case "${archive}" in
    *.zip)
      unzip -o "${archive}" -d "${DOWNLOAD_ROOT}/extracted"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${DOWNLOAD_ROOT}/extracted"
      ;;
  esac
done

for name in PUNet PCNet examples; do
  found="$(find "${DOWNLOAD_ROOT}/extracted" "${DOWNLOAD_ROOT}" -type d -name "${name}" | head -n 1 || true)"
  if [ -n "${found}" ]; then
    rm -rf "${DEST_ROOT}/${name}"
    cp -r "${found}" "${DEST_ROOT}/${name}"
  fi
done

echo "Dataset directory prepared at: ${DEST_ROOT}"
find "${DEST_ROOT}" -maxdepth 3 -type d | sort | sed -n '1,40p'

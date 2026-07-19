#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="${1:-data/objects}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-data/_scoredenoise_downloads}"
GDRIVE_URL="${GDRIVE_URL:-https://drive.google.com/drive/folders/1--MvLnP7dsBgBZiu46H0S32Y1eBa_j6P?usp=sharing}"

echo "Preparing P2P object datasets from ScoreDenoise."
echo "Destination: ${DEST_ROOT}"
echo "Temporary download cache: ${DOWNLOAD_ROOT}"
echo
echo "Approximate size: GB-level after extraction. Public docs do not list an exact"
echo "total size, so this script prints the actual size with 'du -sh' after download."
echo "gdown will show per-file download progress bars."
echo

mkdir -p "${DEST_ROOT}" "${DOWNLOAD_ROOT}"

if ! python -c "import gdown" >/dev/null 2>&1; then
  echo "Installing gdown for Google Drive folder download..."
  python -m pip install gdown
fi

echo "Downloading ScoreDenoise object data..."
if python -m gdown --folder "${GDRIVE_URL}" -O "${DOWNLOAD_ROOT}" --remaining-ok; then
  echo "Download finished."
else
  echo "Retrying without --remaining-ok for older gdown versions..."
  python -m gdown --folder "${GDRIVE_URL}" -O "${DOWNLOAD_ROOT}"
fi

EXTRACT_ROOT="${DOWNLOAD_ROOT}/extracted"
mkdir -p "${EXTRACT_ROOT}"

echo "Extracting archives if any are present..."
while IFS= read -r archive; do
  echo "Extracting ${archive}"
  case "${archive}" in
    *.zip)
      unzip -o "${archive}" -d "${EXTRACT_ROOT}"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${EXTRACT_ROOT}"
      ;;
  esac
done < <(find "${DOWNLOAD_ROOT}" -type f \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.tgz' \) | sort)

echo "Organizing data into ${DEST_ROOT}..."
for name in PUNet PCNet examples; do
  found="$(find "${EXTRACT_ROOT}" "${DOWNLOAD_ROOT}" -type d -name "${name}" | head -n 1 || true)"
  if [ -n "${found}" ]; then
    mkdir -p "${DEST_ROOT}"
    if [ -e "${DEST_ROOT}/${name}" ]; then
      echo "Keeping existing ${DEST_ROOT}/${name}; remove it manually to replace."
    else
      cp -r "${found}" "${DEST_ROOT}/${name}"
    fi
  fi
done

echo
echo "Final dataset layout preview:"
find "${DEST_ROOT}" -maxdepth 4 -type d | sort | sed -n '1,80p'
echo
echo "Actual disk usage:"
du -sh "${DEST_ROOT}" "${DOWNLOAD_ROOT}" 2>/dev/null || true
echo

if [ ! -d "${DEST_ROOT}/PUNet" ]; then
  echo "Warning: ${DEST_ROOT}/PUNet was not found after download/extraction." >&2
  echo "If Google Drive changed layout or blocked automated download, download the" >&2
  echo "ScoreDenoise object data manually and place PUNet/PCNet under ${DEST_ROOT}." >&2
  exit 1
fi

echo "Done. P2P object data is ready at ${DEST_ROOT}."

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <patient_id>"
  echo "Example: $0 1"
  exit 1
fi

PATIENT_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIGMAS=(0.25 0.5 0.125 1.0)

for SIGMA in "${SIGMAS[@]}"; do
  echo "============================================================"
  echo "Running patient ${PATIENT_ID} with sigma=${SIGMA}"
  echo "============================================================"
  "$ROOT_DIR/run_patient_pipeline.sh" "$PATIENT_ID" "$SIGMA"
done

echo "============================================================"
echo "Sigma grid complete for patient ${PATIENT_ID}"

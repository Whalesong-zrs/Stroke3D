#!/usr/bin/env bash
# Historical entry point retained for compatibility. Configure paths through
# MV_DIR, COARSE_DIR, REFINE_DIR, and the optional variables documented in README.md.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/run_mesh_pipeline.sh" "$@"


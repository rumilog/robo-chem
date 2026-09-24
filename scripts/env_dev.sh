#!/usr/bin/env bash
# Activate the perception / simulation environment on a workstation that has
# NO Franka control stack: no ROS Noetic, no catkin workspace, no frankapy,
# no RealSense cameras on the USB bus.
#
# scripts/env.sh is the robot control PC and sources all of the above, so it
# cannot run here. This is its counterpart: a conda env (Python 3.10) holding
# torch + SAM 3 for the grounding service and mujoco for the simulator.
# Everything that talks to the real arm stays unavailable on purpose.
#
# Usage:  source scripts/env_dev.sh [conda-env-name]        # default: robochem

ROBOCHEM_ENV_NAME="${1:-${ROBOCHEM_ENV_NAME:-robochem}}"

# Resolve the repo root from this file, so the script works from any cwd.
_this="${BASH_SOURCE[0]:-$0}"
export ROBOCHEM_ROOT="$(cd "$(dirname "$_this")/.." && pwd)"

# `conda activate` needs conda's shell FUNCTION, which only a shell that ran
# `conda init` has; a bare conda binary on PATH cannot activate. So load the
# hook from wherever conda was installed unless the function already exists.
if ! type conda 2>/dev/null | grep -q function; then
    for _p in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
        [ -f "$_p/etc/profile.d/conda.sh" ] && . "$_p/etc/profile.d/conda.sh" && break
    done
fi
if type conda 2>/dev/null | grep -q function; then
    conda activate "$ROBOCHEM_ENV_NAME" || echo "  WARNING: conda env '$ROBOCHEM_ENV_NAME' not found"
else
    echo "  WARNING: no conda found; using whatever python is on PATH"
fi

export PYTHONPATH="$ROBOCHEM_ROOT:$PYTHONPATH"
export GROUNDING_URL="${GROUNDING_URL:-http://127.0.0.1:5005}"

# .env stores the key as lowercase openai_api_key, but the OpenAI SDK only
# reads OPENAI_API_KEY, so normalize it here (same as scripts/env.sh).
if [ -f "$ROBOCHEM_ROOT/.env" ]; then
    set -a
    . "$ROBOCHEM_ROOT/.env"
    set +a
    if [ -n "$openai_api_key" ] && [ -z "$OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY="$openai_api_key"
    fi
fi

cd "$ROBOCHEM_ROOT"

echo "robochem dev env ready (no ROS / no robot)"
echo "  python  : $(command -v python)"
echo "  root    : $ROBOCHEM_ROOT"
echo "  ground  : $GROUNDING_URL"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "  WARNING: OPENAI_API_KEY is not set; VLM scene description will fail."
fi

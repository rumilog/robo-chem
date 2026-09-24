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
# Usage:  source scripts/env_dev.sh [env-name-or-path]      # default: auto
#
# The sim env is a conda env named `robochem` on some workstations and the
# in-repo venv `perception_env/` (python 3.10 + mujoco) on others, which is
# where requirements.txt says to install mujoco. Look for a venv first and
# fall back to conda, so the same line works on both; pass an explicit name
# or path to override. The robot venv (~/franka, python 3.8) is NOT it --
# mujoco>=3.2 needs python >=3.9 and will not install there.

ROBOCHEM_ENV_NAME="${1:-${ROBOCHEM_ENV_NAME:-}}"

# Resolve the repo root from this file, so the script works from any cwd.
_this="${BASH_SOURCE[0]:-$0}"
export ROBOCHEM_ROOT="$(cd "$(dirname "$_this")/.." && pwd)"

_activated=""

# An explicit path to a venv, or a bare name that matches one in the repo.
for _cand in "$ROBOCHEM_ENV_NAME" "$ROBOCHEM_ROOT/$ROBOCHEM_ENV_NAME" \
             "$ROBOCHEM_ROOT/perception_env"; do
    [ -n "$_cand" ] || continue
    if [ -f "$_cand/bin/activate" ]; then
        . "$_cand/bin/activate" && _activated="venv $_cand"
        break
    fi
done

# `conda activate` needs conda's shell FUNCTION, which only a shell that ran
# `conda init` has; a bare conda binary on PATH cannot activate. So load the
# hook from wherever conda was installed unless the function already exists.
if [ -z "$_activated" ]; then
    if ! type conda 2>/dev/null | grep -q function; then
        for _p in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
            [ -f "$_p/etc/profile.d/conda.sh" ] && . "$_p/etc/profile.d/conda.sh" && break
        done
    fi
    _name="${ROBOCHEM_ENV_NAME:-robochem}"
    if type conda 2>/dev/null | grep -q function; then
        if conda activate "$_name" 2>/dev/null; then
            _activated="conda $_name"
        else
            echo "  WARNING: no venv found and conda env '$_name' does not exist"
        fi
    else
        echo "  WARNING: no venv and no conda found; using whatever python is on PATH"
    fi
fi

# mujoco is the one import that separates the sim env from the robot venv.
if ! python -c "import mujoco" >/dev/null 2>&1; then
    echo "  WARNING: 'import mujoco' fails for $(command -v python)."
    echo "           The simulator will not run. Expected the python 3.10 env"
    echo "           at $ROBOCHEM_ROOT/perception_env (see requirements.txt)."
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

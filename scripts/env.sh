#!/usr/bin/env bash
# Activates everything needed to talk to the Franka arm and the RealSense cage.
#
# The default (base) conda env is Python 3.13 with none of the robot stack
# installed; frankapy lives in the Python 3.8 venv at /home/rumi/franka, and
# it cannot import franka_interface_msgs unless the catkin workspace is sourced.
#
# Usage:  source scripts/env.sh

source /opt/ros/noetic/setup.bash
source /home/rumi/frankapy/catkin_ws/devel/setup.bash
source /home/rumi/franka/bin/activate

export ROBOCHEM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROBOCHEM_ROOT:$PYTHONPATH"
export SAM_CHECKPOINT="$ROBOCHEM_ROOT/weights/sam_vit_b_01ec64.pth"
export SAM_MODEL_TYPE="vit_b"

# Open-vocabulary grounding runs in perception_env (Python 3.10) as a separate
# service, because frankapy pins this environment to Python 3.8.
export GROUNDING_URL="${GROUNDING_URL:-http://127.0.0.1:5005}"

# .env stores the key as lowercase openai_api_key, but the OpenAI SDK only
# reads OPENAI_API_KEY, so normalize it here.
if [ -f "$ROBOCHEM_ROOT/.env" ]; then
    set -a
    . "$ROBOCHEM_ROOT/.env"
    set +a
    if [ -n "$openai_api_key" ] && [ -z "$OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY="$openai_api_key"
    fi
fi

cd "$ROBOCHEM_ROOT"

echo "robochem env ready"
echo "  python  : $(which python)"
echo "  root    : $ROBOCHEM_ROOT"
echo "  ground  : $GROUNDING_URL"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "  WARNING: OPENAI_API_KEY is not set; VLM scene description will fail."
fi

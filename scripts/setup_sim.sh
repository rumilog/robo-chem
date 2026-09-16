#!/usr/bin/env bash
# Set up the MuJoCo simulation of the cell on a fresh machine.
#
# The simulator has no frankapy and no ROS dependency, so it does NOT go in the
# robot venv and you must NOT source scripts/env.sh first. This script builds
# (or reuses) a plain Python environment, installs the six packages the sim
# needs, pre-fetches the Panda model, and runs the smoke test.
#
# Usage:
#   bash scripts/setup_sim.sh                # reuse perception_env, else make sim_env
#   bash scripts/setup_sim.sh path/to/venv   # use this env instead
#   SKIP_TEST=1 bash scripts/setup_sim.sh    # install only
#
# Safe to re-run: it never deletes an environment, only adds to one.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- pick an environment ----------------------------------------------------
if [ "$#" -ge 1 ]; then
    VENV="$1"
elif [ -x "perception_env/bin/python" ]; then
    VENV="perception_env"          # the perception env already exists here
else
    VENV="sim_env"
fi
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    # mujoco needs 3.9+; 3.10 and 3.13 are both known good for this stack.
    BASE="$(command -v python3.13 || command -v python3.12 || \
            command -v python3.11 || command -v python3.10 || command -v python3)"
    echo "Creating $VENV with $BASE ($("$BASE" --version))"
    "$BASE" -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
fi

echo "Using $("$PY" --version) at $PY"

# --- dependencies -----------------------------------------------------------
# numpy/mujoco/robot_descriptions drive the simulation itself. cv2, sklearn and
# openai are imported on the way into robochem.vision, which SimVision subclasses
# so that reconstruction runs through the project's real fusion code; Pillow
# comes in through ExperimentLogger, which run_experiment.py imports at module
# level whether or not the run is simulated. open3d is deliberately absent:
# nothing on the simulated path needs it, and it has no wheel for every Python
# the rest of this list supports.
echo "Installing simulation dependencies..."
"$PY" -m pip install --quiet \
    numpy mujoco robot_descriptions opencv-python scikit-learn openai Pillow

# --- model ------------------------------------------------------------------
# mujoco_menagerie is cloned to ~/.cache/robot_descriptions on first use
# (a few hundred MB). Do it now so the first real run is not a silent wait.
echo "Fetching the Franka Panda model (first time only, this can take a while)..."
"$PY" - <<'PYEOF'
from robot_descriptions import panda_mj_description
print("  model:", panda_mj_description.MJCF_PATH)
PYEOF

# --- verify -----------------------------------------------------------------
# --help first: it is instant and exercises every module-level import in the
# entry point, which is how a missing dependency otherwise surfaces only once
# somebody runs a real skill.
echo "Checking the CLI imports..."
"$PY" scripts/run_experiment.py --help > /dev/null

if [ "${SKIP_TEST:-0}" = "1" ]; then
    echo "Install done (SKIP_TEST=1, not running the smoke test)."
else
    echo "Running the simulation smoke test (a few minutes)..."
    "$PY" scripts/smoke_test_sim.py
fi

cat <<EOF

Setup complete. Run a skill in the viewer with:

  $PY scripts/run_experiment.py --sim \\
    --skill pick_up --params '{"object_name":"plastic beaker","z_offset":0.02}'

See robochem/sim/README.md for the rest.
EOF

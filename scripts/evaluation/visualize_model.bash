#!/bin/bash

SEED=$1
MODEL_PATH_AFTER_TOP="$2"
MODEL_NAME="$3"
CONFIG_NAME="$4"
NUM_EPISODES="$5"
INTENTION="$6"
RENDER="$7"

export PYTHONPATH=$PYTHONPATH:../../
DEFAULT_TOP_DIR="../../"
TOP_DIR=${MODEL_TOP_DIR:=${DEFAULT_TOP_DIR}}
echo "Using TOP_DIR OF ${TOP_DIR}"

COMMON_TOP="${TOP_DIR}/${MODEL_PATH_AFTER_TOP}"
MODEL_PATH="${COMMON_TOP}/${MODEL_NAME}"
CONFIG_PATH="${COMMON_TOP}/${CONFIG_NAME}"

PYTHON_TO_EXEC=$(cat <<-END 
../../rl_sandbox/examples/eval_tools/evaluate.py
--seed=${SEED}
--model_path=${MODEL_PATH} 
--config_path=${CONFIG_PATH}
--num_episodes=${NUM_EPISODES}
--intention=${INTENTION}
--model_path=${MODEL_PATH}
END
)

if [ "${RENDER}" = "true" ]; then
    PYTHON_TO_EXEC+=" --render"
fi

python ${PYTHON_TO_EXEC}
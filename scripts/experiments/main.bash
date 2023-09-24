#!/bin/bash

SEED=$1
DEVICE=$2
MAIN_TASK=$3

export PYTHONPATH=$PYTHONPATH:../../
PYTHON_TO_EXEC=$(cat <<-END
../../rl_sandbox/examples/main/main.py
--seed=${SEED}
--main_task=${MAIN_TASK}
--device=${DEVICE}
--max_steps=2000000
END
)

if [[ "${DEVICE}" == *"cuda"* ]]; then
    PYTHON_TO_EXEC+=" --gpu_buffer"
fi

python ${PYTHON_TO_EXEC}

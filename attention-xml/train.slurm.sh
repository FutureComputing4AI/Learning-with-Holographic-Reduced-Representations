#!/bin/bash

# This is for GPU allocation is available. #SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=output/slurm-%x-%a.out
#SBATCH --error=output/slurm-%x-%a.err

# Set the environment.
# source deactivate # Remove previous environments.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spp # Environment name.

# Execute the code.
set -o xtrace
TASK_ID=$((SLURM_ARRAY_TASK_ID))
NAME=$1
SAVE_MODEL=$2
EXP_NAME=$3
DIMS=$4
WITH_GRAD=${5}
WITHOUT_NEGATIVE=${6}
PROP_A=${7:-"0.55"} # Propensity value A. For Amazon-670K it is 0.6
PROP_B=${8:-"1.5"} # Propensity value B. For Amazon-670K it is 2.6

# Model information.
MODEL=("baseline" "hrr")
MODEL_NETWORK="AttentionXML"

# Select the model.
MODEL_TYPE=${MODEL[${TASK_ID}]}
FIN_EXP_NAME=${NAME}-${EXP_NAME}-${MODEL_TYPE}-${DIMS}-${WITH_GRAD}-${WITHOUT_NEGATIVE}
echo "Parameters: $NAME $SAVE_MODEL"
echo "            $MODEL_TYPE $EXP_NAME $DIMS"
echo "            ${WITH_GRAD} ${WITHOUT_NEGATIVE}"

# Construct list of options.
OPTIONS=""
if [ "$MODEL_TYPE" == "hrr" ]
then
    DATA_YAML=${NAME}-spn
    MODEL_YAML=${MODEL_NETWORK}-${NAME}-spn-${DIMS}
    LABEL_NAME=${MODEL_NETWORK}-${DIMS}-${NAME}-spn-${DIMS}
else
    DATA_YAML=${NAME}
    MODEL_YAML=${MODEL_NETWORK}-${NAME}
    LABEL_NAME=${MODEL_NETWORK}-0-${NAME}-baseline-0
fi

if [ "$WITH_GRAD" == "no-grad" ]
then
        OPTIONS="${OPTIONS} --no-grad"
fi

if [ "${WITHOUT_NEGATIVE}" == "without-negative" ]
then
        OPTIONS="${OPTIONS} --without-negative"
fi

# Train the the models.
echo $DATA_YAML, $MODEL_YAML
echo "OPTIONS: $OPTIONS"
python main.py --data-cnf configure/datasets/${DATA_YAML}.yaml --model-cnf configure/models/${MODEL_YAML}.yaml > results/${FIN_EXP_NAME}.results

# Evaluation.
echo "Test Results..."
python evaluation.py --results results/${LABEL_NAME}-labels.npy \
                     --targets data/${NAME}/test_labels.npy --train-labels data/${NAME}/train_labels.npy >> results/${FIN_EXP_NAME}.results

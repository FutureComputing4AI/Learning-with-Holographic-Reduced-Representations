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
MODEL=("baseline" "hrr")

# Select the model.
MODEL_TYPE=${MODEL[${TASK_ID}]}
# FIN_EXP_NAME=${NAME}-${EXP_NAME}-${MODEL_TYPE}-${DIMS}-${WITH_GRAD}-${WITHOUT_NEGATIVE}
FIN_EXP_NAME=${NAME}-${EXP_NAME}-${MODEL_TYPE}
echo "Parameters: $NAME $SAVE_MODEL"
echo "            $MODEL_TYPE $EXP_NAME $DIMS"
echo "            ${WITH_GRAD} ${WITHOUT_NEGATIVE}"

# Construct list of options.
OPTIONS=""
if [ "$MODEL_TYPE" == "hrr" ]
then
    OPTIONS="${OPTIONS} --hrr_labels"
    NAME="${NAME}_hrr"
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
# --dp 1 for data parallel option.
echo "OPTIONS: $OPTIONS"
python main.py --ds $NAME --mn $FIN_EXP_NAME -a ${PROP_A} -b ${PROP_B} --model_type glove-bin $OPTIONS > ../results/${FIN_EXP_NAME}.results

# Test the model.
echo "Test Results..."
python main.py --ds $NAME -a ${PROP_A} -b ${PROP_B} --model_type glove-bin --tr 0 --lm ../saved_models/$FIN_EXP_NAME/model_best_test $OPTIONS >> ../results/${FIN_EXP_NAME}.results

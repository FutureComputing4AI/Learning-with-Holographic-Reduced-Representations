#!/bin/bash

# This is for GPU allocation is available. # SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=output/slurm-%A-%a.out
#SBATCH --error=output/slurm-%A-%a.err

# Set the environment.
# source deactivate # Remove previous environments.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spp # Environment name.

# Execute the code.
set -o xtrace
TASK_ID=$((SLURM_ARRAY_TASK_ID))
NAME=$1
DATA_FILE=$2
TR_SPLIT=$3
TE_SPLIT=$4
SAVE_MODEL=$5
THRESHOLD=$6
EXP_NAME=$7
DIMS=$8
BATCH_SIZE=$9
TEST_BATCH_SIZE=${10}
WITH_GRAD=${11}
WITHOUT_NEGATIVE=${12}

MODEL=("baseline" "spn")

# Select the model.
MODEL_TYPE=${MODEL[${TASK_ID}]}
echo "Parameters: $NAME $DATA_FILE $TR_SPLIT $TE_SPLIT $SAVE_MODEL $THRESHOLD"
echo "            $MODEL_TYPE $EXP_NAME $DIMS $BATCH_SIZE $TEST_BATCH_SIZE"
echo "            ${WITH_GRAD} ${WITHOUT_NEGATIVE}"

# Construct list of options.
OPTIONS="--th $THRESHOLD --debug"
if [ "$MODEL_TYPE" == "baseline" ]
then
    OPTIONS="${OPTIONS} --baseline"
fi

if [ "$WITH_GRAD" == "no-grad" ]
then
        OPTIONS="${OPTIONS} --no-grad"
fi

if [ "${WITHOUT_NEGATIVE}" == "without-negative" ]
then
        OPTIONS="${OPTIONS} --without-negative"
fi

python run_classifier.py --data-file $DATA_FILE \
                         --tr-split $TR_SPLIT \
                         --te-split $TE_SPLIT --spn-dim $DIMS \
                         --save $SAVE_MODEL --name ${NAME}_${MODEL_TYPE}_${EXP_NAME} \
                         --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE \
                         $OPTIONS > $SAVE_MODEL/${NAME}_${MODEL_TYPE}_${EXP_NAME}.results
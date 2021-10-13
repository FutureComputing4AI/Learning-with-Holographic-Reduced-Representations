#!/bin/bash
# set -vx

# AUTHOR: Ashwinkumar Ganesan.

# NOTE: Usage:-
#       1. ./experiments.sh all (for running experiments on all datasets).
#       2. ./experiments.sh <dataset> (for running experiments on a specific dataset).
#       3. ./experiments.sh gather (for gathering the precision only).

# Config.
NAME=${1:-"all"}
MEM=256000
SAVE_LOC="results"
EXP_NAME=${2:-"test"}
MODEL_TYPE=${3:-"all"}
DIMS=${4:-400}
WITH_GRAD=${5:-"grad"} # no-grad for training with no gradient to p & n vectors.
WITHOUT_NEGATIVE=${6:-"with-negative"} # without-negative for training.
PROP_A=${7:-"0.55"} # Propensity value A.
PROP_B=${8:-"1.5"} # Propensity value B.

create_job () {
    echo "Location to save model: $SAVE_LOC/$1 ..."
    if [[ ( "$MODEL_TYPE" == "all" ) ]]; then
        echo "Creating jobs for both models..."
        sbatch  --job-name=$1-${DIMS}-all --mem=$MEM --array=0-1 --exclude=node[17-32] train.slurm.sh \
                $1 $SAVE_LOC/$1 $EXP_NAME $DIMS $2 $3 ${PROP_A} ${PROP_B}
    elif [[ ( "$MODEL_TYPE" == "baseline" ) ]]; then
        echo "Creating jobs for baseline model..."
        sbatch  --job-name=$1-${DIMS}-base --mem=$MEM --array=0 --exclude=node[17-32] train.slurm.sh \
                $1 $SAVE_LOC/$1 $EXP_NAME $DIMS $2 $3 ${PROP_A} ${PROP_B}
    elif [[ ( "$MODEL_TYPE" == "hrr" ) ]]; then
        echo "Creating jobs for HRR model..."
        sbatch  --job-name=$1-${DIMS}-hrr --mem=$MEM --array=1 --exclude=node[17-32] train.slurm.sh \
                $1 $SAVE_LOC/$1 $EXP_NAME $DIMS $2 $3 ${PROP_A} ${PROP_B}
    fi
}

# NOTE: Individual jobs for each dataset are easier to track.
#       This keeps the SLURM files simple.

# Eurlex dataset.
if [[ ( "$NAME" == "EUR-Lex" ) || ( "$NAME" == "all" ) ]]
then
    create_job EUR-Lex $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Wiki30k dataset.
if [[ ( "$NAME" == "Wiki10-31K" ) || ( "$NAME" == "all" ) ]]
then
    create_job Wiki10-31K $WITH_GRAD $WITHOUT_NEGATIVE
fi

# AmazonCat-13K dataset.
if [[ ( "$NAME" == "AmazonCat-13K" ) || ( "$NAME" == "all" ) ]]
then
    create_job AmazonCat-13K $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Amazon-670K dataset.
if [[ ( "$NAME" == "Amazon-670K" ) || ( "$NAME" == "all" ) ]]
then
    create_job Amazon-670K $WITH_GRAD $WITHOUT_NEGATIVE
fi

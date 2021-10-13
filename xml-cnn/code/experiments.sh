#!/bin/bash
# set -vx

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
        sbatch  --job-name=$1-all --mem=$MEM --array=0-1 --exclude=node[17-32] train.slurm.sh \
                $1 $SAVE_LOC/$1 $EXP_NAME $DIMS $2 $3 ${PROP_A} ${PROP_B}
    elif [[ ( "$MODEL_TYPE" == "baseline" ) ]]; then
        echo "Creating jobs for baseline model..."
        sbatch  --job-name=$1-base --mem=$MEM --array=0 --exclude=node[17-32] train.slurm.sh \
                $1 $SAVE_LOC/$1 $EXP_NAME $DIMS $2 $3 ${PROP_A} ${PROP_B}
    elif [[ ( "$MODEL_TYPE" == "hrr" ) ]]; then
        echo "Creating jobs for HRR model..."
        sbatch  --job-name=$1-hrr --mem=$MEM --array=1 --exclude=node[17-32] train.slurm.sh \
                $1 $SAVE_LOC/$1 $EXP_NAME $DIMS $2 $3 ${PROP_A} ${PROP_B}
    fi
}

# NOTE: Individual jobs for each dataset are easier to track.
#       This keeps the SLURM files simple.

# RCV1 dataset.
if [[ ( "$NAME" == "rcv1" ) || ( "$NAME" == "all" ) ]]
then
    create_job rcv1 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Eurlex dataset.
if [[ ( "$NAME" == "eurlex" ) || ( "$NAME" == "all" ) ]]
then
    create_job eurlex $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Wiki30k dataset.
if [[ ( "$NAME" == "wiki30k" ) || ( "$NAME" == "all" ) ]]
then
    create_job wiki30k $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Amazon12k dataset.
if [[ ( "$NAME" == "amazon12k" ) || ( "$NAME" == "all" ) ]]
then
    create_job amazon12k $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Amazon12k dataset.
if [[ ( "$NAME" == "amazon670K" ) || ( "$NAME" == "all" ) ]]
then
    create_job amazon670K $WITH_GRAD $WITHOUT_NEGATIVE
fi

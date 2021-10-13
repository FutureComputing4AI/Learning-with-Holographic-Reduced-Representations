#!/bin/bash
# set -vx

# Script to execute experiments with different datasets.
# AUTHOR: Ashwinkumar Ganesan.

# NOTE: Usage:-
#       1. ./experiments.sh all (for running experiments on all datasets).
#       2. ./experiments.sh <dataset> (for running experiments on a specific dataset).
#       3. ./experiments.sh gather (for gathering the precision only).

# Config.
NAME=${1:-"all"}
MEM=256000
SAVE_LOC="data/model+results"
EXP_NAME=${2:-"temp-exp"}
MODEL_TYPE=${3:-"all"}
DIMS=${4:-400}
THRESHOLD=${5:-0.3}
WITH_GRAD=${6:-"grad"} # no-grad for training with no gradient to p & n vectors.
WITHOUT_NEGATIVE=${7:-"with-negative"} # without-negative for training.
SAVE_FILE_NAME="$EXP_NAME.results"

create_job () {
    echo "Location to save model: $SAVE_LOC/$1 ..."
    if [[ ( "$MODEL_TYPE" == "all" ) ]]; then
        echo "Creating jobs for both models..."
        sbatch  --job-name=$5 --mem=$MEM --array=0-1 --exclude=node[17-32] train.slurm.sh \
                $1 $2 $3 $4 $SAVE_LOC/$1 $THRESHOLD $EXP_NAME $DIMS $6 $7 $8 $9
    elif [[ ( "$MODEL_TYPE" == "baseline" ) ]]; then
        echo "Creating jobs for baseline model..."
        sbatch  --job-name=$5 --mem=$MEM --array=0 --exclude=node[17-32] train.slurm.sh \
                $1 $2 $3 $4 $SAVE_LOC/$1 $THRESHOLD $EXP_NAME $DIMS $6 $7 $8 $9
    elif [[ ( "$MODEL_TYPE" == "spn" ) ]]; then
        echo "Creating jobs for SPN model..."
        sbatch  --job-name=$5 --mem=$MEM --array=1 --exclude=node[17-32] train.slurm.sh \
                $1 $2 $3 $4 $SAVE_LOC/$1 $THRESHOLD $EXP_NAME $DIMS $6 $7 $8 $9
    fi
}

# NOTE: Individual jobs for each dataset are easier to track.
#       This keeps the SLURM files simple.

# Bibtex dataset.
if [[ ( "$NAME" == "Bibtex" ) || ( "$NAME" == "all" ) ]]
then
    create_job Bibtex data/Bibtex/Bibtex_data.txt data/Bibtex/bibtex_trSplit.txt \
               data/Bibtex/bibtex_tstSplit.txt bibtex 64 64 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Delicious dataset.
if [[ ( "$NAME" == "Delicious" ) || ( "$NAME" == "all" ) ]]
then
    create_job Delicious data/Delicious/Delicious_data.txt data/Delicious/delicious_trSplit.txt \
               data/Delicious/delicious_tstSplit.txt delic 64 64 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Mediamill dataset.
if [[ ( "$NAME" == "Mediamill" ) || ( "$NAME" == "all" ) ]]
then
    create_job Mediamill data/Mediamill/Mediamill_data.txt data/Mediamill/mediamill_trSplit.txt \
               data/Mediamill/mediamill_tstSplit.txt mediam 64 64 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Eurlex-4K dataset.
if [[ ( "$NAME" == "Eurlex4k" ) || ( "$NAME" == "all" ) ]]
then
    create_job Eurlex4k None data/Eurlex4k/eurlex_train.txt data/Eurlex4k/eurlex_test.txt eurlex 64 64 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Wiki10 dataset.
if [[ ( "$NAME" == "Wiki10" ) || ( "$NAME" == "all" ) ]]
then
    create_job Wiki10 None data/Wiki10/train.txt data/Wiki10/test.txt wiki10 64 64 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# AmazonCat13K dataset.
if [[ ( "$NAME" == "AmazonCat13K" ) || ( "$NAME" == "all" ) ]]
then
    create_job AmazonCat13K None data/AmazonCat13K/train.txt data/AmazonCat13K/test.txt ama13k 64 64 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# Amazon670K dataset.
if [[ ( "$NAME" == "Amazon670K" ) || ( "$NAME" == "all" ) ]]
then
    create_job Amazon670K None data/Amazon670K/train.txt data/Amazon670K/test.txt ama670 16 16 $WITH_GRAD $WITHOUT_NEGATIVE
fi

# DeliciousLarge dataset.
if [[ ( "$NAME" == "DeliciousLarge" ) || ( "$NAME" == "all" ) ]]
then
    create_job DeliciousLarge None data/DeliciousLarge/deliciousLarge_train.txt \
               data/DeliciousLarge/deliciousLarge_test.txt dlarge 8 8 $WITH_GRAD $WITHOUT_NEGATIVE
fi

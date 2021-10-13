#!/bin/bash
# set -vx

# Script to combine results from different experiments.
# AUTHOR: Ashwinkumar Ganesan.

# Config.
NAME=${1:-"all"}
MEM=256000
SAVE_LOC="data/model+results"
EXP_NAME=${2:-"temp-exp"}
MODEL_TYPE=${3:-"all"}
DIMS=${4:-400}
THRESHOLD=${5:-0.3}
SAVE_FILE_NAME="$EXP_NAME.results"

get_results () {
    if [[ ( "$NAME" == "$1" ) || ( "$NAME" == "all" ) || ( "$NAME" == "gather" ) ]]
    then
        SAVE_FILE=$SAVE_LOC/$SAVE_FILE_NAME
        echo -e "\n" >> $SAVE_FILE
        echo "Dataset: $1" >> $SAVE_FILE

        if [[ ( "$MODEL_TYPE" == "baseline" ) || ( "$MODEL_TYPE" == "all" ) ]]; then
            echo "Baseline..." >> $SAVE_FILE
            tail -7 $SAVE_LOC/$1/$1_baseline_$EXP_NAME.results >> $SAVE_FILE
        fi

        if [[ ( "$MODEL_TYPE" == "spn" ) || ( "$MODEL_TYPE" == "all" ) ]]; then
            echo -e "\nSPN..." >> $SAVE_FILE
            tail -7 $SAVE_LOC/$1/$1_spn_$EXP_NAME.results >> $SAVE_FILE
        fi
    fi
}

echo "Delete old results..."
rm $SAVE_LOC/$SAVE_FILE_NAME

# Combine results.
get_results Bibtex
get_results Delicious
get_results Mediamill
get_results Eurlex4k
get_results Wiki10
get_results AmazonCat13K
get_results Amazon670K
get_results DeliciousLarge

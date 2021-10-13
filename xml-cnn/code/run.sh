#!/bin/bash
# set -vx

DATASET=${1:-"eurlex"}
USE_HRR=${2:-"False"}
EXP_NAME=${3:-"test"}
PARALLELIZE=${4:-"False"}

OPTIONS=""
if [[ "$PARALLELIZE" == "True" ]]
then
    echo "Train WITHOUT data parallelism..."
    OPTIONS="$OPTIONS --dp 1"
fi

# Build name for HRR and basic baseline models.
if [[ "$USE_HRR" == "False" ]]
then
    echo "Train WITHOUT HRR representations.."
elif [[ "$USE_HRR" == "True" ]]
then
    echo "Train WITH HRR representations.."
    OPTIONS="$OPTIONS --hrr_labels"
fi

echo "OPTIONS: $OPTIONS"
python main.py --ds $DATASET --mn $DATASET --model_type glove-bin $OPTIONS > ../results/${DATASET}.results

# Test the model.
echo "Test Results..."
python main.py --ds $DATASET --model_type glove-bin --tr 0 --lm ../saved_models/$DATASET/model_best_test >> ../results/${DATASET}.results

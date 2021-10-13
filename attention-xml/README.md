# HRR-AttentionXML
This is a modified implementation of [AttentionalXML architecture](https://arxiv.org/abs/1811.01727) to train with HRR label representations and perform inference.

## List of changes to the Codebase.
The XML-CNN codebase has been modified to with the following list of changes:
1. Retooled to use semantic pointers. The architecture can use HRRs to learn and infer labels.
2. The dataset and model YAML files have additional arguments for HRR label representations.

## NOTES
1. For details about datasets and how to setup the repository, please look at instructions [here](https://github.com/yourh/AttentionXML).
2. AttentionXML is NOT configured for tree-based inference and HRR is applied only to a standard inference with a softmax layer.
3. The codebase also contains two scripts, i.e., ```experiments.sh``` and ```train.slurm.sh``` for execution of training and evaluation jobs on a SLURM enabled cluster.

## Datasets Locations
* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Wiki-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)
* [Amazon-3M](https://drive.google.com/open?id=187vt5vAkGI2mS2WOMZ2Qv48YKSjNbQv4)

## Preprocess
Run ```preprocss.py``` including tokenizing the raw texts by NLTK as follows:
```bash
python preprocess.py \
--text-path data/Wiki10-31K/train_raw_texts.txt \
--tokenized-path data/Wiki10-31K/train_texts.txt \
--label-path data/Wiki10-31K/train_labels.txt \
--vocab-path data/Wiki10-31K/vocab.npy \
--emb-path data/Wiki10-31K/emb_init.npy \
--w2v-model data/glove.840B.300d.gensim

python preprocess.py \
--text-path data/Wiki10-31K/test_raw_texts.txt \
--tokenized-path data/Wiki10-31K/test_texts.txt \
--label-path data/Wiki10-31K/test_labels.txt \
--vocab-path data/Wiki10-31K/vocab.npy 
```

## XML Experiments
In this example let us consider the dataset: ```Wiki10-31K```.

To execute the baseline model:
```bash
python main.py --data-cnf configure/datasets/Wiki10-31K.yaml --model-cnf configure/models/AttentionXML-Wiki10-31K.yaml
```

To execute the same model with HRR labels:
```bash
python main.py --data-cnf configure/datasets/Wiki10-31K-spn.yaml --model-cnf configure/models/AttentionXML-Wiki10-31K.yaml
```

To evaluate the model:
```bash
LABEL_NAME=AttentionXML-400-Wiki10-31K-spn-400 # For baseline the LABEL_NAME is AttentionXML-0-Wiki10-31K-spn-baseline-0.
NAME=Wiki10-31K
python evaluation.py --results results/${LABEL_NAME}-labels.npy \
                     --targets data/${NAME}/test_labels.npy --train-labels data/${NAME}/train_labels.npy
```
where ```${LABEL_NAME}``` is the name of the file containing labels for the above experiment run. ```${NAME}``` is the name of the dataset.

References
----------
[AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727)
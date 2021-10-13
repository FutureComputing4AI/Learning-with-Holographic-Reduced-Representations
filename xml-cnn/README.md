# HRR-CNN
  This is a modified implementation of [XML-CNN](https://github.com/siddsax/XML-CNN) from this [repository](https://github.com/siddsax/XML-CNN) that uses HRR for labal representation and inference. The Pytorch implementation is of the paper [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf) with dynamic pooling.

## List of changes to the Codebase.
The XML-CNN codebase has been modified to with the following list of changes:
1. Retooled to use semantic pointers. The architecture can use HRRs to learn and infer labels.
2. Modifications to operate seamlessly with large datasets and models using a Pytorch dataset object.
3. The codebase also contains two scripts, i.e., ```experiments.sh``` and ```train.slurm.sh``` for execution of training and evaluation jobs on a SLURM enabled cluster.

### NOTE: Before running experiments, perform preprocessing as discussed [here](https://github.com/siddsax/XML-CNN).

Example Execution with RCV Dataset
----------------------------------
To train the model with HRR.
```bash
EXP_NAME="test"
PROP_A=0.55
PROP_B=1.5
python main.py --ds rcv1 --mn rcv1-${EXP_NAME}-hrr -a ${PROP_A} -b ${PROP_B} --model_type glove-bin --hrr_labels
```

To evaluate the model:
```bash
python main.py --ds $NAME -a ${PROP_A} -b ${PROP_B} --model_type glove-bin --tr 0 --lm ../saved_models/rcv1-${EXP_NAME}-hrr/model_best_test --hrr_labels
```

References
----------
[Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
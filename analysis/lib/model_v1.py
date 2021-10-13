"""
Models for training Multilabel classification tasks.
"""

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"

import numpy as np
from tqdm import tqdm
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Faiss for MIPS (maximum inner product search)
import faiss

# Internal.
from .metrics import compute_prop_metrics
from .embeddings import get_vectors, load_embeddings
from .mathops import get_appx_inv, circular_conv, complexMagProj
from .utils import Measure

# Create FFN (feedforward network) for the classification task.
# NOTE: For the baseline the output layer is n binary classification tasks.
# NOTE: Most optimal (when tested with Wiki10.) is size 768
FC_LAYER_SIZE = 768

# Network Design.
# Basic multilabel multilayer perceptron.
class MultiLabelMLP(nn.Module):
    def __init__(self, num_features, num_classes, debug=False):
        super(MultiLabelMLP, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.fc_layer_size = FC_LAYER_SIZE
        # self.MUL_FACTOR = 4
        self.MUL_FACTOR = 1

        if debug is True:
            print("Feature Size: {}".format(self.num_features))
            print("Number of Labels: {}".format(self.num_classes))

        # Network Layers.
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size * self.MUL_FACTOR)
        # self.fc3 = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.fc_layer_size * self.MUL_FACTOR)
        self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.num_classes)

        # Create measurements.
        self.time = {
            'train': Measure("Train"),
            'train_forward_pass': Measure("Train Forward Pass"),
            'train_loss': Measure("Train Loss"),
            'optimization': Measure("Optimization"),
            'test_forward_pass': Measure("Test Forward Pass"),
            'inference': Measure("Inference"),
            'data_load': Measure("Data Loader"),
        }


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        x = self.olayer(x)
        y = F.sigmoid(x) # NOTE: Can we use binary softmax instead of sigmoid?
        return x, y


# Baseline.
def baseline_train(log_interval, model, device, train_loader, optimizer, epoch):
    """
    Baseline network training.
    """
    total_loss = 0.0; batch_idx = 0
    pbar_main = tqdm(enumerate(train_loader), desc="Samples Completed: ")
    model.train()

    model.time['train'].start()
    model.time['data_load'].start()
    for batch_idx, (data, target) in pbar_main:
        model.time['data_load'].end()

        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()

        model.time['train_forward_pass'].start()
        s, _ = model(data)
        model.time['train_forward_pass'].end()

        # BCEWithLogitsLoss for multilabel-binary classification.
        model.time['train_loss'].start()
        loss = nn.BCEWithLogitsLoss()(s, target)
        model.time['train_loss'].end()

        model.time['optimization'].start()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        model.time['optimization'].end()

        model.time['data_load'].start()

    model.time['data_load'].end()

    model.time['train'].end()
    return total_loss/(batch_idx + 1)

"""
NOTE: Baseline inference does not have an extra label like SPN's inference.
      This is because SPN treats the labels similar to a sequence.
"""
def baseline_test(model, device, test_loader, threshold=0.25, propensity=None,
                  topk=5):
    """
    Threshold defines the decision point for a binary classification task.
    NOTE: Optimize threshold using Grid Search.
    """
    model.eval()
    with torch.no_grad():
        total_pr = 0.0; total_rec = 0.0; total_f1 = 0.0; total_loss = 0.0;
        all_acc = []
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device).float(), target.to(device)

            model.time['test_forward_pass'].start()
            s, y = model(data) # Y are the predictions.
            model.time['test_forward_pass'].end()

            loss = nn.BCEWithLogitsLoss()(s, target.float()) # Loss.
            total_loss += loss.item()

            model.time['inference'].start()
            predictions = (y >= threshold).long()
            model.time['inference'].end() # Measure forward pass during inference.

            # Correct predictions.
            correct_pred = torch.sum(predictions & target, axis=1).float()

            # Precision.
            pr = torch.mean(correct_pred / torch.sum(target, axis=1)).item()
            total_pr += pr

            # Recall.
            ind_rec = correct_pred / torch.sum(predictions, axis=1)
            ind_rec[ind_rec != ind_rec] = 0.0
            rec = torch.mean(ind_rec).item()
            total_rec += rec

            # F-1 Score.
            f1 = torch.mean(2 * correct_pred / (torch.sum(predictions, axis=1) + torch.sum(target, axis=1))).item()
            total_f1 += f1

            if propensity is not None:
                """
                NOTE: fout is an ordered list of labels based on their binary probability scores.
                      This is not ideal but is a requirement for the xmetrics package.
                      It is not suitable for the baseline architecture as label
                      predictions are considered independent of each other.
                """
                actual_outputs = target.cpu().numpy()
                predicted_outputs = y.cpu().numpy()
                acc = compute_prop_metrics(sparse.csr_matrix(actual_outputs),
                                           sparse.csr_matrix(predicted_outputs), propensity,
                                           topk=topk)
                all_acc.append(acc)

    # Compute metrics for current threshold.
    num_itr = idx + 1

    if propensity is not None:
        return total_loss/num_itr, total_f1/num_itr,\
               total_pr/num_itr, total_rec/num_itr, all_acc
    else:
        return total_loss/num_itr, total_f1/num_itr, \
               total_pr/num_itr, total_rec/num_itr


class SemanticPointerNetwork(nn.Module):
    def __init__(self, num_features, num_classes, dims, max_pred_size,
                 no_grad=False, load_vec=None, debug=False):
        super(SemanticPointerNetwork, self).__init__()

        # Initialization Parameters.
        self.num_classes = num_classes # Number of labels in the datasets.
        self.num_features = num_features
        self.fc_layer_size = FC_LAYER_SIZE
        # self.MUL_FACTOR = 4
        self.MUL_FACTOR = 1
        self.dims = dims
        self.debug = debug
        self.load_vec = load_vec

        # NOTE: Defines the maximum number of positive labels in a sample.
        self.max_label_size = max_pred_size

        if self.debug is True:
            print("Feature Size: {}".format(self.num_features))
            print("Number of Labels: {}".format(self.num_classes))
            print("Class vector dimension: {}".format(self.dims))

        # Network Layers.
        self.fc1 = nn.Linear(self.num_features, self.fc_layer_size)
        self.fc2 = nn.Linear(self.fc_layer_size, self.fc_layer_size * self.MUL_FACTOR)
        # self.fc3 = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.fc_layer_size * self.MUL_FACTOR)
        self.olayer = nn.Linear(self.fc_layer_size * self.MUL_FACTOR, self.dims)

        # Create a label embedding layer.
        # self.create_label_embedding()

        # P & N vectors.
        p_n_vec = get_vectors(2, self.dims, ortho=True)

        if no_grad:
            if self.debug:
                print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            if self.debug:
                print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)

        # Create measurements.
        self.time = {
            'train': Measure("Train"),
            'train_forward_pass': Measure("Train Forward Pass"),
            'train_loss': Measure("Train Loss"),
            'optimization': Measure("Optimization"),
            'test_forward_pass': Measure("Test Forward Pass"),
            'inference': Measure("Inference"),
            'faiss_inference': Measure("Faiss Inference"),
            'data_load': Measure("Data Loader"),
        }

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        x = self.olayer(x)
        return x

    def create_label_embedding(self):
        if self.load_vec is not None:
            if self.debug:
                print("Loading Pretrained Embeddings...")

            # Class labels.
            self._class_vectors = load_embeddings(self.load_vec, self.num_classes - 1)
        else:
            if self.debug:
                print("Generate new label embeddings...")

            # Class labels.
            self._class_vectors = get_vectors(self.num_classes, self.dims)

        if self.debug:
            print("Label Vectors: {}".format(self._class_vectors.shape))

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.num_classes, self.dims)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.num_classes, 1), dtype=torch.int8)
        weights[self.num_classes - 1] = 0 # NOTE IMPORTANT: Padding vector is made 0.
        self.class_weights = nn.Embedding(self.num_classes, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False


    def build_class_index(self):
        # Build an index of the class vectors.
        self._class_vec_index = faiss.IndexFlatIP(self.dims)
        self._class_vec_index.add(self._class_vectors.numpy())

    def search(self, query, k):
        # d, i are the distances and index for each query.
        # print(self._class_vectors.numpy().shape, np.transpose(query).shape)
        d, i = self._class_vec_index.search(query, k)
        return d, i

    # This is because torch is unable to pickle the index.
    # NOTE: This is temporary, in the future we need to implement a method to
    #       save the index, or rebuild while loading the model.
    # https://github.com/pytorch/pytorch/issues/32046
    def reset_index(self):
        self._class_vec_index = None

    def inference(self, s, size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(size, self.dims)
        else:
            vec = self.n.unsqueeze(0).expand(size, self.dims)

        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)
        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

def spp_loss(s, model, data, target, device, without_negative=False, normalize=False):
    """
    Train with SPP.
    """
    pos_classes = model.class_vec(target)   #(batch, no_label, dims)
    pos_classes = pos_classes * model.class_weights(target)

    # Normalize the class vectors.
    if normalize:
        tgt_shape = pos_classes.shape
        pos_classes = torch.reshape(pos_classes, (tgt_shape[0] * tgt_shape[1],
                                                  tgt_shape[2]))
        pos_classes = torch.reshape(complexMagProj(pos_classes), (tgt_shape[0], tgt_shape[1],
                                                   tgt_shape[2]))

    # Remove the padding idx vectors.
    pos_classes = pos_classes.to(device)

    # Positive prediction loss
    convolve = model.inference(s, data.size(0))
    cosine = torch.matmul(pos_classes, convolve.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
    J_p = torch.mean(torch.sum(1 - torch.abs(cosine), dim=-1))

    # Negative prediction loss.
    J_n = 0.0
    if without_negative is False:
        convolve = model.inference(s, data.size(0), positive=False)
        cosine = torch.matmul(pos_classes, convolve.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        J_n = torch.mean(torch.sum(torch.abs(cosine), dim=-1))

    # Total Loss.
    loss = J_n + J_p
    return loss, J_n, J_p


def spp_train(log_interval, model, device, train_loader, optimizer, epoch,
              without_negative=False):
    total_loss = 0; total_j_n = 0.0; total_j_p = 0.0
    pbar_main = tqdm(enumerate(train_loader), desc="Samples Completed: ")
    model.train()
    model.time['train'].start()
    model.time['data_load'].start()
    for batch_idx, (data, target) in pbar_main:
        model.time['data_load'].end()

        # Select.
        # Train with actual negative samples.
        data = data.to(device).float()
        optimizer.zero_grad()

        model.time['train_forward_pass'].start()
        s = model(data)
        model.time['train_forward_pass'].end()

        model.time['train_loss'].start()
        loss, J_n, J_p = spp_loss(s, model, data, target, device,
                                  without_negative=without_negative)
        model.time['train_loss'].end()

        # Send to GPU.
        target = target.to(device)

        model.time['optimization'].start()
        loss.backward()
        optimizer.step()
        model.time['optimization'].end()

        # Losses.
        total_loss += loss.item()
        total_j_p += J_p.item()
        if without_negative is False:
            total_j_n += J_n.item()

        model.time['data_load'].start()

    model.time['data_load'].end()
    model.time['train'].end()
    return total_loss/(batch_idx + 1), total_j_n/(batch_idx + 1), total_j_p/(batch_idx + 1)


def spp_test(model, device, test_loader, threshold=0.25, propensity=None,
             topk=5, without_negative=False):
    """
    Threshold defines the decision point for a binary classification task.
    """
    # Before evaluation, build the index for current class vectors.
    model.build_class_index()

    # Start evaluation.
    model.eval()
    with torch.no_grad():
        total_pr = 0.0; total_rec = 0.0; total_f1 = 0.0; total_loss = 0.0;
        all_acc = []
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device).float()

            model.time['test_forward_pass'].start()
            s = model(data) # Y are the predictions.
            batch_size = s.shape[0]
            y = model.inference(s, batch_size)
            model.time['test_forward_pass'].end()

            # Inference with faiss.
            y_cpu = y.cpu()
            y_faiss = y_cpu.numpy()
            model.time['faiss_inference'].start()
            _, faiss_predictions = model.search(y_faiss, model.max_label_size * 2)
            model.time['faiss_inference'].end()

            # (batch, no_classes)
            # weights = complexMagProj(model.class_vec.weight)
            model.time['inference'].start()
            y_cpu = torch.abs(torch.mm(y_cpu, model.class_vec.weight.t()))
            predictions = (y_cpu >= threshold).long()
            model.time['inference'].end()

            # Loss.
            loss, _, _ = spp_loss(s, model, data, target, device,
                                  without_negative=without_negative)
            total_loss += loss.item()

            # Construct the target one-hot vector.
            y_onehot = torch.LongTensor(batch_size, model.num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, target, 1)

            # Correct predictions.
            correct_pred = torch.sum(predictions & y_onehot, axis=1).float()

            # Precision.
            pr = torch.mean(correct_pred / torch.sum(y_onehot, axis=1)).item()
            total_pr += pr

            # Recall.
            ind_rec = correct_pred / torch.sum(predictions, axis=1)
            ind_rec[ind_rec != ind_rec] = 0.0
            rec = torch.mean(ind_rec).item()
            total_rec += rec

            # F-1 Score.
            f1 = torch.mean(2 * correct_pred / (torch.sum(predictions, axis=1) + torch.sum(y_onehot, axis=1))).item()
            total_f1 += f1

            if propensity is not None:
                # NOTE: fout is an ordered list of labels based on probability scores.
                # This is not ideal but is a requirement for the xmetrics package.
                actual_outputs = y_onehot.numpy()[:, :-1] # Remove the last column (padding_idx)
                predicted_outputs = y_cpu.numpy()[:, :-1]
                acc = compute_prop_metrics(sparse.csr_matrix(actual_outputs),
                                           sparse.csr_matrix(predicted_outputs), propensity,
                                           topk=topk)
                all_acc.append(acc)

    # Compute metrics for current threshold.
    num_itr = idx + 1

    # Reset the index.
    model.reset_index()

    if propensity is not None:
        return total_loss/num_itr, total_f1/num_itr,\
               total_pr/num_itr, total_rec/num_itr, all_acc
    else:
        return total_loss/num_itr, total_f1/num_itr, \
               total_pr/num_itr, total_rec/num_itr

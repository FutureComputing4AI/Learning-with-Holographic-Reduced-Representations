"""
Generate label vectors for a given dataset.
"""

from __future__ import print_function, division

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"


import os.path
import pprint as pp

# Graphs.
import pandas as pd
import numpy as np
import matplotlib
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px

import argparse

from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE
from gensim.test.utils import datapath

import torch
from torch.utils.data import Dataset
from lib.utils import print_command_arguments
from lib.mathops import npcomplexMagProj

# Commandline Arguments.
parser = argparse.ArgumentParser(description='Generate Label vectors from the dataset.')
parser.add_argument('--name', type=str, default='test', help='A unique experiment name.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 100)')
parser.add_argument('--topk', type=int, default=5, metavar='S',
                    help='Retreive top k labels (Default: 5).')
parser.add_argument('--save', type=str, default='.', help='Directory to save model and results.')
parser.add_argument('--data-file', type=str, default="None", help='Location of the data CSV file.')
parser.add_argument('--tr-split', type=str, help='Get the training split for dataset.')
parser.add_argument('--te-split', type=str, help='Get the test split for dataset.')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Print debug statements for verification.')
parser.add_argument('--spn-dim', type=int, default=400, metavar='S',
                    help='Label vector dimensions (Default: 400)')
parser.add_argument('--test', action='store_true', default=False,
                    help='Verify Label vectors previously generated.')


"""
Create a dataset class for loading data.
"""
class SPNLabelDataset(Dataset):
    def __init__(self, data_file):
        """
        The data is the form of a sparse representation.
        Args:
            data_file (string): Path to the file contain the training or testing data.
        """
        self._file_loc = data_file
        self.labels = []
        with open(self._file_loc, "r") as fptr:
            info = fptr.readline().split("\n")[0].split(" ")
            self.size, self.num_features, self.num_labels = int(info[0]), int(info[1]), int(info[2])
            max_size = 0; avg_size = 0
            for idx, row in enumerate(fptr):
                # Create empty rows for both features and labels.
                feat_row = [0.0 for i in range(0, self.num_features)]
                lbl_row = []

                # Extract labels.
                data = row.split("\n")[0].split(" ")
                for lb in data[0].split(","):
                    try:
                        lbl_row.append(lb)
                    except ValueError:
                        # NOTE: In case of an error duplicate the row.
                        #       Sometimes the features of current row spilover to a
                        #       new line. In this case, the next line features are
                        #       added to the previous line's features and it is replicated to maintain
                        #       the training / test split.
                        lbl_row = self.labels[-1]

                self.labels.append(lbl_row)
                avg_size += len(lbl_row)
                if max_size < len(lbl_row): # Find the input with maximum number of labels.
                    max_size = len(lbl_row)

        self.padding_idx = self.num_labels # This is the last index in the label vector vobcaulary.
        self.num_labels += 1
        self.max_size = max_size
        self.avg_size = int(avg_size / self.size)

        print("Padding IDX: {}".format(self.padding_idx))

        # Dataset Statistics.
        print("Number of Features: {}".format(self.num_features))
        print("Number of Labels: {}".format(self.num_labels))
        print("Number of Samples: {}".format(self.size))
        print("Maximum number of labels / sample: {}".format(self.max_size))
        print("Average number of labels / sample: {}".format(self.avg_size))

        # # Padding.
        # for row in self.labels:
        #     row += [str(self.padding_idx)]

    # Define the length of the dataset.
    def __len__(self):
        return self.size

    # Return a single sample from the dataset.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.labels[idx]

    def get_max_size(self):
        return self.max_size

# Use a skip-gram model to generate vectors.
def generate_vectors(dataset, args):
    fname = args.save + args.name + ".bin"
    if not os.path.isfile(fname):
        # Train a skip-gram model.
        if args.debug:
            print("Generating label vectors...")
        model = Word2Vec(sentences=dataset.labels , size=args.spn_dim, min_count=1,
                         sg=1, window=dataset.max_size - 1)
        model.wv.save_word2vec_format(fname, fvocab=args.save + args.name,
                                      binary=True)
    else:
        if args.debug:
            print("Loading pretrained label vectors...")
        model = KeyedVectors.load_word2vec_format(fname, binary=True)

    return model


def get_complex_model(model, args):
    fname = args.save + args.name + "-complex.bin"
    if not os.path.isfile(fname):
        vectors = [] # positions in vector space
        labels = [] # keep track of words to label our data again later
        for word in model.wv.vocab:
            vectors.append(model.wv[word])
            labels.append(word)

        # Normalize the embeddings.
        vectors = np.array(vectors) / np.linalg.norm(np.array(vectors))
        complex_vectors = np.array([npcomplexMagProj(v) for v in vectors])
        complex_model = KeyedVectors(args.spn_dim)
        complex_model.add(labels, complex_vectors)
        complex_model.wv.save_word2vec_format(fname, fvocab=args.save + args.name + "-complex",
                                              binary=True)
    else:
        complex_model = KeyedVectors.load_word2vec_format(fname, binary=True)

    return complex_model


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model.wv[word])
        labels.append(word)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = np.array([v[0] for v in vectors])
    y_vals = np.array([v[1] for v in vectors])
    print(x_vals.shape, y_vals.shape)
    return x_vals, y_vals, labels


def plot_graph(x_vals, y_vals, labels, fig_name="graphs/test"):
    # Plot the labels.
    traces = [
        go.Scatter(x=x_vals, y=y_vals, mode="markers", text=labels)
    ]
    fig = go.Figure(data=traces)
    if fig_name:
        print("Writing to file...")
        py.plot(fig, output_type='file', image_width=800, image_height=600,
                filename=fig_name + ".html", validate=False)


def find_common(model, complex_model, word, topn=10):
    nn_m = model.most_similar(word, topn=topn)
    nn_m_words = set([w[0] for w in nn_m])

    nn_c = complex_model.most_similar(word, topn=topn)
    nn_c_words = set([w[0] for w in nn_c])

    return len(nn_m_words & nn_c_words) / len(nn_m_words | nn_c_words)

# Start.
if __name__ == "__main__":
    args = parser.parse_args()
    print_command_arguments(args)

    # Locations.
    model_path = args.save + "/" + args.name + ".pth"

    # Begin.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True} if use_cuda else {'drop_last': True, 'num_workers': 8}

    # Device.
    print("Device Used: {}".format(device))

    # Create the dataset.
    eld = None
    if not args.test:
        if args.data_file != "None":
            eld = SPNLabelDataset(args.data_file)
        else: # When a single train split are available.
            eld = SPNLabelDataset(args.tr_split)

        max_pred_size = eld.get_max_size()
        num_classes = eld.num_labels # Number of labels in the datasets.
        num_features = eld.num_features

    # Load models.
    model = generate_vectors(eld, args)
    complex_model = get_complex_model(model, args)

    word = "3"
    iou = find_common(model, complex_model, word, topn=25)
    print("Word: {} IoU: {}".format(word, iou))

    # x_vals, y_vals, labels = reduce_dimensions(model)
    # plot_graph(x_vals, y_vals, labels, fig_name="analysis/graphs/" + args.name)
    #
    # x_vals, y_vals, labels = reduce_dimensions(complex_model)
    # plot_graph(x_vals, y_vals, labels, fig_name="analysis/graphs/" + args.name + "-complex")

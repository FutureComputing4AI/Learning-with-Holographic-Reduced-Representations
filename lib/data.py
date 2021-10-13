"""
Manage different datasets.
"""

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"


import numpy as np
from scipy import sparse
from scipy.stats import uniform

from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, SubsetRandomSampler


"""
Create a sparse matrix form to manage large datasets.
"""
class SparseMatrix(object):
    def __init__(self):
        self.row = []
        self.column = []
        self.data = []

    def append(self, r, c, value):
        self.row.append(r)
        self.column.append(c)
        self.data.append(value)

    def get_csrmatrix(self, shape):
        data, rows, columns = np.array(self.data), np.array(self.row), np.array(self.column)
        coo_matrix = sparse.coo_matrix((data, (rows,columns)), shape=shape)
        return coo_matrix.tocsc()

"""
Create a dataset class for loading data.
"""
class SPNDataset(Dataset):
    def __init__(self, data_file, dimension_reduction=False):
        """
        The data is the form of a sparse representation.
        Args:
            data_file (string): Path to the file contain the training or testing data.
        """
        self.dimension_reduction = dimension_reduction
        self._file_loc = data_file
        self.features = []
        self.labels = []
        with open(self._file_loc, "r") as fptr:
            info = fptr.readline().split("\n")[0].split(" ")
            self.size, self.num_features, self.num_labels = int(info[0]), int(info[1]), int(info[2])
            max_size = 0
            for idx, row in enumerate(fptr):
                # Create empty rows for both features and labels.
                feat_row = [0.0 for i in range(0, self.num_features)]
                lbl_row = []

                # Extract labels.
                data = row.split("\n")[0].split(" ")
                for lb in data[0].split(","):
                    try:
                        lbl_row.append(int(lb))
                    except ValueError:
                        # NOTE: In case of an error duplicate the row.
                        #       Sometimes the features of current row spilover to a
                        #       new line. In this case, the next line features are
                        #       added to the previous line's features and it is replicated to maintain
                        #       the training / test split.
                        lbl_row = self.labels[-1]
                        feat_row = self.features[-1]

                self.labels.append(lbl_row)
                if max_size < len(lbl_row): # Find the input with maximum number of labels.
                    max_size = len(lbl_row)

                # Extract features.
                for f in data[1:]:
                    feat = f.split(":")
                    feat_row[int(feat[0])] = float(feat[1])
                self.features.append(feat_row)

        self.padding_idx = self.num_labels # This is the last index in the label vector vobcaulary.
        self.num_labels += 1

        print("Padding IDX: {}".format(self.padding_idx))

        # Padding.
        for row in self.labels:
            row += [self.padding_idx for i in range(max_size - len(row))]

        # NOTE: Dimension is useful to create a dense vector.
        self.labels = np.array(self.labels)
        self.features = np.array(self.features)
        self.sparse_labels = sparse.csr_matrix(self.labels)

        if self.dimension_reduction is True:
            self.raw_features = np.array(self.features)
            self._pca = PCA(n_components=0.95)
            self.features = self._pca.fit_transform(self.raw_features)
        # else:
        #     self.features = np.array(self.features)

    # Define the length of the dataset.
    def __len__(self):
        return self.size

    # Return a single sample from the dataset.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], self.labels[idx]

    def get_one_hot(self, idx=None):
        y = np.zeros((self.labels.shape[0], self.num_labels))

        # https://stackoverflow.com/questions/39614516/using-a-numpy-array-to-assign-values-to-another-array
        y[np.arange(self.labels.shape[0])[:, None], self.labels] = 1
        if idx is None:
            return y[:, :-1] # Leave the last column as it is the padding value.
        else:
            return y[idx][:, :-1]

    def get_max_size(self):
        return self.labels.shape[1]


"""
Create a dataset class for loading data.
"""
class SPNSparseDataset(Dataset):
    def __init__(self, data_file, dimension_reduction=False):
        """
        The data is the form of a sparse representation.
        Args:
            data_file (string): Path to the file contain the training or testing data.
            dimension_reduction (boolean): Can be used in the future for dimension reduction of features.
        """
        self.dimension_reduction = dimension_reduction
        self._file_loc = data_file
        self.features = SparseMatrix()
        self.splabels = SparseMatrix()
        self.labels = []
        with open(self._file_loc, "r") as fptr:
            info = fptr.readline().split("\n")[0].split(" ")
            self.size, self.num_features, self.num_labels = int(info[0]), int(info[1]), int(info[2])
            max_size = 0
            for row_id, row in enumerate(fptr):
                lbl_row = []

                # Extract labels.
                data = row.split("\n")[0].split(" ")
                for lb in data[0].split(","):
                    try:
                        lbl_row.append(int(lb))
                        self.splabels.append(row_id, int(lb), 1)
                    except:
                        print(row_id)

                self.labels.append(lbl_row)
                if max_size < len(lbl_row): # Find the input with maximum number of labels.
                    max_size = len(lbl_row)

                # Extract features.
                for f in data[1:]:
                    feat = f.split(":")
                    self.features.append(row_id, int(feat[0]), float(feat[1]))

        self.padding_idx = self.num_labels # This is the last index in the label vector vobcaulary.
        self.num_labels = self.num_labels + 1
        print("Padding IDX: {}".format(self.padding_idx))

        # Padding.
        for row in self.labels:
            row += [self.padding_idx for i in range(max_size - len(row))]

        # NOTE: Dimension is useful to create a dense vector.
        self.labels = np.array(self.labels)
        self.features = self.features.get_csrmatrix(shape=(self.size,
                                                           self.num_features))

        # Manage sparse label form.
        self.splabels = self.splabels.get_csrmatrix(shape=(self.size,
                                                           self.num_labels))

    # Define the length of the dataset.
    def __len__(self):
        return self.size

    # Return a single sample from the dataset.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return np.squeeze(self.features[idx].toarray()), self.labels[idx]

    def get_one_hot(self, idx=None):
        y = np.zeros((self.labels.shape[0], self.num_labels))

        # https://stackoverflow.com/questions/39614516/using-a-numpy-array-to-assign-values-to-another-array
        y[np.arange(self.labels.shape[0])[:, None], self.labels] = 1
        if idx is None:
            return y[:, :-1]
        else:
            return y[idx][:, :-1]

    def get_max_size(self):
        return self.labels.shape[1]


"""
Create a dataset class for loading multilabel data.
"""
class MultiLabelDataset(Dataset):
    def __init__(self, data_file, dimension_reduction=False):
        """
        The data is the form of a sparse representation.
        Args:
            data_file (string): Path to the file contain the training or testing data.
        """
        self.dimension_reduction = dimension_reduction
        self._file_loc = data_file
        self.features = []
        self.labels = []
        with open(self._file_loc, "r") as fptr:
            info = fptr.readline().split("\n")[0].split(" ")
            self.size, self.num_features, self.num_labels = int(info[0]), int(info[1]), int(info[2])
            for row in fptr:
                try:
                    # Create empty rows for both features and labels.
                    feat_row = [0.0 for i in range(0, self.num_features)]
                    lbl_row = [0 for i in range(0, self.num_labels)]

                    # Extract labels.
                    data = row.split("\n")[0].split(" ")
                    for lb in data[0].split(","):
                        try:
                            lbl_row[int(lb)] = 1
                        except ValueError:
                            # NOTE: In case of an error duplicate the row.
                            #       Sometimes the features of current row spilover to a
                            #       new line. In this case, the next line features are
                            #       added to the previous line's features and it is replicated to maintain
                            #       the training / test split.
                            lbl_row = self.labels[-1]
                            feat_row = self.features[-1]


                    self.labels.append(lbl_row)

                    # Extract features.
                    for f in data[1:]:
                        feat = f.split(":")
                        feat_row[int(feat[0])] = float(feat[1])
                    self.features.append(feat_row)
                except:
                    print(row)
                    break

        self.labels = np.array(self.labels)
        self.features = np.array(self.features)
        self.sparse_labels = sparse.csr_matrix(self.labels)

        if self.dimension_reduction is True:
            self.raw_features = np.array(self.features)
            self._pca = PCA(n_components=0.95)
            self.features = self._pca.fit_transform(self.raw_features)

    # Define the length of the dataset.
    def __len__(self):
        return self.size

    # Return a single sample from the dataset.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], self.labels[idx]


"""
Create a dataset class for loading multilabel data.
Manage dataset in sparse format when it is large.
"""
class MultiLabelSparseDataset(Dataset):
    def __init__(self, data_file, dimension_reduction=False):
        """
        The data is the form of a sparse representation.
        Args:
            data_file (string): Path to the file contain the training or testing data.
        """
        self.dimension_reduction = dimension_reduction
        self._file_loc = data_file
        self.features = SparseMatrix()
        self.labels = SparseMatrix()
        with open(self._file_loc, "r") as fptr:
            info = fptr.readline().split("\n")[0].split(" ")
            self.size, self.num_features, self.num_labels = int(info[0]), int(info[1]), int(info[2])
            for row_id, row in enumerate(fptr):
                try:
                    # Extract labels.
                    data = row.split("\n")[0].split(" ")
                    for lb in data[0].split(","):
                        self.labels.append(row_id, int(lb), 1)

                    # Extract features.
                    for f in data[1:]:
                        feat = f.split(":")
                        self.features.append(row_id, int(feat[0]), float(feat[1]))
                except:
                    print(row)
                    break

        self.labels = self.labels.get_csrmatrix(shape=(self.size,
                                                       self.num_labels))
        self.features = self.features.get_csrmatrix(shape=(self.size,
                                                           self.num_features))

    # Define the length of the dataset.
    def __len__(self):
        return self.size

    # Return a single sample from the dataset.
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return np.squeeze(self.features[idx].toarray()), np.squeeze(self.labels[idx].toarray())


# Create the Sampler.
# This is for datasets where the sampled indices exist.
class Sampler(object):
    def __init__(self, train_sample_file, test_sample_file=None):
        def file_to_np(filename):
            # Read the sample file.
            idx = []
            with open(filename, "r") as trf:
                for row in trf:
                    split = row.split("\n")[0].split(" ")
                     # All data in the file seems to be shift by 1. The array starts from 1. Hence int(x) - 1
                    idx.append(list(map(lambda x: int(x) - 1, split)))

            return np.array(idx)

        # Training file.
        self.train_loc = train_sample_file
        self.train_idx = file_to_np(self.train_loc)
        self.size = self.train_idx.shape[0]

        # Split the training file into train / validation.
        all_idx = [i for i in range(self.size)]
        val = np.random.choice(all_idx, int(self.size / 10), replace=False) # 10% of training for validation.
        tr = np.array(list(set(all_idx) - set(val)))

        self.validation_idx = np.transpose(self.train_idx[val])
        self.train_idx = np.transpose(self.train_idx[tr])

        if test_sample_file is not None:
            self.test_loc = test_sample_file
            self.test_idx = np.transpose(file_to_np(self.test_loc))
        else:
            self.test_loc = None

    def __len__(self):
        return self.train_idx.shape[0]

    def __getitem__(self, idx):
        if self.test_loc is not None:
            return SubsetRandomSampler(self.train_idx[idx]),\
                   SubsetRandomSampler(self.validation_idx[idx]),\
                   SubsetRandomSampler(self.test_idx[idx])
        else:
            return SubsetRandomSampler(self.train_idx[idx]),\
                   SubsetRandomSampler(self.validation_idx[idx])

"""
Operations to generate embeddings.
"""

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"

import numpy as np
import torch
from gensim.models import KeyedVectors

from .mathops import complex_multiplication, complex_division, circular_conv
from .mathops import get_appx_inv, get_inv, complexMagProj, normalize
from .mathops import npcomplexMagProj

"""
Load Pretrained Label Embeddings.
"""
def load_embeddings(save_loc, vocab_size):
    fname = save_loc + "-complex.bin"
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    rand_vec_cnt = 0
    vectors = [] # positions in vector space.
    for i in range(0, vocab_size):
        if str(i) in model.wv.vocab:
            vectors.append(model.wv[str(i)])
        else:
            # NOTE: When a label is not present in training then we generate a
            #       default vector and add it to the label vector matrix.
            #  As SPN select the label based on the index it remains consistent while training.
            rand_vec_cnt += 1
            vectors.append(gen_rand_vec(model.vector_size))

    # Add Padding idx.
    print("Vocabulary Size: {}".format(vocab_size))
    print("Number of Random vectors generated: {}".format(rand_vec_cnt))
    vectors.append(gen_rand_vec(model.vector_size))
    vectors = torch.from_numpy(np.array(vectors, dtype=np.float32))
    return vectors

"""
NumPY operations for embeddings.
"""
def generate_vectors(num_vectors, dims):
    """
    Generate n vectors of size dims that are orthogonal to each other.
    """
    if num_vectors > dims:
        raise ValueError("num_vectors cannot be greater than dims!")

    # Intializing class vectors.
    vecs = torch.randn(dims, num_vectors, dtype=torch.float)

    # Using QR decomposition to get orthogonal vectors.
    vecs, _ = torch.qr(vecs)
    vecs = vecs.t()
    vecs = vecs / torch.norm(vecs, dim=-1, keepdim=True)
    return vecs


def gen_rand_vec(dims):
    """
    Generate a random vector of size dims.
    """
    return npcomplexMagProj(np.random.normal(0, 1. / dims, size=(dims)))


"""
Torch functions.
"""
def get_vectors(num_vectors, dims, ortho=False):
    if ortho:
        vectors = generate_vectors(num_vectors, dims)
        return complexMagProj(vectors)
    else:
        vectors = [gen_rand_vec(dims) for i in range(num_vectors)]
        return torch.from_numpy(np.array(vectors, dtype=np.float32))

def get_static_embedding(seeds, dims):
    vec = []
    for s in seeds:
        torch.manual_seed(s)
        vec.append(torch.randn((1, dims), dtype=torch.float))

    return torch.cat(vec, dim=0)

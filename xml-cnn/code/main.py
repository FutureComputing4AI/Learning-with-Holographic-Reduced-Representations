from header import *
from cnn_train import *
from cnn_test import *
import pdb
from lib.metrics import compute_inv_propensity
from lib.utils import print_command_arguments

# ------------------------ Params -------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--zd', dest='Z_dim', type=int, default=100, help='Latent layer dimension')
parser.add_argument('--mb', dest='mb_size', type=int, default=20, help='Size of minibatch, changing might result in latent layer variance overflow')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--p', dest='plot_flg', type=int, default=0, help='1 to plot, 0 to not plot')
# parser.add_argument('--e', dest='num_epochs', type=int, default=50, help='step for displaying loss')
parser.add_argument('--e', dest='num_epochs', type=int, default=2, help='step for displaying loss')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('-a', type=float, default=0.55,
                    help='Inverse propensity value A (Default: 0.55).')
parser.add_argument('-b', type=float, default=1.5,
                    help='Inverse propensity value A (Default: 1.5).')

parser.add_argument('--d', dest='disp_flg', type=int, default=0, help='display graphs')
parser.add_argument('--sve', dest='save', type=int, default=1, help='save models or not')
parser.add_argument('--ss', dest='save_step', type=int, default=10, help='gap between model saves')
parser.add_argument('--mn', dest='model_name', type=str, default='', help='model name')
parser.add_argument('--tr', dest='training', type=int, default=1, help='model name')
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model name')
parser.add_argument('--ds', dest='data_set', type=str, default="rcv", help='dataset name')

parser.add_argument('--pp', dest='pp_flg', type=int, default=0, help='1 is for min-max pp, 2 is for gaussian pp, 0 for none')
parser.add_argument('--loss', dest='loss_type', type=str, default="BCELoss", help='Loss')

parser.add_argument('--hidden_dims', type=int, default=512, help='hidden layer dimension')
# parser.add_argument('--hidden_dims', type=int, default=1024, help='hidden layer dimension') # Amazon670K
parser.add_argument('--sequence_length',help='max sequence length of a document', type=int,default=500)
parser.add_argument('--embedding_dim', help='dimension of word embedding representation', type=int, default=300)
parser.add_argument('--model_variation', help='model variation: CNN-rand or CNN-pretrain', type=str, default='pretrain')
parser.add_argument('--pretrain_type', help='pretrain model: GoogleNews or glove', type=str, default='glove')
parser.add_argument('--vocab_size', help='size of vocabulary keeping the most frequent words', type=int, default=30000)
parser.add_argument('--drop_prob', help='Dropout probability', type=int, default=.3)
parser.add_argument('--load_data', help='Load Data or not', type=int, default=0)
parser.add_argument('--mg', dest='multi_gpu', type=int, default=0, help='1 for 2 gpus and 0 for normal')
parser.add_argument('--filter_sizes', help='number of filter sizes (could be a list of integer)', type=int, default=[2, 4, 8], nargs='+')

# Large Datasets.
parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=32)
parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=32)

# Small Datasets.
# parser.add_argument('--num_filters', help='number of filters (i.e. kernels) in CNN model', type=int, default=128)
# parser.add_argument('--pooling_units', help='number of pooling units in 1D pooling layer', type=int, default=128)

parser.add_argument('--pooling_type', help='max or average', type=str, default='max')
parser.add_argument('--model_type', help='glove or GoogleNews', type=str, default='glove')
parser.add_argument('--num_features', help='50, 100, 200, 300', type=int, default=300)
parser.add_argument('--dropouts', help='0 for not using, 1 for using', type=int, default=0)
parser.add_argument('--clip', help='gradient clipping', type=float, default=1000)
# parser.add_argument('--clip', help='gradient clipping', type=float, default=2.0)
parser.add_argument('--dataset_gpu', help='load dataset in full to gpu', type=int, default=1)
parser.add_argument('--dp', dest='dataparallel', help='to train on multiple GPUs or not', type=bool, default=False)

# HRR specific arguments.
parser.add_argument('--hrr_labels', action='store_true', default=False, help='Use HRR Labels.')
parser.add_argument('--hrr_dim', type=int, default=400, help='HRR Label Dimension.')
parser.add_argument('--no-grad', action='store_true', default=False,
                    help='Update Label vectors.')
parser.add_argument('--without-negative', action='store_true', default=False,
                    help='disable negative loss.')

params = parser.parse_args()
print_command_arguments(params)

if(len(params.model_name)==0):
    params.model_name = "Gen_data_CNN_Z_dim-{}_mb_size-{}_hidden_dims-{}_preproc-{}_loss-{}_sequence_length-{}_embedding_dim-{}_params.vocab_size={}".format(params.Z_dim, params.mb_size, params.hidden_dims, params.pp_flg, params.loss_type, params.sequence_length, params.embedding_dim, params.vocab_size)

print('Saving Model to: ' + params.model_name)

# Begin.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = not params.no_cuda and torch.cuda.is_available()
torch.manual_seed(params.seed)
np.random.seed(params.seed)
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True,
              'batch_size': params.batch_size, 'shuffle': True}
else:
    kwargs = {'drop_last': True, 'num_workers': 8,
              'batch_size': params.batch_size, 'shuffle': True}

# ------------------ data ----------------------------------------------
params.data_path = '../datasets/' + params.data_set

# Create training and test data loaders.
train_dataset = XMLDataset(params)
print("-----------Training Dataset Statistics-----------")
print("Features: {}".format(train_dataset.features.shape))
print("Labels: {}".format(train_dataset.labels.shape))

# Compute Propensity Scores.
inv_propen = compute_inv_propensity(train_dataset.labels, A=params.a, B=params.b)

# Create dataloader.
train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)

test_dataset = XMLDataset(params, train=False)
print("-----------Testing Dataset Statistics------------")
print("Features: {}".format(test_dataset.features.shape))
print("Labels: {}".format(test_dataset.labels.shape))
test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

params = update_params(params)
# -----------------------  Loss ------------------------------------------------
if not params.hrr_labels:
    params.loss_fn = torch.nn.BCELoss(size_average=False)

# -------------------------- Params --------------------------------------------
if params.model_variation == 'pretrain':
    embedding_weights = load_word2vec(params)
else:
    embedding_weights = None

if torch.cuda.is_available():
    params.dtype = torch.cuda.FloatTensor
else:
    params.dtype = torch.FloatTensor


if(params.training):
    train(train_loader, test_loader, embedding_weights, params, device,
          propensity=inv_propen)
else:
	test_class(test_loader, params, model=model, device=device, verbose=False,
               propensity=inv_propen)

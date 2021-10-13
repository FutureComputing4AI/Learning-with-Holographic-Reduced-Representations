from header import *
from cnn_encoder import cnn_encoder
from lib.utils import Measure

class xmlCNN(nn.Module):
    def __init__(self, params, embedding_weights):
        super(xmlCNN, self).__init__()
        self.params = params
        self.embedding_layer = embedding_layer(params, embedding_weights)
        self.classifier = cnn_encoder(params)

        if params.hrr_labels:
            self.loss = self.classifier.spp_loss
        else:
            self.loss = self.params.loss_fn

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

    def forward(self, batch_x, batch_y):
        # ----------- Encode (X, Y) --------------------------------------------
        self.time['train_forward_pass'].start()
        e_emb = self.embedding_layer.forward(batch_x)
        Y = self.classifier.forward(e_emb)
        self.time['train_forward_pass'].end()

        # Compute time for loss.
        self.time['train_loss'].start()
        loss = self.loss(Y, batch_y)
        self.time['train_loss'].end()

        if(loss < 0):
            print(cross_entropy)
            print(Y[0:100])
            print(batch_y[0:100])
            sys.exit()

        return loss.view(-1,1), Y

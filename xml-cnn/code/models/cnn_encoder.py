from header import *
from lib.embeddings import get_vectors
from lib.mathops import get_appx_inv, circular_conv, complexMagProj

def out_size(l_in, kernel_size, padding=0, dilation=1, stride=1):
    a = l_in + 2*padding - dilation*(kernel_size - 1) - 1
    b = int(a/stride)
    return b + 1

class cnn_encoder(torch.nn.Module):

    def __init__(self, params):
        super(cnn_encoder, self).__init__()
        self.params = params
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        fin_l_out_size = 0

        if(params.dropouts):
            self.drp = nn.Dropout(p=.25)
            self.drp5 = nn.Dropout(p=.5)

        for fsz in params.filter_sizes:
            l_out_size = out_size(params.sequence_length, fsz, stride=2)
            pool_size = l_out_size // params.pooling_units
            l_conv = nn.Conv1d(params.embedding_dim, params.num_filters, fsz, stride=2)
            torch.nn.init.xavier_uniform_(l_conv.weight)
            if params.pooling_type == 'average':
                l_pool = nn.AvgPool1d(pool_size, stride=None, count_include_pad=True)
                pool_out_size = (int((l_out_size - pool_size)/pool_size) + 1)*params.num_filters
            elif params.pooling_type == 'max':
                l_pool = nn.MaxPool1d(2, stride=1)
                pool_out_size = (int(l_out_size*params.num_filters - 2) + 1)
            fin_l_out_size += pool_out_size

            self.conv_layers.append(l_conv)
            self.pool_layers.append(l_pool)

        self.fc_layer_1 = nn.Linear(fin_l_out_size, params.hidden_dims)
        torch.nn.init.xavier_uniform_(self.fc_layer_1.weight)

        # NOTE: Comment out fc2 and fc3 for Amazon670K
        self.fc_layer_2 = nn.Linear(params.hidden_dims, params.hidden_dims)
        torch.nn.init.xavier_uniform_(self.fc_layer_2.weight)

        self.fc_layer_3 = nn.Linear(params.hidden_dims, params.hidden_dims)
        torch.nn.init.xavier_uniform_(self.fc_layer_3.weight)
        ###

        if params.hrr_labels:
            self.out_layer = nn.Linear(params.hidden_dims, params.hrr_dim)
            self.create_label_embedding() # Create the labels.
        else:
            self.out_layer = nn.Linear(params.hidden_dims, params.y_dim)

        torch.nn.init.xavier_uniform_(self.out_layer.weight)


    def create_label_embedding(self):
        # Class labels. # +1 for the END of LIST Label.
        self._class_vectors = get_vectors(self.params.y_dim + 1, self.params.hrr_dim)

        # Initialize embedding layer.
        self.class_vec = nn.Embedding(self.params.y_dim + 1, self.params.hrr_dim)
        self.class_vec.load_state_dict({'weight': self._class_vectors})
        self.class_vec.weight.requires_grad = False

        # Initialize weights vector.
        weights = torch.ones((self.params.y_dim + 1, 1), dtype=torch.int8)
        weights[self.params.y_dim] = 0 # Padding vector is made 0.
        self.class_weights = nn.Embedding(self.params.y_dim + 1, 1)
        self.class_weights.load_state_dict({'weight': weights})
        self.class_weights.weight.requires_grad = False

        # P & N vectors.
        p_n_vec = get_vectors(2, self.params.hrr_dim, ortho=True)
        if self.params.no_grad:
            print("P & N vectors WILL NOT be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=False)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=False)
        else:
            print("P & N vectors WILL be updated while training...")
            self.p = nn.Parameter(p_n_vec[0], requires_grad=True)
            self.n = nn.Parameter(p_n_vec[1], requires_grad=True)


    def inference(self, s, batch_size, positive=True):
        #(batch, dims)
        if positive:
            vec = self.p.unsqueeze(0).expand(batch_size, self.params.hrr_dim)
        else:
            vec = self.n.unsqueeze(0).expand(batch_size, self.params.hrr_dim)

        # vec = complexMagProj(vec)
        inv_vec = get_appx_inv(vec)
        y = circular_conv(inv_vec, s) #(batch, dims)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y

    def spp_loss(self, s, target):
        """
        Train with SPP.
        """
        pos_classes = self.class_vec(target)   #(batch, no_label, dims)
        pos_classes = pos_classes * self.class_weights(target)

        # Normalize the class vectors.
        # tgt_shape = pos_classes.shape
        # pos_classes = torch.reshape(pos_classes, (tgt_shape[0] * tgt_shape[1],
        #                                           tgt_shape[2]))
        # pos_classes = torch.reshape(complexMagProj(pos_classes), (tgt_shape[0], tgt_shape[1],
        #                                            tgt_shape[2]))

        # Remove the padding idx vectors.
        # pos_classes = pos_classes.to(device)

        # Positive prediction loss
        convolve = self.inference(s, target.size(0))
        cosine = torch.matmul(pos_classes, convolve.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        J_p = torch.mean(torch.sum(1 - torch.abs(cosine), dim=-1))

        # Negative prediction loss.
        J_n = 0.0
        if self.params.without_negative is False:
            convolve = self.inference(s, target.size(0), positive=False)
            cosine = torch.matmul(pos_classes, convolve.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
            J_n = torch.mean(torch.sum(torch.abs(cosine), dim=-1))

        # Total Loss.
        loss = J_n + J_p
        return loss


    def forward(self, inputs):
        #o0 = self.drp(self.bn_1(inputs)).permute(0,2,1)
        o0 = inputs.permute(0,2,1)# self.bn_1(inputs.permute(0,2,1))
        if(self.params.dropouts):
            o0 = self.drp(o0)
        conv_out = []

        for i in range(len(self.params.filter_sizes)):
            o = self.conv_layers[i](o0)
            o = o.view(o.shape[0], 1, o.shape[1] * o.shape[2])
            o = self.pool_layers[i](o)
            o = nn.functional.relu(o)
            o = o.view(o.shape[0],-1)
            conv_out.append(o)
            del o
        if len(self.params.filter_sizes)>1:
            o = torch.cat(conv_out,1)
        else:
            o = conv_out[0]

        # Additional fully connected layers added to the model.
        o = self.fc_layer_1(o)
        o = nn.functional.relu(o)

        # NOTE: Comment out fc2 and fc3 for Amazon670K
        o = self.fc_layer_2(o)
        o = nn.functional.relu(o)

        o = self.fc_layer_3(o)
        o = nn.functional.relu(o)
        ###

        if(self.params.dropouts):
            o = self.drp5(o)
        o = self.out_layer(o)

        if not self.params.hrr_labels:
            o = torch.sigmoid(o)

        return o

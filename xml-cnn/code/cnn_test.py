from header import *
from collections import OrderedDict
from sklearn.metrics import log_loss
from lib.metrics import compute_prop_metrics, display_metrics

def test_class(test_loader, params, device, model=None, embedding_weights=None,
               verbose=True, propensity=None, topk=5):
    if(model == None):
        if(embedding_weights is None):
            print("Error: Embedding weights needed!")
            exit()
        else:
            model = xmlCNN(params, embedding_weights)
            model = load_model(model, params.load_model)

    if(torch.cuda.is_available()):
        params.dtype_f = torch.cuda.FloatTensor
        params.dtype_i = torch.cuda.LongTensor
        model = model.cuda()
    else:
        params.dtype_f = torch.FloatTensor
        params.dtype_i = torch.LongTensor

    # Testing data.
    loss = 0.0; prec = 0.0; num_batch = 0.0; all_acc = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        # Load Data.
        batch_x = batch_x.type(torch.LongTensor).to(device)
        batch_y = batch_y.type(torch.FloatTensor).to(device)

        model.time['test_forward_pass'].start()
        e_emb = model.embedding_layer.forward(batch_x)
        s = model.classifier(e_emb)
        model.time['test_forward_pass'].end()

        model.time['inference'].start()
        if params.hrr_labels:
            batch_size = batch_y.size()[0]
            combined_y = model.classifier.inference(s, batch_size)
            y_pred = torch.abs(torch.mm(combined_y, model.classifier.class_vec.weight.t())).cpu().data[:, :batch_y.shape[1]].numpy()
        else:
            y_pred = s.cpu().data.numpy()
        model.time['inference'].end() # Measure forward pass during inference.

        # Measure.
        y_cpu = batch_y.cpu().data.numpy()
        loss += log_loss(y_cpu, y_pred)
        acc = compute_prop_metrics(sparse.csr_matrix(y_cpu),
                                   sparse.csr_matrix(y_pred), propensity,
                                   topk=topk)
        all_acc.append(acc)
        num_batch += 1

    loss /= num_batch
    print('Test Loss; Cross Entropy {};'.format(loss))
    display_metrics(all_acc)
    return loss

from header import *
from cnn_test import *

# ---------------------------------------------------------------------------------

def train(train_loader, test_loader, embedding_weights, params, device,
		  propensity=None):
	loss_best = float('Inf')
	bestTotalLoss = float('Inf')
	best_test_loss = float("inf")
	max_grad = 0
	num_mb = np.ceil(params.N/params.mb_size)
	model = xmlCNN(params, embedding_weights)
	if(torch.cuda.is_available()):
		print("--------------- Using GPU! ---------")
		model.params.dtype_f = torch.cuda.FloatTensor
		model.params.dtype_i = torch.cuda.LongTensor
		model = model.to(device)
	else:
		model.params.dtype_f = torch.FloatTensor
		model.params.dtype_i = torch.LongTensor
		print("=============== Using CPU =========")

	optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=params.lr)
	print(model);print("%"*100)

	if params.dataparallel:
		model = nn.DataParallel(model)

	if(len(params.load_model)):
		params.model_name = params.load_model
		print(params.load_model)
		model, optimizer, init = load_model(model, params.load_model, optimizer=optimizer)
	else:
		init = 0

	# =============================== TRAINING =================================
	for epoch in range(init, params.num_epochs):
		totalLoss = 0.0
		model.time['train'].start()
		model.time['data_load'].start()
		for i, (batch_x, batch_y) in enumerate(train_loader):
			model.time['data_load'].end()
			model.train()
			optimizer.zero_grad()

			# Load data to GPU.
			batch_x = batch_x.type(torch.LongTensor).to(device)
			if params.hrr_labels:
				batch_y = batch_y.type(torch.LongTensor).to(device)
			else:
				batch_y = batch_y.type(torch.FloatTensor).to(device)

			# Model forward.
			loss, output = model.forward(batch_x, batch_y)

			# ------------------------------------------------------------------
			loss = loss.mean().squeeze()
			totalLoss += loss.data

			# NOTE: This block is not part of training.
			model.time['train'].end()
			if i % int(num_mb/12) == 0:
				print('Iter-{}; Loss: {:.4}; best_loss: {:.4}; max_grad: {}:'.format(i, loss.data, loss_best, max_grad))
				if not os.path.exists('../saved_models/' + params.model_name ):
					os.makedirs('../saved_models/' + params.model_name)
				save_model(model, optimizer, epoch, params.model_name + "/model_best_batch")
				if(loss<loss_best):
					loss_best = loss.data
			model.time['train'].start()

			# ------------------------ Propagate loss --------------------------
			model.time['optimization'].start()
			loss.backward()
			loss = loss.data
			model.time['optimization'].end()

			torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip)

			model.time['optimization'].start()
			optimizer.step()
			model.time['optimization'].end()

			model.time['data_load'].start()

		model.time['data_load'].end()
		model.time['train'].end()

		if(totalLoss < bestTotalLoss):
			bestTotalLoss = totalLoss
			if not os.path.exists('../saved_models/' + params.model_name ):
				os.makedirs('../saved_models/' + params.model_name)
			save_model(model, optimizer, epoch, params.model_name + "/model_best_epoch")

		print('End-of-Epoch: {} Loss: {:.4}; best_loss: {:.4};'.format(epoch, totalLoss, bestTotalLoss))

		model.eval()
		test_ce_loss = test_class(test_loader, params, model=model,
								  device=device, verbose=False, propensity=propensity)

		if(test_ce_loss < best_test_loss):
			best_test_loss = test_ce_loss
			print("This loss is better than previous recorded CE loss:- {}".format(best_test_loss))
			if not os.path.exists('../saved_models/' + params.model_name ):
				os.makedirs('../saved_models/' + params.model_name)
			save_model(model, optimizer, epoch, params.model_name + "/model_best_test")

		if epoch % params.save_step == 0:
			save_model(model, optimizer, epoch, params.model_name + "/model_" + str(epoch))

	print("-----------Running Measurements---------")
	print("Training time / epoch: {:0.3f}".format(model.time['train'].get_elapsed_time() / params.num_epochs))
	print("Data Loader time / epoch: {:0.3f}".format(model.time['data_load'].get_elapsed_time() / params.num_epochs))
	print("Train Forward Pass time / epoch: {:0.3f}".format(model.time['train_forward_pass'].get_elapsed_time() / params.num_epochs))
	print("Train Loss time / epoch: {:0.3f}".format(model.time['train_loss'].get_elapsed_time() / params.num_epochs))
	print("Optimization time / epoch: {:0.3f}".format(model.time['optimization'].get_elapsed_time() / params.num_epochs))
	print("Test Forward Pass time / epoch: {:0.3f}".format(model.time['test_forward_pass'].get_elapsed_time() / params.num_epochs))
	print("Inference time / epoch: {:0.3f}".format(model.time['inference'].get_elapsed_time() / params.num_epochs))

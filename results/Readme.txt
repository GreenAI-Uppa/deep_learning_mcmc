XP June 2021

Net1 = cnn
-conv1 = 32 filters of size 11x11
-one fc of size 8*8*32*10 (stride = 3)
-optim = {grad,mcmc} 

results_cnn_optim_proba contains xp of 100 epochs with minibatch size 64 (grad) and 100 epochs with 800 MCMC mooves and full batch (batchsize=50000) and evaluation for sparse networks of size 90 -> 10% (quantile threshold)
results_cnn_optim_sparsindex_proba same as above but with model=model_sparse at each epoch

-----------------------------------------------------------------------------------------------

Nets2
-conv1 = 64 filters of size 11x11
-one fc of size 8*8*64*10 (stride = 3)
-optim = {grad,mcmc} 

results_cnn_optim_proba contains xp of 100 epochs with minibatch size 64 (grad) and 100 epochs with 800 MCMC mooves and full batch (batchsize=50000) and evaluation for sparse networks of size 90 -> 10% (quantile threshold)
results_cnn_optim_sparsindex_proba same as above but with model=model_sparse at each epoch

grad ok
mcmc en cours
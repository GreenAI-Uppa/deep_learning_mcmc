# Deep learning by MCMC


This repo is a work in progress to provide the code to optimize a linear model with Markov Chain Monte Carlo Optimization


 Installation

```
pip install torch 
```

Train the model on cifar-10 with mcmc (run --help for additional arguments)

```
python exp_mini_batch.py --exp_name mcmc_full_batch
```
This command will generate a json file containing the loss and the accuracy for the different epochs. 
To visualise the curves, use: 

```
python plot_curve.py exp_name_1000000_999.json
```

Train the model with **gradient descent** for comparison purposes

```
python exp_mini_batch.py --batch_size 64 --use_gradient
```

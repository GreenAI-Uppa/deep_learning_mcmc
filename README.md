# Deep learning by MCMC


This repo is a work in progress to provide the code to optimize neural networks with Markov Chain Monte Carlo Optimization


## installation

```
pip install torch 
pip install -r requirements.txt
python setup.py install
```

## usage

Train the model on cifar-10 with mcmc (run --help for additional arguments)

```
python run_conv.py --config_file configs/config_mcmc_conv_real.json --data_folder /home/paul/data/pytorch_cifar10/ --verbose
```
This command will generate a json file containing the loss and the accuracy for the different epochs. 

## configuration

The json configuration file is organised around lists where the ith element is the configuration of the ith layer 


# Deep learning by MCMC - PAC-Bayesian Sparse and Binary CNNs


This branch is an illustration of the paper "PAC-Bayesian Sparse and Binary CNNs" submitted to the *Journal of Machine Learning Research*.

It is a small part of our research activity dedicated to the optimization of neural networks with Markov Chain Monte Carlo.


## installation

```
pip install torch 
pip install -r requirements.txt
python setup.py install
```

## usage

Test a learning process of a two-layer CNN (one convolution layer of 11*11*3 filters and one dense layer) on cifar-10 with mcmc (run --help for additional arguments)

```
python run_jmlr.py --config_file configs/config_mcmc_conv_cnn.json --data_folder /data/pytorch_cifar10/ --verbose
```
This command will generate a json file containing the loss and the accuracy for the different epochs, as well as different logs on the training process and the robustness to pruning.

## configuration

Three files are available corresponding to three different optimizer: vanilla gradient descent, binary connect (to train a binary neural network with gradient descent and STE estimation) and the MCMC optimizer.

The json configuration of the MCMC optimizer is organised as follows:

```json
{
        "batch_size": 50000,
        "exp_name": "jmlr_mcmc",
        "architecture": {
                "boolean_flags": [0,0],
                "activations" : ["Sigmoid", "Softmax"],
                "nb_filters" : 64
        },
        "variance_init": 0.00000001,
        "optimizer": {
                "name": "mcmc",
                "selector" : {
                    "name": "Selector",
                    "layer_conf" : [
                       {"layer_distr" :0.5, "get_idx":"get_idces_uniform_conv", "get_idx_param":200},
                       {"layer_distr" :0.5, "get_idx":"get_idces_uniform_linear", "get_idx_param":200}]
                },
                "samplers" : [
                    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000},
                    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000}
                    ],
                "iters":600,
                "progressive_pruning": 0,
                "pruning_level_start":0.1,
                "pruning_schedule":[2,0.1]
        },
        "naive_pruning_evaluation": 0,
	"measure_power":0
}
```

The configuration file has 7 keys, described below:
- "batch_size": size of a minibatch in the MCMC optimization. We strongly encourage to let batch_size = 50000 (the full training set) as mentionned in the paper. Reducing the batch size allows to have "iters" moves for each minibatch,
- "exp_name": name of the json generated for your result,
- "architecture": the main hyper-parameters of the network:
  - "boolean_flags": list of 0 (continuous layer) or 1 (binary layer). Default is [0,0] for no binarization.
  - "activations": activations used after the convolution layer, and the dense layer.
  - "nb_filters": number of filters of the convolution layer.
- "variance_init": variance of the prior distribution at initialization of the continuous layers,
- "optimizer": the main hyper-parameters of the MCMC random walk:
  - "name": "mcmc" or "gradient".
  - "selector": definition of the neighborhood construction in the random walk: "layer_distr" is the probability to choose a layer, and "get_idx" is how you choose a neighborhood ("get_idces_uniform_conv" to choose at random "get_idx_param" values, or "get_idces_filter_conv" to select one filter at random, ie 11*11*3 values).
  - "sampler": proposal distribution, given a layer and a neighborhood ("Student" for a continuous layer, "BinaySampler" ie Rademacher variable with proba 1/2 for a binary layer).
  - "iters": numbers of MCMC iterations (the same for SGD).
  - "progressive_pruning": 0 (no progressive pruning) or 1 (progressive pruning during the training process, Section 5.2).
  - "pruning_level_start": level of pruning at the first iteration if progressive_pruning is 1.
  - "pruning_schedule": if progressive_pruning is 1, [int,level] where int is the period to prune the network (condition iters%int in the loop) and level is the amount of pruning level to add at each period int.
- "naive_pruning_evaluation: if 1, run_jmlr.py computes and returns test accuracies for different level of naive pruning, after training (Section 5.1 of the paper).
- "measure_power": if 1, strongly depends on our repo [AIPowerMeter](https://github.com/GreenAI-Uppa/AIPowerMeter) and return the power draws of the training process based on rapl and nvidia-smi.

The json configuration of the standard SGD is more classical, since the optimizer has just a learning rate and a mini-batch size. Note that in the progressive pruning setting, we use Mozer pruning (see the reference in the paper) with particular schedule described above.

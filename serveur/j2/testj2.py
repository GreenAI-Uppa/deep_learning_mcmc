# test j2 with only one linear dense 
import asyncio
import time
import sys

import torch

from deep_learning_mcmc import nets, optimizers, selector, stats

CHANNELS = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss()

params = {
        "batch_size": 128, 
        "epochs": 10,
        "exp_name": "mozer",
        "architecture": {
                "boolean_flags": [0, 0],
                "activations" : ["Softmax"],
                "nb_filters" : 32
        },
        "variance_init": 0.00000001,
        "optimizer": {
                "name": "mcmc",
                "pruning_level":0,
                "selector" : {
                    "name": "Selector",
                    "layer_conf" : [
                       {"layer_distr" :1, "get_idx":"get_idces_uniform_linear","get_idx_param":363}]
                },
                "samplers" : [
                    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000}
                ],
                "iter_mcmc" : 200
        },
	"dataset": "CIFAR",
	"measure_power": 0
}

sp = [
    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000}
]


def init_model():
    '''
    On passe ici à une conv sans stride avec un padding de 5
    TODO: ajouter des paramêtres (kwargs par ex) pour la gestion du pading / stride / taille filtre 
    '''
    return nets.LinearNet(32, activations=["Softmax"])


async def trainer(reading_queue=None):
    '''trainer linear classifier for cifar10 with mcmc'''
    print("Init started\n")
    model = init_model()
    model = model.to(device)
    print("Init finished\n")
    
    config = {'name': params['optimizer']['selector']['name'], 'layer_conf':[]}
    for layer_conf in params['optimizer']['selector']['layer_conf']:
        layer_distr = layer_conf['layer_distr']
        if 'get_idx_param' in layer_conf:
            get_idx = getattr(selector, layer_conf['get_idx'])(layer_conf['get_idx_param'])
        else:
            get_idx = getattr(selector, layer_conf['get_idx'])()
        config['layer_conf'].append({'layer_distr': layer_distr, 'get_idx': get_idx})
    
    samplers = stats.build_samplers(sp) # tire un échantillons qui suit une loi de student selon les paramètres donnés
    select =  selector.build_selector(config) # renvoie n poids du layer tirés aléatoirement
    optimizer = optimizers.MCMCOptimizer(
        sampler=samplers,
        iter_mcmc=200,
        prior=samplers,
        selector=select,
        pruning_level=0,
        reading_queue=reading_queue,
    )
    print("Start training\n")
    
    _ = await optimizer.train_1_epoch(model, loss_fn, verbose=False)

def init_data():
    training_data = datasets.CIFAR10(root=".",
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.CIFAR10(root=".",
        train=False,
        download=True,
        transform=ToTensor())
    return DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=16), DataLoader(test_data, batch_size=50000, num_workers=16)

async def reading_client(reading_queue=None):
    # feed queue with random cifar 10 
    
    return
    
    
    
async def main():
    '''final output of mcmc modeling grappe'''
    reading_queue = asyncio.Queue()
    
    # faire une convolution => output -> linear
    
    client = asyncio.create_task(reading_client(reading_queue=reading_queue))
    runner = asyncio.create_task(trainer(reading_queue=reading_queue))
    
    
    
if __name__ == "__main__":
    asyncio.run(main())
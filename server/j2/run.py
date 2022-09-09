import asyncio
import time
import sys

import torch

from deep_learning_mcmc import nets, optimizers, selector, stats, connexion

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

def set_config():
    '''set config for mcmc'''
    config = {'name': params['optimizer']['selector']['name'], 'layer_conf':[]}
    for layer_conf in params['optimizer']['selector']['layer_conf']:
        layer_distr = layer_conf['layer_distr']
        if 'get_idx_param' in layer_conf:
            get_idx = getattr(selector, layer_conf['get_idx'])(layer_conf['get_idx_param'])
        else:
            get_idx = getattr(selector, layer_conf['get_idx'])()
        config['layer_conf'].append({'layer_distr': layer_distr, 'get_idx': get_idx})
    
    return config

async def trainer(reading_queue):
    '''trainer linear classifier for cifar10 with mcmc'''
    print("Init started\n")
    model = init_model()
    model = model.to(device)
    print("Init finished\n")
    
    config = set_config()
    
    samplers = stats.build_samplers(sp) # tire un échantillons qui suit une loi de student selon les paramètres donnés
    select =  selector.build_selector(config) # renvoie n poids du layer tirés aléatoirement
    optimizer = optimizer = optimizers.AsyncMcmcOptimizer(
        sampler=samplers,
        iter_mcmc=200,
        prior=samplers,
        selector=select,
        pruning_level=0,
        reading_queue=reading_queue, # voir pour la lecture des données sur la manière de s'y prendre
        log_path="/home/gdev/tmp/mcmc/data"
    )
    print("Start training\n")
    
    _ = await optimizer.train_1_epoch(model, loss_fn, verbose=False)


async def main():
    '''final output of mcmc modeling grappe'''
    reading_queue = asyncio.Queue()
    with open("/home/gdev/tmp/mcmc/latency", "w") as latency:
        latency.write("lecture;envoie\n")
        cl = connexion.Client(local_name="j2", connect_to=('10.0.12.90', 5000), reading_queue=reading_queue, log_latency=latency, verbose=True)

        reader = asyncio.create_task(cl.start())
        runner = asyncio.create_task(trainer(reading_queue=reading_queue))
        await reader
        await runner
    print('fin d entrainement')
    
if __name__ == "__main__":
    asyncio.run(main())
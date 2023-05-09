import asyncio
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from deep_learning_mcmc import nets, optimizers, selector, stats, connexion

# PATH_LOG = "/home/gdev/tmp/mcmc"
PATH_LOG = "/home/mfrancois/Documents/mas/p8"
CHANNELS = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss()
iter_mcmc = 10

params = {
        "batch_size": 50000, 
        "epochs": 10,
        "exp_name": "mozer",
        "architecture": {
                "boolean_flags": [0, 0],
                "activations" : ["ReLU", "Softmax"],
                "nb_filters" : 32
        },
        "variance_init": 0.00000001,
        "optimizer": {
                "name": "mcmc",
                "pruning_level":0,
                "selector" : {
                    "name": "Selector",
                    "layer_conf" : [
                       {"layer_distr" :0.5, "get_idx":"get_idces_filter_conv"},
                       {"layer_distr" :0.5, "get_idx":"get_idces_uniform_linear","get_idx_param":363}]
                },
                "samplers" : [
                    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000},
                    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000}
                ],
                "iter_mcmc" : 200
        },
	"dataset": "CIFAR",
	"measure_power": 0
}

sp = [
    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000},
    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000}
]



def init_data():
    test_data = datasets.CIFAR10(root=".",
        train=False,
        download=True,
        transform=ToTensor())
    return DataLoader(test_data, batch_size=50000, num_workers=16)        

def init_model():
    '''
    On passe ici à une conv sans stride avec un padding de 5
    '''
    return nets.ConvNet(32, CHANNELS, binary_flags=[False, False],  activations=["ReLU", "Softmax"], pruning_level = 0, padding=5, stride=1)

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

async def trainer(reading_queue, sending_queue):
    '''create and train model'''
    print("Init started\n")
    # test_dataloader = init_data()
    model = init_model()
    model = model.to(device)
    print("Init finished\n")
    
    config = set_config()
    
    samplers = stats.build_samplers(sp) # tire un échantillons qui suit une loi de student selon les paramètres donnés
    select =  selector.build_selector(config) # renvoie n poids du layer tirés aléatoirement
    optimizer = optimizers.AsyncMcmcOptimizer(
        sampler=samplers,
        iter_mcmc=iter_mcmc,
        prior=samplers,
        selector=select,
        pruning_level=0,
        sending_queue=sending_queue,
        reading_queue=reading_queue, # voir pour la lecture des données sur la manière de s'y prendre
        log_path=f"{PATH_LOG}/data"
    )
    print("Start training\n")
    
    _ = await optimizer.train_1_epoch(model, loss_fn, verbose=False, activation_layer="conv1")
    optimizer.doc.close()



async def main():
    with open(f"{PATH_LOG}/latency", "w") as latency:
        latency.write("lecture;envoie\n")
        reading_queue = asyncio.Queue()
        sending_queue = asyncio.Queue()
        # sv = connexion.Serveur(local_name="p8", sending_queue=sending_queue, reading_queue=reading_queue, log_latency=latency, read_from="j4", send_to="p4", verbose=True)
        sv = connexion.Serveur(local_name="p8", address=("localhost",5000), sending_queue=sending_queue, reading_queue=reading_queue, log_latency=latency, read_from="j4", send_to="p4", verbose=True)

        server = asyncio.create_task(sv.start())
        runner = asyncio.create_task(trainer(reading_queue=reading_queue, sending_queue=sending_queue))

        await runner
        await sending_queue.put('__stop')
        await reading_queue.join()
        await sending_queue.join()
        server.cancel()
    print('fin')

if __name__ == "__main__":
    asyncio.run(main())

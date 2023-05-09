import asyncio
import sys
import time

import torch

from deep_learning_mcmc import nets, optimizers, selector, stats, connexion

# PATH_LOG = "/home/gdev/tmp/mcmc"
PATH_LOG = "/home/mfrancois/Documents/mas/p4"
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


def init_model():
    '''
    On passe ici à une conv sans stride avec un padding de 5
    TODO: ajouter des paramêtres (kwargs par ex) pour la gestion du pading / stride / taille filtre 
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
    print("Init started\n")
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
    
    _ = await optimizer.train_1_epoch(model=model, loss_fn=loss_fn, verbose=False, activation_layer="conv1")

            
async def main():
    reading_queue = asyncio.Queue()
    sending_queue = asyncio.Queue()
    with open(f"{PATH_LOG}/latency", "w") as latency:
        latency.write("lecture;envoie\n")
        # cl1 = connexion.Client(local_name="p4", connect_to=('10.0.12.18', 5000), reading_queue=reading_queue, log_latency=latency, verbose=True)
        # cl2 = connexion.Client(local_name="p4", connect_to=('10.0.12.90', 5000), sending_queue=sending_queue, log_latency=latency, verbose=True)

        cl1 = connexion.Client(local_name="p4", connect_to=('localhost', 5000), reading_queue=reading_queue, log_latency=latency, verbose=True)
        cl2 = connexion.Client(local_name="p4", connect_to=('localhost', 5001), sending_queue=sending_queue, log_latency=latency, verbose=True)
        
        reader = asyncio.create_task(cl1.start())
        sender = asyncio.create_task(cl2.start())
        runner = asyncio.create_task(trainer(reading_queue=reading_queue, sending_queue=sending_queue))

        await reader
        print('end reader')
        await runner
        await sending_queue.put(['__stop'])
        print("stop sent")
        await sending_queue.join()
        await sender
        print('end send')

    print(f'fin: {time.ctime()} - ts: {time.time()}s')



if __name__ == "__main__":
    asyncio.run(main())
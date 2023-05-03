# J4
import asyncio
import sys
import time 

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from deep_learning_mcmc import nets, optimizers, selector, stats, connexion

PATH_LOG = "/home/gdev/tmp/mcmc"
BATCH_SIZE = 1024
CHANNELS = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = torch.nn.CrossEntropyLoss()
n_epochs = 5

params = {
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
        },
	"dataset": "CIFAR",
	"measure_power": 0
}

sp = [
    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000},
    {"sampler": {"name" : "Student","variance":0.0000001 }, "prior" : {"name" : "Student", "variance": 0.001}, "lamb":100000}
]

# datasets
def init_data():
    '''Create train and test CIFAR10 dataset'''
    training_data = datasets.CIFAR10(root=".",
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.CIFAR10(root=".",
        train=False,
        download=True,
        transform=ToTensor())
    return DataLoader(training_data, batch_size=BATCH_SIZE), DataLoader(test_data, batch_size=BATCH_SIZE)

def init_model():
    '''init our convnet'''
    return nets.ConvNet(32, CHANNELS, binary_flags=[False, False],  activations=["ReLU", "Softmax"], pruning_level = 0, padding=0, stride=3)

def set_config():
    '''parse config var for mcmc architecture & optimizer'''
    config = {'name': params['optimizer']['selector']['name'], 'layer_conf':[]}
    for layer_conf in params['optimizer']['selector']['layer_conf']:
        layer_distr = layer_conf['layer_distr']
        if 'get_idx_param' in layer_conf:
            get_idx = getattr(selector, layer_conf['get_idx'])(layer_conf['get_idx_param'])
        else:
            get_idx = getattr(selector, layer_conf['get_idx'])()
        config['layer_conf'].append({'layer_distr': layer_distr, 'get_idx': get_idx})
    
    return config
 
async def train_model(queue):
    '''create and train model'''
    print("Init started\n")
    train_dataloader, test_dataloader = init_data()
    model = init_model()
    model = model.to(device)
    print("Init finished\n")
    
    
    config = set_config()
    samplers = stats.build_samplers(sp) 
    select =  selector.build_selector(config) 
    optimizer = optimizers.AsyncMcmcOptimizer(
        sampler=samplers,
        iter_mcmc=2,
        prior=samplers,
        selector=select,
        pruning_level=0,
        # sending_queue=queue,
        log_path=f"{PATH_LOG}/data"
    )
    print("Start training\n")
    
    for epoch in range(n_epochs):
        _ = await optimizer.train_1_epoch(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            verbose=False,
            activation_layer="conv1",
            # test_dataloader=test_dataloader
        )
        
        print(f'fin epoch n°{epoch+1}')
    optimizer.doc.close() # close log file after finishing training
    
    
async def main():
    """
    2 concurrency task are running:
    - 200 mcmc iteration training our model => feed the queue_to_send
    - client creation, connection to p8 server and consume the queue by sending to it
    """
    queue_to_send = asyncio.Queue()
    with open(f"{PATH_LOG}/latency", "w") as latency:
        latency.write("lecture;envoie\n")
        latency.write("0;0\n")
        latency.flush()
        cl = connexion.Client(local_name="j4", connect_to=("10.0.12.18", 5000), log_latency=latency, verbose=True)
        # cl = connexion.Client(local_name="j4", connect_to=("localhost", 5000), sending_queue=queue_to_send, log_latency=latency, verbose=True)

        trainer = asyncio.create_task(train_model(queue_to_send))
        client = asyncio.create_task(cl.start())
        await trainer
        print('fin trainer')
        await queue_to_send.put('__stop')
        await client
        await queue_to_send.join()
    print('fin')
    


if __name__ == "__main__":
    asyncio.run(main())

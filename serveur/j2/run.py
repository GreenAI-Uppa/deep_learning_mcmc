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


to_send = b''
request = b''

async def reading_client(reading_queue):
    """echo data to server"""
    global to_send
    global request
    
    p2 = ('10.0.12.90', 5000)
    p2 = ('localhost', 4999) # ansabere
    reader, writer = await asyncio.open_connection(*p2) # -> ouverture de la connexion avec le serveur
    print(f'client connected to {p2}')
    
    writer.write('j2'.encode())
    await writer.drain()
    
    new = True 
    i=0
    
    while ('__stop__' not in request.decode()): 
        request = await reader.read(1024) # -> va lire un packet de bytes du buffer de la socket
        i += 1
        
        # calcul du temps de reception
        if new:
            t0 = time.time()
            new = False
        to_send += request # -> stocke le message reçu dans une variable globale 
        
        if (i % 2500 == 0 ):
            full_data = to_send.decode()
            if "__fin__" in full_data:
                lecture = time.time()
                
                # decoding
                d = full_data.split('__fin__')
                data = eval(d[0])
                
                envoie = time.time()
                
                # ajout des données à la queue
                await reading_queue.put(data) 
                    
                print(
                    f'''
--------------------------------
|Nouvelle Entrée               |
|------------------------------|
|i               : {i:<12,}|
|size            : {sys.getsizeof(to_send):<12,}|
|envoie          : {round(envoie-data[2],2):<12,}|
|lecture         : {round(lecture-t0,2):<12,}|
|taille buffer   : {reading_queue.qsize():<12,}|
--------------------------------''')
                latency.write(f'{lecture-t0};{envoie-data[2]}\n')
                latency.flush()
                i = 0
                
                to_send = d[1].encode()
                new = True
                
            if "__stop__" in full_data.decode():
                print("stop__")
                break

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


async def main():
    '''final output of mcmc modeling grappe'''
    global latency
    
    reading_queue = asyncio.Queue()
    latency = open("/home/gdev/tmp/mcmc/latency", "w")
    latency.write("lecture;envoie\n")
    
    
    client = asyncio.create_task(reading_client(reading_queue=reading_queue))
    runner = asyncio.create_task(trainer(reading_queue=reading_queue))
    
    await client
    await runner
    
    latency.close()
    print('fin d entrainement')
    
if __name__ == "__main__":
    asyncio.run(main())
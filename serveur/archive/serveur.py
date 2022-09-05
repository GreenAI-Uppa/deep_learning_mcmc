# serveur
import asyncio
import torch
from torch import tensor
import sys
import time
# add security close socket ?
# 10.0.12.5  -> client j4 : envoie les données
# 10.0.12.90 -> server p2 : reçoit j4 et envoie à p4
# 10.0.12.21 -> client p4 : reçoit p2 etc...


request = b''
to_send = b''
async def handle_client(reader, writer):
    '''Lecture et écriture de données par ou pour un client'''
    global to_send
    global request
    t0 = 0
    addr = writer.get_extra_info('peername')[0] # -> ip client

    print(f"connected from {addr}")
    print(f"len data to send: {len(to_send)}")

    # recv part
    if addr != '10.0.12.21': 
        while "quit" not in request.decode():
            print('nvlle itération') 
            request = await reader.read(1024) # -> va lire un packet de bytes du buffer de la socket
            if len(to_send) == 0:
                t0 = time.time()
            to_send += request # -> stocke le message reçu dans une variable globale 
            if "quit" in to_send.decode():
                data = eval(to_send.decode().replace("quit", ""))
                print(f"temps d'envoie:   {time.time()-data[1]:,}s")
                print(f"temps de lecture: {time.time()-t0:,}s")
                print(f'size get: {sys.getsizeof(to_send):,}')
                break
            if not request:
                break
#            if 'quit' in request.decode().split(';'):
#                print('add to buffer')
#                to_send = b''
#            try:
#                print(f"recieved: {request.decode()}")
#                print(f"tot: {sys.getsizeof(to_send):,} Bytes")
#            except:
#                pass
#            try:
#                print(f"tot: {eval(to_send.decode())}")
#            except:
#                print('failed')
        event.set()
        
    else:    
        await event.wait()
        print(f'sending: {to_send.decode()}')
        writer.write(to_send)
        await writer.drain()
        writer.close()
        to_send = b'' # remise à 0 des données à envoyer
        print(f"len data to send: {len(to_send)} | {addr}")
    

async def run_server():
    global event
    
    event = asyncio.Event() # -> création d'un event pour le signal de fin de connexion
    
    server = await asyncio.start_server(handle_client, '0.0.0.0', 5000) # -> création du serveur
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print(to_send.decode())
        print(f'size get: {sys.getsizeof(to_send):,}')
        


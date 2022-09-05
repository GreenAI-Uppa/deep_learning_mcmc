import asyncio

# add security close socket ?
# 10.0.12.5  -> client j4 : envoie les données
# 10.0.12.90 -> server p2 : reçoit j4 et envoie à p4
# 10.0.12.21 -> client p4 : reçoit p2 etc...

to_send = b''

async def handle_client(reader, writer):
    '''Lecture et écriture de données par ou pour un client'''
    global to_send
    
    stop = False # -> stop la boucle while si True
    addr = writer.get_extra_info('peername')[0] # -> ip client

    print(f"connected from {addr}")
    print(f"len data to send: {len(to_send)}")

    while not stop:
        # recv part
        if addr != '10.0.12.21': 
            request = await reader.read(1064) # -> va lire un packet de bytes du buffer de la socket
            stop = 'quit' in request.decode().split(';')  # permet la fermeture de la connexion client
            to_send += request # -> stocke le message reçu dans une variable globale 
            print(f"recieved: {request.decode()}") 
        
        # send part
        if (addr == '10.0.12.21') & (len(to_send) > 0):
            print(f'sending: {to_send.decode()}')
            writer.write(to_send)
            await writer.drain()
            writer.close()
            to_send = b''
            stop = True
        
        await asyncio.sleep(0.1)
        
        print(f"len data to send: {len(to_send)} | {addr}")
        if stop:
            break
    

async def run_server():
    server = await asyncio.start_server(handle_client, '0.0.0.0', 5000) # -> création du serveur
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(run_server())


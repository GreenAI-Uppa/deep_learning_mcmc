import asyncio
import time
import sys

'''
common function as write or read ? or check stop ?

cl = Client(connect_to=("0.0.0.0", 5000))
cl.start()

sv = Serveur = (reception_from=("0.0.0.0", 5000), sending_to=("0.0.0.0", 5000))
sv.start()
'''

class Connect():
    def __init__(self, local_name, reading_queue=None, sending_queue=None, log_latency=None, verbose=False):
        self.local_name = local_name
        self.reading_queue = reading_queue
        self.sending_queue = sending_queue
        self.log_latency = log_latency
        self.verbose = verbose
        self.stop_string = '__stop'.encode()
        
    async def sending(self, writer):
        data_to_send = await self.sending_queue.get()
        continue_while_loop = '__stop' not in data_to_send
        data_to_send = f'{data_to_send}__fin__'.encode()

        if self.verbose:
            print(f'''\nsending {sys.getsizeof(data_to_send):,} Bytes | {self.sending_queue.qsize():>3} in queue\n-------------''')

        writer.write(data_to_send)
        await writer.drain()

        print('! sent')
        self.sending_queue.task_done()

        return continue_while_loop
        
    async def reading(self, reader):
        new = True 
        i=0
        to_send = b''
        request = b''
        while (self.stop_string not in request): 
            request = await reader.read(1024) # -> va lire un packet de bytes du buffer de la socket
            i += 1
            # calcul du temps de reception 
            if new:
                t0 = time.time()
                new = False
            to_send += request # -> stocke le message reçu dans une variable globale 
            
            if (i % 2500 == 0):
                full_data = to_send.decode()
                if "__fin__" in full_data:
                    lecture = time.time()
                    # decoding
                    d = full_data.split('__fin__')
                    data = eval(d[0])
                    envoie = time.time()
                    # ajout des données à la queue
                    await self.reading_queue.put(data) 
                        
                    if self.verbose: self._details(i, sys.getsizeof(to_send), envoie, lecture, t0, data[2])
                    
                    if self.log_latency:
                        self.log_latency.write(f'{lecture-t0};{envoie-data[2]}\n')
                        self.log_latency.flush()
                    i = 0
                    
                    to_send = d[1].encode()
                    new = True
                    
                if "__stop" in full_data:
                    print("end of reading__")
                    break
                
    def _details(self, i, byte_size, envoie, lecture, t0, t1): # -> Connect
        print(f'''
--------------------------------
|Nouvelle Entrée               |
|------------------------------|
|i               : {i:<12,}|
|size            : {byte_size:<12,}|
|envoie          : {round(envoie-t1,2):<12,}|
|lecture         : {round(lecture-t0,2):<12,}|
|taille buffer   : {self.reading_queue.qsize():<12,}|
--------------------------------''')


class Client(Connect):
    def __init__(self, local_name, connect_to, reading_queue=None, sending_queue=None, log_latency=None, verbose=False):
        super().__init__(local_name, reading_queue, sending_queue, log_latency, verbose)
        self.connect_to = connect_to
        self.reader, self.writer = None, None
        if self.reading_queue and self.sending_queue:
            raise ValueError("define client with reading or sending")
    
    async def start(self):
        self.reader, self.writer = await asyncio.open_connection(*self.connect_to)
        print(f'{self.local_name} client connected to {self.connect_to}')
        
        await self.declare()
        
        if self.reading_queue:
            await self.reading(self.reader)
        if self.sending_queue:
            run = True
            while run:
                run = await self.sending(self.writer)
            print("end of writing__")
        
    async def declare(self):
        self.writer.write(self.local_name.encode())
        await self.writer.drain()
        
        
class Serveur(Connect):
    def __init__(self, local_name, reading_queue=None, sending_queue=None, log_latency=None, verbose=False, address=('0.0.0.0', 5000), send_to=None, read_from=None):
        super().__init__(local_name, reading_queue, sending_queue, log_latency, verbose)
        self.address = address
        self.send_to = send_to.encode() if type(send_to) is str else send_to
        self.read_from = read_from.encode() if type(read_from) is str else read_from
        self.stop_string = '__stop'.encode()
        
    async def start(self):
        server = await asyncio.start_server(self.handle_client, *self.address) # -> création du serveur
        async with server:
            await server.serve_forever()
        
    async def handle_client(self, reader, writer):
        
        addr = writer.get_extra_info('peername')[0] # -> get client ip
        print(f"New connection from {addr}")

        # recv part
        request = await reader.read(1024) # -> va lire un packet de bytes du buffer de la socket
        if request.decode() == self.read_from.decode(): # lecture de la socket
            await self.reading(reader)
        else: # sinon envoie de données dans la socket
            # envoie des données pretes dans le buffer de sortie
            run = True
            while run:
                run = await self.sending(writer)
            print("end of writing__")
import asyncio
import sys
import time
import socket
import json
class Connect():
    """
    A root class to send / read data based on asyncio tcp sockets
    ...
    Attributes
    ----------
    local_name: str
        name of the machina (allow the sender to identify itself)
    reading_queue: asyncio.Queue
        Buffer with reading data accumulated in
    sending_queue: asyncio.Queue
        Buffer with sending data accumulated in
    log_latency: _io.TextIOWrapper
        writing object to log latency
    verbose: bool
        print logs or not
        
    Methods
    -------
        wip
        
    Examples
    --------
    declare server
    
    >>> sv = connexion.Serveur(
        local_name="p8",
        sending_queue=sending_queue,
        reading_queue=reading_queue,
        log_latency=latency,
        read_from="j4",
        send_to="p4",
        verbose=True
    )
    >>> await sv.start()
    New connection from 10.0.12.21
    New connection from 10.0.12.239

    declare client
    >>> cl = connexion.Client(
        local_name="j4",
        connect_to=("10.0.12.18", 5000),
        sending_queue=queue_to_send,
        log_latency=latency,
        verbose=True
    )
    >>> cl.start()
    j4 client connected to ('10.0.12.18', 5000)
    """
    def __init__(self, local_name, reading_queue=None, sending_queue=None, log_latency=None, verbose=False):
        self.local_name = local_name
        self.reading_queue = reading_queue
        self.sending_queue = sending_queue
        self.log_latency = log_latency
        self.verbose = verbose
        self.stop_string = '__stop'.encode()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    async def sending(self):
        """sending data to tcp socket

        Args:
            writer (_type_): writer object from asyncio

        Returns:
            bool: continue while loop to send again data or not
        """
        data_to_send = await self.sending_queue.get()
        continue_while_loop = '__stop' not in data_to_send
        data_to_send = f'{data_to_send}__fin'.encode()

        if self.verbose:
            print(f'''\nsending {sys.getsizeof(data_to_send):,} Bytes | {self.sending_queue.qsize():>3} in queue\n-------------''')

        self.sock.sendall(data_to_send)
        print('! sent')
        self.sending_queue.task_done()

        return continue_while_loop
        
    async def reading(self):
        """reading data from tcp socket

        Args:
            reader (_type_): reader object from asyncio
        """
        new = True 
        i=0
        to_send = b''
        request = b''
        while (self.stop_string not in request): 
            request = await self.sock.recv(8_388_608)# -> va lire un packet de bytes du buffer de la socket
            i += 1
            # calcul du temps de reception 
            if new:
                t0 = time.time()
                new = False
            to_send += request # -> stocke le message reçu dans une variable globale 
            
            if b'__fin' in request:
                full_data = to_send.decode()
                lecture = time.time()
                # decoding
                d = full_data.split('__fin')
                data = json.loads(d[0])
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
        '''print details during reading data'''
        print(f'''
--------------------------------
|Nouvelle Entrée               |
|------------------------------|
|i               : {i:<12,}|
|size            : {byte_size:<11,}B|
|envoie          : {round(envoie-t1,2):<11,}s|
|lecture         : {round(lecture-t0,2):<11,}s|
|taille buffer   : {self.reading_queue.qsize():<12,}|
--------------------------------''')


class Client(Connect):
    def __init__(self, local_name, connect_to, reading_queue=None, sending_queue=None, log_latency=None, verbose=False):
        super().__init__(local_name, reading_queue, sending_queue, log_latency, verbose)
        self.connect_to = connect_to
        if self.reading_queue and self.sending_queue:
            raise ValueError("define your client with reading OR sending")
        
    async def start(self):
        '''start tcp client'''
        self.sock.connect(self.connect_to)
        print(f'{self.local_name} client connected to {self.connect_to}')
        
        await self.declare()
        
        if self.reading_queue:
            await self.reading()
            
        if self.sending_queue:
            run = True
            while run:
                run = await self.sending()
            print("end of writing__")
        
    async def declare(self):
        '''declare client to server with a tcp send with its local name'''
        self.sock.sendall(self.local_name.encode())
        
        
class Serveur(Connect):
    def __init__(self, local_name, reading_queue=None, sending_queue=None, log_latency=None, verbose=False, address=('0.0.0.0', 5000), send_to=None, read_from=None):
        super().__init__(local_name, reading_queue, sending_queue, log_latency, verbose)
        self.address = address
        self.send_to = send_to.encode() if type(send_to) is str else send_to
        self.read_from = read_from.encode() if type(read_from) is str else read_from
        self.stop_string = '__stop'.encode()
        # self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8_388_608) 
        self.sock.bind(self.address)
        self.sock.listen(2)
        self.sock.setblocking(False)
        
    async def start(self):
        '''start tcp server'''
        loop = asyncio.get_event_loop()
        try:
            while True:
                client, _ = await loop.sock_accept(self.server)
                loop.create_task(self.handle_client(client))
        except KeyboardInterrupt as e:
            print("server stopped")
            self.sock.close()
            
    async def handle_client(self, client):
        """applied function to each client connected to server

        Args:
            reader (_type_): reading object from asyncio
            writer (_type_): writing object from asyncio
        """
        # addr = writer.get_extra_info('peername')[0] # -> get client ip
        # print(f"New connection from {addr}")
        loop = asyncio.get_event_loop()
        
        # recv part
        request = await loop.sock_recv(client, 1024) # -> va lire un packet de bytes du buffer de la socket
        if request.decode() == self.read_from.decode(): # lecture de la socket
            await self.reading(client, loop)
        else: # sinon envoie de données dans la socket
            # envoie des données pretes dans le buffer de sortie
            run = True
            while run:
                run = await self.sending(client, loop)
            print("end of writing__")


    async def sending(self, client, loop): # different sending function for server because based on loop asyncio event
        """sending data to tcp socket

        Args:
            writer (_type_): writer object from asyncio

        Returns:
            bool: continue while loop to send again data or not
        """
        data_to_send = await self.sending_queue.get()
        continue_while_loop = '__stop' not in data_to_send
        data_to_send = f'{data_to_send}__fin'.encode()

        if self.verbose:
            print(f'''\nsending {sys.getsizeof(data_to_send):,} Bytes | {self.sending_queue.qsize():>3} in queue\n-------------''')

        await loop.sock_sendall(client, data_to_send)
        print('! sent')
        self.sending_queue.task_done()

        return continue_while_loop

    async def reading(self, client, loop): # different reading function for server because based on loop asyncio event
        """reading data from tcp socket

        Args:
            reader (_type_): reader object from asyncio
        """
        new = True 
        i=0
        to_send = b''
        request = b''
        while (self.stop_string not in request): 
            request = await loop.sock_recv(client, 8_388_608)# -> va lire un packet de bytes du buffer de la socket
            i += 1
            # calcul du temps de reception 
            if new:
                t0 = time.time()
                new = False
            to_send += request # -> stocke le message reçu dans une variable globale 
            
            if b'__fin' in request:
                full_data = to_send.decode()
                lecture = time.time()
                # decoding
                d = full_data.split('__fin')
                data = json.loads(d[0])
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
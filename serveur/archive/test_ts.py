import asyncio
import time
import torch
import sys
import concurrent

# data = [torch.tensor(i/100) for i in range(200000)]

async def tcp_echo_client():
    global event
    reader, writer = await asyncio.open_connection(
        '10.0.12.90', 5000)
    print("starting server")
    # print(f'size: {sys.getsizeof(str(data).encode()):,}')
    while True:
        print('waiting')
        await event.wait() # if data > 0
        print(f'triggered with : {r:,}')
        event.clear()
        # delete event
        # delete data
        # send data
        # writer.write(str((data, time.time())).encode())
        # await writer.drain()
        # writer.write('quit'.encode())
        # await writer.drain()
        # print('Close the connection')

        # writer.close()
        if r == -1:
            break


async def toto(f=None):
    global r
    r = 0
    for i in range(10_000_000):
        if i % 100_000 == 0:
            r=i
            print(f'trigger: {r:,}')
            event.set()
    r = -1
            
async def choose_func(func):
    if func == 'toto':
        await toto()
    else:
        await tcp_echo_client()


async def main():
    # corps de l'lago
    global event
    
    event = asyncio.Event()     
    # with concurrent.futures.ThreadPoolExecutor() as exec:
        # exec.map(choose_func, ['toto', 'tcp_echo_client'])
    asyncio.gather(choose_func, ['toto', 'tcp_echo_client'])
    # s = asyncio.create_task(tcp_echo_client())
    # t = asyncio.create_task(toto())

    # await s
    # await t
    
if __name__ == '__main__':
    asyncio.run(main())


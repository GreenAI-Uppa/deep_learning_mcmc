import asyncio

async def tcp_echo_client():
    stop = False
    data = b''
    reader, writer = await asyncio.open_connection(
        '10.0.12.90', 5000)
    
    while not stop:
        request = await reader.read(1064)
        stop = 'quit' in request.decode().split(';') 
        data += request
    print(f'Received: {data.decode().split(";")!r}')
    # suite et envoie des données etc.    

    print('Close the connection')
    writer.close()

asyncio.run(tcp_echo_client())
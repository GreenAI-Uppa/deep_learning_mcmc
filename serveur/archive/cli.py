import asyncio

data = [f'{i/100};'.encode() for i in range(100)]

async def tcp_echo_client():
    """echo data to server"""
    reader, writer = await asyncio.open_connection(
        '10.0.12.90', 5000) # -> ouverture de la connexion avec le serveur

    writer.write(b''.join(data)) # -> envoie des données concaténées par batch de 1064bytes
    writer.write('quit'.encode()) # -> instruction pour la fermeture de la socket
    
    print('Close the connection')
    writer.close()

def main():
    # corps de l'lago
    asyncio.run(tcp_echo_client()) # -> démarre le client TCP contenant du code asynchrone
    # le code continue dans un autre thread / pool

if __name__ == '__main__':
    main()

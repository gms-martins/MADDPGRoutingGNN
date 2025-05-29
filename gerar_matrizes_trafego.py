import json
import os
import random
import pickle

# Define o caminho da simulação
PATH_SIMULATION = os.path.dirname(os.path.abspath(__file__))

# Carrega a topologia do provedor de serviço
graph_topology = pickle.load(open(f"{PATH_SIMULATION}/TopologyFiles/service_provider_network.pickle", "rb"))

# Lista de todos os hosts
all_hosts = [f"H{node + 1}" for node in graph_topology.nodes()]

# A nova lista de nós que enviam tráfego (nós de extremidade)
nodes_list = ['H57', 'H65', 'H2', 'H3', 'H5', 'H6', 'H9', 'H10', 'H11', 'H15', 'H16', 'H18', 'H19', 'H24']

# Função para obter um destino aleatório
def get_random_dst(origin):
    while True:
        dst = random.choice(all_hosts)
        if dst != origin:  # Não enviar para si mesmo
            return dst

# Gerar as matrizes de tráfego
def generate_traffic_matrices(num_sequences, output_file):
    print(f"Gerando {num_sequences} sequências para {output_file}...")
    list_all_communications = []
    
    for j in range(num_sequences):
        communications = {}
        for host in nodes_list:
            communications[host] = []
            for i in range(20):  # 20 destinos por host
                dst = get_random_dst(host)
                communications[host].append(dst)
        list_all_communications.append(communications)
    
    # Salvar as matrizes no arquivo
    with open(output_file, "w") as f:
        json.dump(list_all_communications, f, indent=4)
    
    print(f"Arquivo {output_file} gerado com sucesso!")

# Gerar matrizes de tráfego para treinamento (1000 sequências)
generate_traffic_matrices(1000, f"{PATH_SIMULATION}/TrafficMatrix/tms_service_provider_train.json")

# Gerar matrizes de tráfego para teste (500 sequências)
generate_traffic_matrices(500, f"{PATH_SIMULATION}/TrafficMatrix/tms_service_provider_test.json")

print("Concluído! As matrizes de tráfego foram atualizadas para os novos nós transmissores (nós de extremidade).")
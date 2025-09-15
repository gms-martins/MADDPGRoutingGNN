import json
import os
import random
import pickle

PATH_SIMULATION = os.path.dirname(os.path.abspath(__file__))

# Carrega a topologia da internet
graph_topology = pickle.load(open(f"{PATH_SIMULATION}/TopologyFiles/small_network.pickle", "rb"))

# Lista de todos os hosts
all_hosts = [f"H{node + 1}" for node in graph_topology.nodes()]

# Hosts com grau 1 (degree 1)
degree_one_hosts = [f"H{node + 1}" for node in graph_topology.nodes() if graph_topology.degree(node) == 1]

# ESCOLHER APENAS 5 hosts de grau 1
num_hosts_to_select = 5
if len(degree_one_hosts) >= num_hosts_to_select:
    nodes_list = random.sample(degree_one_hosts, num_hosts_to_select)
else:
    nodes_list = degree_one_hosts  # Se tiver menos de 5, usa todos

print(f"Hosts de grau 1 disponíveis: {len(degree_one_hosts)}")
print(f"Hosts selecionados para matrizes: {nodes_list}")
print(f"Total selecionado: {len(nodes_list)}")

def get_random_dst(origin):
    while True:
        dst = random.choice(all_hosts)
        if dst != origin:
            return dst

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
    with open(output_file, "w") as f:
        json.dump(list_all_communications, f, indent=4)
    print(f"Arquivo {output_file} gerado com sucesso!")

# Matrizes para treino e teste
generate_traffic_matrices(1000, f"{PATH_SIMULATION}/TrafficMatrix/tms_internet_train.json")
generate_traffic_matrices(500, f"{PATH_SIMULATION}/TrafficMatrix/tms_internet_test.json")

print("Concluído! As matrizes de tráfego para a internet foram atualizadas para os hosts de grau 1.")
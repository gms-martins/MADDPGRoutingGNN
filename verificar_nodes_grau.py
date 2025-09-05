import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt  # Acrescenta esta linha

# Define o caminho da simulação
PATH_SIMULATION = os.path.dirname(os.path.abspath(__file__))

# Arquivo para resultado
resultado_file = os.path.join(PATH_SIMULATION, "resultado_analise_nos.txt")

# Função para escrever no arquivo e na tela
def print_and_write(text, f):
    print(text)
    f.write(text + "\n")

with open(resultado_file, "w") as f:
    # Carrega a topologia do provedor de serviço
    print_and_write("Carregando topologia da rede...", f)
    graph_topology = pickle.load(open(f"{PATH_SIMULATION}/TopologyFiles/service_provider_network.pickle", "rb"))

    # Lista de nós transmissores atuais do NetworkEngine.py
    nodes_list = ['H8', 'H13', 'H22', 'H31', 'H39', 'H48','H57','H12', 'H30', 'H21', 'H38', 'H47', 'H56', 'H65']

    # Analisar grau de cada nó (número de conexões)
    print_and_write("\nAnalisando número de conexões dos nós:", f)
    print_and_write("="*40, f)

    # Contar quantos nós existem por grau (número de conexões)
    degree_counts = {}
    node_degrees = {}
    for node in graph_topology.nodes():
        degree = graph_topology.degree(node)
        node_degrees[node] = degree
        degree_counts[degree] = degree_counts.get(degree, 0) + 1

    # Mostrar estatísticas gerais dos graus
    print_and_write("\nDistribuição dos graus dos nós:", f)
    for degree, count in sorted(degree_counts.items()):
        print_and_write(f"Nós com {degree} conexões: {count}", f)

    # Analisar os nós transmissores atuais
    print_and_write("\nAnalisando os nós transmissores atuais:", f)
    print_and_write("="*40, f)

    transmitter_degrees = []
    for node_name in nodes_list:
        node_index = int(node_name[1:]) - 1  # Converter de H1, H2... para 0, 1...
        if node_index in graph_topology.nodes():
            degree = graph_topology.degree(node_index)
            transmitter_degrees.append(degree)
            print_and_write(f"Nó {node_name}: {degree} conexões", f)
        else:
            print_and_write(f"Nó {node_name}: não encontrado na topologia", f)

    # Calcular a média de conexões dos transmissores atuais
    avg_transmitter_degree = sum(transmitter_degrees) / len(transmitter_degrees) if transmitter_degrees else 0
    print_and_write(f"\nMédia de conexões dos transmissores atuais: {avg_transmitter_degree:.2f}", f)

    # Identificar os nós com menor grau (potenciais nós de extremidade)
    min_degree = min(degree_counts.keys())
    print_and_write(f"\nNós com menor número de conexões (grau {min_degree}):", f)
    edge_nodes = [f"H{node + 1}" for node in graph_topology.nodes() if graph_topology.degree(node) == min_degree]
    print_and_write(str(edge_nodes), f)
    print_and_write(f"Total de nós de extremidade: {len(edge_nodes)}", f)

    # Verificar quantos dos transmissores atuais são nós de extremidade
    edge_transmitters = [node for node in nodes_list if node in edge_nodes]
    print_and_write(f"\nTransmissores atuais que são nós de extremidade: {len(edge_transmitters)} de {len(nodes_list)}", f)
    print_and_write(str(edge_transmitters), f)

    # Sugerir uma nova lista de nós transmissores baseada em nós de extremidade
    print_and_write("\nSugestão de nós transmissores (nós de extremidade):", f)
    suggested_transmitters = edge_nodes[:len(nodes_list)]  # Mesmo número que a lista atual
    print_and_write(str(suggested_transmitters), f)

    plt.figure(figsize=(12, 10))
    node_colors = []
    for node in graph_topology.nodes():
        node_name = f"H{node + 1}"
        if node_name in edge_nodes:
            if node_name in nodes_list:
                node_colors.append('red')      # Nó de extremidade e transmissor
            else:
                node_colors.append('orange')   # Só nó de extremidade
        elif node_name in nodes_list:
            node_colors.append('green')        # Só transmissor
        else:
            node_colors.append('skyblue')      # Restantes

    pos = nx.spring_layout(graph_topology, seed=42)
    nx.draw(graph_topology, pos, with_labels=True, labels={n: f"H{n+1}" for n in graph_topology.nodes()},
            node_color=node_colors, node_size=500, font_size=10)
    plt.title("Topologia Service Provider\nVermelho: extremidade+transmissor, Laranja: extremidade, Verde: transmissor")
    plt.tight_layout()
    img_path = os.path.join(PATH_SIMULATION, "service_provider_extremidades.png")
    plt.savefig(img_path)
    plt.close()
    print_and_write(f"\nImagem da topologia gerada em: {img_path}", f)
  

print(f"\nAnálise concluída! Resultados salvos em: {resultado_file}")

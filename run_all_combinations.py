import subprocess
import os
import time
import sys
import importlib

script_dir = os.environ.get("PATH_SIMULATION", "c:/Users/Utilizador/Ambiente de Trabalho/Tese/RRC_DRL_Update/RRC_DRL_Updates")
script_path = os.path.join(script_dir, "MADDPG.py")

# Topologia fixa
topology = "internet"  # ou "internet", "service_provider"
number_of_hosts = 25 #25 33 65
number_of_agents = 25 #25 33 65
nr_max_links = 11 #11 6 4

# Combinações de treino
train_configs = [
    ("central_critic", "duelling_q_network"),
    ("central_critic", "simple_q_network"),
    ("local_critic", "duelling_q_network"),
    ("shortest", "shortest")
]

# Cenários de teste
test_scenarios = [
    (False, False, False),  # Apenas treino
    (True, False, False),   # Apenas avaliar
    (True, True, False),    # Avaliar com update
    (True, True, True),     # Avaliar + treino (cenário 4)
]

# Adicionar timestamp único para pasta
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for critic, network in train_configs:
    for eval_flag, update_flag, train_flag in test_scenarios:
        # Gerar cenário
        if not eval_flag:
            scenario_name = "train"
        elif eval_flag and not train_flag and not update_flag:
            scenario_name = "eval"
        elif eval_flag and not train_flag and update_flag:
            scenario_name = "eval_update"
        else:
            scenario_name = "eval_train"
            
        # Gerar código atualizado para as variáveis
        env_file = os.path.join(script_dir, "environmental_variables.py")
        with open(env_file, "w") as f:
            f.write(f'''
NR_ACTIVE_CONNECTIONS = 10
NUMBER_OF_PATHS = 3

NR_EPOCHS = 5
EPOCH_SIZE = 5

NOTES = ""
PATH_SIMULATION = "{script_dir}"
SIM_NR = "{timestamp}_{critic}_{network}"
CHECKPOINT = False
CHECKPOINT_FILE = ""

INCREASE_BANDWIDTH_INTERVAL = 3
BANDWIDTH_INCREASE_FACTOR = 1.4
MAX_BANDWIDTH_MULTIPLIER = 2.0
STABILIZE_BANDWIDTH = True
STABILIZE_AFTER_MULTIPLIER = 1.75  
SAVE_REMOVED_LINKS_SCENARIO4 = True

TOPOLOGY_TYPE = "{topology}"; NUMBER_OF_HOSTS = {number_of_hosts}; NUMBER_OF_AGENTS = {number_of_agents}; NR_MAX_LINKS = {nr_max_links}
STATE_SIZE = NR_MAX_LINKS + 1 + NR_ACTIVE_CONNECTIONS * 2 + 1

CRITIC_DOMAIN = "{critic}"
NEURAL_NETWORK = "{network}"

EVALUATE = {eval_flag}
UPDATE_WEIGHTS = {update_flag}
TRAIN = {train_flag}

MODIFIED_NETWORK = "remove_edges"
''')

        # Garantir que o arquivo seja salvo completamente
        time.sleep(1)
        
        print(f"Running with: {critic} | {network} | EVAL={eval_flag}, UPDATE={update_flag}, TRAIN={train_flag}")
        
        # Usar o caminho completo e executar em um novo ambiente
        subprocess.run(["python", script_path], cwd=script_dir)
        
        # Forçar um atraso entre execuções para garantir que os
        # arquivos temporários sejam liberados e o cache seja resetado
        time.sleep(2)
NR_ACTIVE_CONNECTIONS = 10 # C: Número máximo de conexões ativas
NUMBER_OF_PATHS = 3   # Número de ações possíveis por destino k

NR_EPOCHS = 1    #epoch
EPOCH_SIZE = 1    #episode

NOTES = "" #notes to add to text file

PATH_SIMULATION = "c:/Users/Utilizador/Ambiente de Trabalho/Tese/RRC_DRL_Update/RRC_DRL_Updates"
#PATH_SIMULATION = "/home/server2/Thesis"  #path to test in the server

SIM_NR = "" #different folders for different simulations 1,2...

#continue simulation from saved file
CHECKPOINT = False
CHECKPOINT_FILE = ""

### NETWORK TOPOLOGY TYPE ###
#TOPOLOGY_TYPE = "internet"; NUMBER_OF_HOSTS = 25; NUMBER_OF_AGENTS = 25; NR_MAX_LINKS = 11
TOPOLOGY_TYPE = "arpanet"; NUMBER_OF_HOSTS = 33; NUMBER_OF_AGENTS = 33; NR_MAX_LINKS = 6
#TOPOLOGY_TYPE = "service_provider"; NUMBER_OF_HOSTS = 65; NUMBER_OF_AGENTS = 65; NR_MAX_LINKS = 4

STATE_SIZE = NR_MAX_LINKS + 1 + NR_ACTIVE_CONNECTIONS * 2 + 1

### TRAIN COMBINATIONS ###
#1 
#CRITIC_DOMAIN = "central_critic"; NEURAL_NETWORK = "duelling_q_network"
#2
CRITIC_DOMAIN = "central_critic"; NEURAL_NETWORK = "simple_q_network"
#3
#CRITIC_DOMAIN = "local_critic"; NEURAL_NETWORK = "duelling_q_network"
#4
#CRITIC_DOMAIN = "shortest"; NEURAL_NETWORK = "shortest"

### TEST COMBINATIONS ###
#EVALUATE = False; UPDATE_WEIGHTS = False; TRAIN = False # Tamanho total do estado
#EVALUATE = True; UPDATE_WEIGHTS = False; TRAIN = False
#EVALUATE = True; UPDATE_WEIGHTS = True; TRAIN = False
EVALUATE = True; UPDATE_WEIGHTS = True; TRAIN = True #train after topology changes

### Modified Network In Evaluate  ###
MODIFIED_NETWORK = "remove_edges"
#MODIFIED_NETWORK = "add_edges"

#GRAPH_BATCH_SIZE = 5 #batch size for graph

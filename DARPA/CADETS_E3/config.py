########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
raw_dir = "/the/absolute/path/of/cadets_e3/"

# The directory to save all artifacts
artifact_dir = "./artifact/"

# The directory to save the vectorized graphs
graphs_dir = artifact_dir + "graphs/"

# The directory to save the models
models_dir = artifact_dir + "models/"

# The directory to save the results after testing
test_re = artifact_dir + "test_re/"

# The directory to save all visualized results
vis_re = artifact_dir + "vis_re/"



########################################################
#
#               Database settings
#
########################################################

# Database name
database = 'test_e3_cadet_dataset_db'

# Only config this setting when you have the problem mentioned
# in the second point of the Problem&Solution section in README.
# Otherwise, please set it as None
host = '/var/run/postgresql/'
# host = None

# Database user
user = 'postgres'

# The password to the database user
password = 'postgres'

# The port number for Postgres
port = '5432'


########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
    "EVENT_ACCEPT",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG"
]

# The following edge types are not considered when loading the raw
# logs to the database
edge_excluded = ["EVENT_FLOWS_TO"]

# The following edges are the types only considered to construct the
# temporal graph for experiments.
include_edge_type=[
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

# The map between edge type and edge ID
rel2id = {
 1: 'EVENT_WRITE',
 'EVENT_WRITE': 1,
 2: 'EVENT_READ',
 'EVENT_READ': 2,
 3: 'EVENT_CLOSE',
 'EVENT_CLOSE': 3,
 4: 'EVENT_OPEN',
 'EVENT_OPEN': 4,
 5: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 5,
 6: 'EVENT_SENDTO',
 'EVENT_SENDTO': 6,
 7: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 7
}

########################################################
#
#                   Model dimensionality
#
########################################################

# Node Embedding Dimension
node_embedding_dim = 16

# Node State Dimension
node_state_dim = 100

# Neighborhood Sampling Size
neighbor_size = 20

# Edge Embedding Dimension
edge_dim = 100

# The time encoding Dimension
time_dim = 100


########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 1024

# Parameters for optimizer
lr=0.00005
eps=1e-08
weight_decay=0.01

epoch_num=50

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 15
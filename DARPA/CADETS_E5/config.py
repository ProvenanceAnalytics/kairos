########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
raw_dir = "/the/absolute/path/of/cadets_e5/"

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
database = 'tc_e5_cadets_dataset_db'

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
edge_reversed=[
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG"
]

# The following edges are the types only considered to construct the
# temporal graph for experiments.
include_edge_type=[
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_WRITE",
    "EVENT_EXECUTE",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_SENDMSG",
    "EVENT_SENDTO",
]

# The map between edge type and edge ID
rel2id={
 1: 'EVENT_CLOSE',
 'EVENT_CLOSE': 1,
 2: 'EVENT_OPEN',
 'EVENT_OPEN': 2,
 3: 'EVENT_READ',
 'EVENT_READ': 3,
 4: 'EVENT_WRITE',
 'EVENT_WRITE': 4,
 5: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 5,
 6: 'EVENT_RECVFROM',
 'EVENT_RECVFROM': 6,
 7: 'EVENT_RECVMSG',
 'EVENT_RECVMSG': 7,
 8: 'EVENT_SENDMSG',
 'EVENT_SENDMSG': 8,
 9: 'EVENT_SENDTO',
 'EVENT_SENDTO': 9}

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
edge_dim = 200

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

epoch_num=31

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 15


########################################################
#
#                   Threshold
#
########################################################

beta_day15 = 100
beta_day16 = 100
beta_day17 = 100

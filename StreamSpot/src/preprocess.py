# Input: StreamSpot dataset
# Output: Vectorized graphs

import functools
import os
import json
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *

from config import *

# If the datum have already been loaded in the database before, set this as False.
# Set it as True if it is the first time running this code.
process_raw_data = False


import psycopg2
from psycopg2 import extras as ex

connect = psycopg2.connect(database = 'streamspot',
                           host = '/var/run/postgresql/',
                           user = 'postgres',
                           password = 'postgres',
                           port = '5432'
                           )

# Create a cursor to operate the database
cur = connect.cursor()
# Rollback when there exists any problem
connect.rollback()

if process_raw_data:
    path = "/home/yinyuanl/Desktop/all.tsv"  # The paths to the dataset.
    datalist = []
    with open(path) as f:
        for line in tqdm(f):
            spl = line.strip().split('\t')
            datalist.append(spl)
            if len(datalist) >= 10000:
                sql = '''insert into raw_data
                 values %s
                '''
                ex.execute_values(cur, sql, datalist, page_size=10000)
                connect.commit()
                datalist = []

node_type = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}
edge_type = {'A',
             'B',
             'C',
             'D',
             'E',
             'F',
             'G',
             'H',
             'i',
             'j',
             'k',
             'l',
             'm',
             'n',
             'o',
             'p',
             'q',
             'r',
             's',
             't',
             'u',
             'v',
             'w',
             'x',
             'y',
             'z'}
maps = {'process': 'a',
        'thread': 'b',
        'file': 'c',
        'MAP_ANONYMOUS': 'd',
        'NA': 'e',
        'stdin': 'f',
        'stdout': 'g',
        'stderr': 'h',
        'accept': 'i',
        'access': 'j',
        'bind': 'k',
        'chmod': 'l',
        'clone': 'm',
        'close': 'n',
        'connect': 'o',
        'execve': 'p',
        'fstat': 'q',
        'ftruncate': 'r',
        'listen': 's',
        'mmap2': 't',
        'open': 'u',
        'read': 'v',
        'recv': 'w',
        'recvfrom': 'x',
        'recvmsg': 'y',
        'send': 'z',
        'sendmsg': 'A',
        'sendto': 'B',
        'stat': 'C',
        'truncate': 'D',
        'unlink': 'E',
        'waitpid': 'F',
        'write': 'G',
        'writev': 'H',
        }
nodevec = torch.nn.functional.one_hot(torch.arange(0, len(node_type)), num_classes=len(node_type))
edgevec = torch.nn.functional.one_hot(torch.arange(0, len(edge_type)), num_classes=len(edge_type))

edge2onehot = {}
node2onehot = {}
c = 0
for i in node_type:
    node2onehot[i] = nodevec[c]
    c += 1
c = 0
for i in edge_type:
    edge2onehot[i] = edgevec[c]
    c += 1

os.system("mkdir -p ../data/")
for graph_id in tqdm(range(600)):
    sql = "select * from raw_data where graph_id='{graph_id}' ORDER BY _id;".format(graph_id=graph_id)
    cur.execute(sql)
    rows = cur.fetchall()
    from torch_geometric.data import TemporalData

    dataset = TemporalData()
    src = []
    dst = []
    msg = []
    t = []
    for i in rows:
        src.append(int(i[0]))
        dst.append(int(i[2]))
        msg_t = torch.cat([node2onehot[i[1]], edge2onehot[i[4]], node2onehot[i[3]]], dim=0)
        msg.append(msg_t)
        t.append(int(i[-1]))    # Use logical order of the event to represent the time

    dataset.src = torch.tensor(src)
    dataset.dst = torch.tensor(dst)
    dataset.t = torch.tensor(t)
    dataset.msg = torch.vstack(msg)
    dataset.src = dataset.src.to(torch.long)
    dataset.dst = dataset.dst.to(torch.long)
    dataset.msg = dataset.msg.to(torch.float)
    dataset.t = dataset.t.to(torch.long)
    torch.save(dataset, "../data/graph_" + str(graph_id) + ".TemporalData")

print("end")

# 输入：Manzoor数据集
# 输出：一共600个行为对应的图数据集

import functools
import os
import json
import re
import torch
from tqdm import tqdm
from torch_geometric.data import *

from config import *

process_raw_data = False  # 如果数据库中已经将原始数据加载进来就设置为False， 如果第一次运行此代码，则需要设置为True

# 加载参数
parser = (argparse.ArgumentParser())
parse_encoder(parser)
args = parser.parse_args()

# 连接数据库
import psycopg2
from psycopg2 import extras as ex

connect = psycopg2.connect(host=args.db_host,
                           database=args.db_name,
                           user=args.db_user,
                           password=args.db_passwd,
                           port='5432'  # 一般是5432
                           )
# 创建一个cursor来执行数据库的操作
cur = connect.cursor()
# 发生错误时需要回滚
connect.rollback()

if process_raw_data:
    path = "../data/all.tsv"  # 数据集的路径
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
                connect.commit()  # 需要手动提交
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
        t.append(int(i[-1]))    # Here Zijun uses logical order of the event to represent the time

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

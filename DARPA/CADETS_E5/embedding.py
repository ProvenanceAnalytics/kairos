from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import numpy as np
import logging
import torch
import os

from config import *
from kairos_utils import *

# Setting for logging
logger = logging.getLogger("embedding_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'embedding.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

def gen_feature(cur):
    # Firstly obtain all node labels
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for i in tqdm(nodeid2msg.keys()):
        if type(i) == int:
            if 'netflow' in nodeid2msg[i].keys():
                higlist = ['netflow']
                higlist += ip2higlist(nodeid2msg[i]['netflow'])

            if 'file' in nodeid2msg[i].keys():
                higlist = ['file']
                higlist += path2higlist(nodeid2msg[i]['file'])

            if 'subject' in nodeid2msg[i].keys():
                higlist = ['subject']
                higlist += path2higlist(nodeid2msg[i]['subject'])
            node_msg_dic_list.append(list2str(higlist))

    # Featurize the hierarchical node labels
    FH_string = FeatureHasher(n_features=node_embedding_dim, input_type="string")
    node2higvec=[]
    for i in tqdm(node_msg_dic_list):
        vec=FH_string.transform([i]).toarray()
        node2higvec.append(vec)
    node2higvec = np.array(node2higvec).reshape([-1, node_embedding_dim])
    torch.save(node2higvec, artifact_dir + "node2higvec")
    return node2higvec

def gen_relation_onehot():
    relvec=torch.nn.functional.one_hot(torch.arange(0, len(rel2id.keys())//2), num_classes=len(rel2id.keys())//2)
    rel2vec={}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i]= relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]]=i
    torch.save(rel2vec, artifact_dir + "rel2vec")
    return rel2vec

def gen_vectorized_graphs(cur, node2higvec, rel2vec, logger):
    for day in tqdm(range(8, 18)):
        start_timestamp = datetime_to_ns_time_US('2019-05-' + str(day) + ' 00:00:00')
        end_timestamp = datetime_to_ns_time_US('2019-05-' + str(day + 1) + ' 00:00:00')
        sql = """
        select * from event_table
        where
              timestamp_rec>'%s' and timestamp_rec<'%s'
               ORDER BY timestamp_rec;
        """ % (start_timestamp, end_timestamp)
        cur.execute(sql)
        events = cur.fetchall()
        logger.info('2019-05-' + str(day), " events count:", str(len(events)))
        edge_list = []
        for e in events:
            edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
            if e[2] in include_edge_type:
                edge_list.append(edge_temp)
        logger.info('2019-05-' + str(day), " edge list len:", str(len(edge_list)))
        dataset = TemporalData()
        src = []
        dst = []
        msg = []
        t = []
        for i in edge_list:
            src.append(int(i[0]))
            dst.append(int(i[1]))
            msg.append(
                torch.cat([torch.from_numpy(node2higvec[i[0]]), rel2vec[i[2]], torch.from_numpy(node2higvec[i[1]])]))
            t.append(int(i[3]))
        if len(edge_list) > 0:
            dataset.src = torch.tensor(src)
            dataset.dst = torch.tensor(dst)
            dataset.t = torch.tensor(t)
            dataset.msg = torch.vstack(msg)
            dataset.src = dataset.src.to(torch.long)
            dataset.dst = dataset.dst.to(torch.long)
            dataset.msg = dataset.msg.to(torch.float)
            dataset.t = dataset.t.to(torch.long)
            torch.save(dataset, graphs_dir + "/graph_5_" + str(day) + ".TemporalData.simple")


if __name__ == "__main__":
    logger.info("Start logging.")

    os.system(f"mkdir -p {graphs_dir}")

    cur, _ = init_database_connection()
    node2higvec = gen_feature(cur=cur)
    rel2vec = gen_relation_onehot()
    gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, logger=logger)


# encoding=utf-8
import os.path as osp
import os
import copy
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import ICEWS18
from torch_geometric.nn import TGNMemory, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)
from torch_geometric import *
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import re
import time
import json
import pandas as pd
from random import choice
import gc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# msg的特征采取    [src_node_feature,edge_attr,dst_node_feature]的格式

# compute the best partition 计算出最佳的社区划分
import datetime
import community as community_louvain

import xxhash
# 查找edge向量所对应的下标
def tensor_find(t,x):
    t_np=t.numpy()
    idx=np.argwhere(t_np==x)
    return idx[0][0]+1


def std(t):
    t = np.array(t)
    return np.std(t)


def var(t):
    t = np.array(t)
    return np.var(t)


def mean(t):
    t = np.array(t)
    return np.mean(t)

def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()

# 单独计算出每条边的loss值来分析出 具体的异常行为
def cal_pos_edges_loss(link_pred_ratio):
    loss=[]
    for i in link_pred_ratio:
        loss.append(criterion(i,torch.ones(1)))
    return torch.tensor(loss)

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)

def cal_pos_edges_loss_autoencoder(decoded,msg):
    loss=[]
    for i in range(len(decoded)):
        loss.append(criterion(decoded[i].reshape(1,-1),msg[i].reshape(-1)))
    return torch.tensor(loss)





class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels*8, out_channels,heads=1, concat=False,
                             dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x



class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels*2)
        self.lin_dst = Linear(in_channels, in_channels*2)

        self.lin_seq = nn.Sequential(

            Linear(in_channels*4, in_channels*8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels*8, in_channels*2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels*2, int(in_channels//2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels//2), train_data.msg.shape[1]-32)
        )





    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src) , self.lin_dst(z_dst)],dim=-1)

        h = self.lin_seq (h)

        return h

# 更改节点嵌入向量维度
memory_dim = time_dim = embedding_dim = 100

memory = TGNMemory(
    max_node_num,
    train_data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(train_data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=train_data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()), lr=0.00005, eps=1e-08,weight_decay=0.01)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)


saved_nodes=set()



BATCH=1024
def train(train_data):


    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    saved_nodes=set()

    total_loss = 0

#     print("train_before_stage_data:",train_data)
    for batch in train_data.seq_batches(batch_size=BATCH):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()
#         n_id = torch.cat([src, pos_dst, neg_src, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        y_pred = torch.cat([pos_out], dim=0)

#         y_true = torch.cat([torch.zeros(pos_out.size(0),1),torch.ones(neg_out.size(0),1)], dim=0)# 0 代表正常 1 代表异常
        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l)

        y_true = torch.tensor(y_true)
        y_true=y_true.reshape(-1).to(torch.long)

        loss = criterion(y_pred, y_true)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

#         for i in range(len(src)):
#             saved_nodes.add(int(src[i]))
#             saved_nodes.add(int(pos_dst[i]))

        loss.backward()
        optimizer.step()
        memory.detach()
#         print(z.shape)
        total_loss += float(loss) * batch.num_events
#     print("trained_stage_data:",train_data)
    return total_loss / train_data.num_events




# 以下为定义的测试函数，可以输出每条edge的loss值
import time

# 检查点的时间是否设置长一些会更好呢？  比如说十分钟？
#  300000000000  是五分钟的纳秒时间段
# 测试一天的数据，每五分钟给出一个loss值  每五分钟给出一个checkpoint 的loss值
@torch.no_grad()#声明以下函数不执行梯度
def test_day_new(inference_data,path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

#     m=torch.load("model_saved_emb100.pt")
#     memory,gnn, link_pred,neighbor_loader=m
    memory.eval()
    gnn.eval()
    link_pred.eval()

    memory.reset_state()  # Start with a fresh memory.  # 为什么不可以使用历史的memory呢？  应该 可以吧？
    neighbor_loader.reset_state()  # Start with an empty graph.

    time_with_loss={}# key: 时间段，  value： 该时间段的loss值
    total_loss = 0
    edge_list=[]

    unique_nodes=torch.tensor([])
    total_edges=0

#     test_memory=copy.deepcopy(memory)
#     test_gnn=copy.deepcopy(gnn)
#     test_link_pred=copy.deepcopy(link_pred)
#     test_neighbor_loader=copy.deepcopy(neighbor_loader)

# 记录起始的时间点

    start_time=inference_data.t[0]
    event_count=0

    pos_o=[]

    loss_list=[]

#     print("before merge:",train_data)

#     nique_node_count=len(torch.cat([train_data.src,train_data.dst]).unique())

    print("after merge:",inference_data)

    # 记录程序运行时间  评估运行效率
    start = time.perf_counter()

    for batch in inference_data.seq_batches(batch_size=BATCH):

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        unique_nodes=torch.cat([unique_nodes,src,pos_dst]).unique()
        total_edges+=BATCH


        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        #如果memory和neighbor_loader在测试阶段都没有被更新的话，此处不需要采用gnn获得z矩阵，只需要从memory查出来对应的节点特征向量即可
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        pos_o.append(pos_out)
        y_pred = torch.cat([pos_out], dim=0)
#         y_true = torch.cat(
#             [torch.ones(pos_out.size(0))], dim=0).to(torch.long)
#         y_true=y_true.reshape(-1).to(torch.long)

        y_true=[]
        for m in msg:
            l=tensor_find(m[16:-16],1)-1
            y_true.append(l)
        y_true = torch.tensor(y_true)
        y_true=y_true.reshape(-1).to(torch.long)

        # 只考虑边有没有被正确预测，对于正常行为的图而言，行为模式比较相似所以loss较低。
        # 对于异常行为，会存在一些行为没见过，所以对这些行为预测存在的概率就地，所以loss也会高。
        loss = criterion(y_pred, y_true)

        total_loss += float(loss) * batch.num_events


# 将当前batch 发生的edge 更新到memory 和neighbor中
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        #计算每条边的loss值
        each_edge_loss= cal_pos_edges_loss_multiclass(pos_out,y_true)

        for i in range(len(pos_out)):
            srcnode=int(src[i])
            dstnode=int(pos_dst[i])

            srcmsg=str(nodeid2msg[srcnode])
            dstmsg=str(nodeid2msg[dstnode])
            t_var=int(t[i])
            edgeindex=tensor_find(msg[i][16:-16],1) # find 找出来的范围是 1-n   rel2id中的id也是1-n
            edge_type=rel2id[edgeindex]
            loss=each_edge_loss[i]

            temp_dic={}
            temp_dic['loss']=float(loss)
            temp_dic['srcnode']=srcnode
            temp_dic['dstnode']=dstnode
            temp_dic['srcmsg']=srcmsg
            temp_dic['dstmsg']=dstmsg
            temp_dic['edge_type']=edge_type
            temp_dic['time']=t_var

            # 先不考虑与socket 节点 当找出可疑的进程与文件之后再将socket找出
#             if "netflow" in srcmsg or "netflow" in dstmsg:
#                 temp_dic['loss']=0
            edge_list.append(temp_dic)

        event_count+=len(batch.src)
        if t[-1]>start_time+60000000000*15:
            # 此处为一个checkpoint  输出loss值 并且清空全局的loss值  保存处理过的节点
#             loss=total_loss/event_count
            time_interval=ns_time_to_datetime_US(start_time)+"~"+ns_time_to_datetime_US(t[-1])

            end = time.perf_counter()
            time_with_loss[time_interval]={'loss':loss,

                                          'nodes_count':len(unique_nodes),
                                          'total_edges':total_edges,
                                          'costed_time':(end-start)}


            log=open(path+"/"+time_interval+".txt",'w')
            # 减去train data中没有被训练好的

            for e in edge_list:
#                 temp_key=e['srcmsg']+e['dstmsg']+e['edge_type']
#                 if temp_key in train_edge_set:
# #                     e['loss']=(e['loss']-train_edge_set[temp_key]) if e['loss']>=train_edge_set[temp_key] else 0
# #                     e['loss']=abs(e['loss']-train_edge_set[temp_key])

#                     e['modified']=True
#                 else:
#                     e['modified']=False
                loss+=e['loss']

            loss=loss/event_count
            print(f'Time: {time_interval}, Loss: {loss:.4f}, Nodes_count: {len(unique_nodes)}, Cost Time: {(end-start):.2f}s')
            edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True)   # 按照loss 值排序  或者按照edge的时间顺序排列
            for e in edge_list: 
                log.write(str(e))
                log.write("\n")
            event_count=0
            total_loss=0
            loss=0
            start_time=t[-1]
            log.close()
            edge_list.clear()


    return time_with_loss

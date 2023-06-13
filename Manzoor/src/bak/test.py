from config import *
from utils import *

# encoding=utf-8
import os.path as osp
import os
import copy
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
# from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
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

# msg的特征采取    [src_node_feature,edge_attr,dst_node_feature]的格式
# compute the best partition 计算出最佳的社区划分
import datetime


# import community as community_louvain

# 单独计算出每条边的loss值来分析出 具体的异常行为
def cal_pos_edges_loss(link_pred_ratio):
    loss = []
    for i in link_pred_ratio:
        loss.append(criterion(i, torch.ones(1)))
    return torch.tensor(loss)


def cal_pos_edges_loss_multiclass(link_pred_ratio, labels):
    loss = []
    for i in range(len(link_pred_ratio)):
        loss.append(criterion(link_pred_ratio[i].reshape(1, -1), labels[i].reshape(-1)))
    return torch.tensor(loss)


def cal_pos_edges_loss_autoencoder(decoded, msg):
    loss = []
    for i in range(len(decoded)):
        loss.append(criterion(decoded[i].reshape(1, -1), msg[i].reshape(-1)))
    return torch.tensor(loss)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# 加载参数
parser = (argparse.ArgumentParser())
parse_encoder(parser)
args = parser.parse_args()

# load data
train_data = test_data = torch.load("../data/graph_305.TemporalData")
val_data = torch.load("../data/graph_505.TemporalData")
max_node_num = 5045000 + 1  # +1
min_dst_idx, max_dst_idx = 0, max_node_num
neighbor_loader = LastNeighborLoader(max_node_num, size=args.nei_size, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=8,
                                    dropout=0.0, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels * 8, out_channels, heads=1, concat=False,
                                     dropout=0.0, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        last_update.to(device)
        x = x.to(device)
        t = t.to(device)
        msg = msg.to(device)
        edge_index = edge_index.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(

            Linear(in_channels * 4, in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), train_data.msg.shape[1] - 2 * args.node_enc_size)
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)

        h = self.lin_seq(h)

        return h


memory_dim = args.state_size
time_dim = 100
embedding_dim = args.emb_size

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
    | set(link_pred.parameters()), lr=args.lr, eps=1e-08, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

saved_nodes = set()
BATCH = args.batch_size


def cal_anomaly_loss(loss_list):
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)

    thr = loss_mean + 1.5 * loss_std
    #     thr=0 # 因为训练的时候采用的就是整个batch均值的方式进行训练
    print("thr:", thr)

    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1
            loss_sum += loss_list[i]
    return loss_sum / count+0.0000001


@torch.no_grad()
def test_new(inference_data):
    #     m=torch.load("model_saved.pt")
    #     memory,gnn, link_pred,neighbor_loader=m

    total_loss = 0
    memory.reset_state()  # Start with a fresh memory.  # 为什么不可以使用历史的memory呢？  应该 可以吧？
    neighbor_loader.reset_state()  # Start with an empty graph.

    aps, aucs = [], []
    pos_o = []
    total_loss = 0
    test_loader = TemporalDataLoader(inference_data, batch_size=BATCH)
    edges_loss = []
    for batch in test_loader:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        y_pred = torch.cat([pos_out], dim=0)

        y_true = []
        for m in msg:
            l = tensor_find(m[args.node_enc_size:-args.node_enc_size], 1) - 1
            y_true.append(l)
        y_true = torch.tensor(y_true)
        loss = criterion(y_pred, y_true)
        total_loss += float(loss) * batch.num_events

        # update node state and neighbor
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # 计算每条边的loss值
        each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

        edges_loss += list(each_edge_loss)

    loss = total_loss / inference_data.num_events
    edges_loss = list(map(float, edges_loss))
    return float(torch.tensor(aps).mean()), float \
        (torch.tensor(aucs).mean()), pos_out.sigmoid().cpu(), loss, edges_loss


m = torch.load("../models/model_saved_emb100.pt")
memory, gnn, link_pred, neighbor_loader = m
memory.eval()
gnn.eval()
link_pred.eval()

# start = time.time()
# test_ap, test_auc, pos_out_test, loss_test = test_new(test_data)
# print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
# print(f"test cost time:{time.time() - start}")

# 生成阈值
SET_THRESHOLD = False
if SET_THRESHOLD:
    threshold = 0

    val_loss_list = []

    ap = []
    auc = []
    graph_label = []
    edges_loss_list = []
    all_loss = []
    start = time.time()
    for i in tqdm(range(1, 25)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
        print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        edges_loss_list.append(edges_loss)
        graph_label.append(0)

    for i in tqdm(range(101, 125)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
        print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        edges_loss_list.append(edges_loss)
        graph_label.append(0)
    for i in tqdm(range(201, 225)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
        print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        edges_loss_list.append(edges_loss)
        graph_label.append(0)
    #
    for i in tqdm(range(401, 425)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
        print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        edges_loss_list.append(edges_loss)
        graph_label.append(0)
    for i in tqdm(range(501, 525)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
        print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        edges_loss_list.append(edges_loss)
        graph_label.append(0)

    print(f"test cost time:{time.time() - start}")

    threshold = max(all_loss)

    test_ans = [all_loss, graph_label, edges_loss_list]
    torch.save(test_ans, "val_ans.pt")
else:
    test_ans = torch.load("val_ans.pt")
    loss_list=[]
    for i in test_ans[-1]:
        loss_list.append(cal_anomaly_loss(i))
    threshold = max(loss_list)

# 收集测试的数据
ap = []
auc = []
graph_label = []
all_loss = []
edges_loss_list = []
start = time.time()

for i in tqdm(range(25, 100)):
    path = "../data/graph_" + str(i) + ".TemporalData"
    test_graph = torch.load(path)
    test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
    print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
    all_loss.append(loss_test)
    edges_loss_list.append(edges_loss)
    graph_label.append(0)

for i in tqdm(range(125, 200)):
    path = "../data/graph_" + str(i) + ".TemporalData"
    test_graph = torch.load(path)
    test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
    print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
    all_loss.append(loss_test)
    edges_loss_list.append(edges_loss)
    graph_label.append(0)

for i in tqdm(range(225, 300)):
    path = "../data/graph_" + str(i) + ".TemporalData"
    test_graph = torch.load(path)
    test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
    print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
    all_loss.append(loss_test)
    edges_loss_list.append(edges_loss)
    graph_label.append(0)

for i in tqdm(range(300, 400)):
    path = "../data/graph_" + str(i) + ".TemporalData"
    test_graph = torch.load(path)
    test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
    print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
    all_loss.append(loss_test)
    edges_loss_list.append(edges_loss)
    graph_label.append(1)

for i in tqdm(range(425, 500)):
    path = "../data/graph_" + str(i) + ".TemporalData"
    test_graph = torch.load(path)
    test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
    print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
    all_loss.append(loss_test)
    edges_loss_list.append(edges_loss)
    graph_label.append(0)

for i in tqdm(range(525, 600)):
    path = "../data/graph_" + str(i) + ".TemporalData"
    test_graph = torch.load(path)
    test_ap, test_auc, pos_out_test, loss_test, edges_loss = test_new(test_graph)
    print(f'Graph:{i}  Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Loss: {loss_test:.4f}')
    all_loss.append(loss_test)
    edges_loss_list.append(edges_loss)
    graph_label.append(0)

print(f"test cost time:{time.time() - start}")

test_ans = [all_loss, graph_label, edges_loss_list]
torch.save(test_ans, "test_ans.pt")

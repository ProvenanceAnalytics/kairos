##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import os
import os.path as osp

import torch
from torch.nn import Linear

# from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage,
                                           LastAggregator)
from torch_geometric import *
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import re
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# msg structure:      [src_node_feature,edge_attr,dst_node_feature]
train_data = torch.load("../data/graph_0.TemporalData")

max_node_num = 5045000
min_dst_idx, max_dst_idx = 0, max_node_num
neighbor_loader = LastNeighborLoader(max_node_num, size=5, device=device)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


memory_dim = time_dim = embedding_dim = 200

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
    | set(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
BATCH=1024

@torch.no_grad()
def test_new(inference_data):
    # memory.eval()
    # gnn.eval()
    # link_pred.eval()

    total_loss = 0
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    torch.manual_seed(12345)  # Ensure determi|nistic sampling across epochs.

    aps, aucs = [], []
    pos_o = []

    # test_loader = TemporalDataLoader(inference_data, batch_size=BATCH)
    for batch in inference_data.seq_batches(batch_size=BATCH):
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])
        pos_o.append(pos_out)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
        total_loss += float(loss) * batch.num_events

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    loss = total_loss / inference_data.num_events
    return float(torch.tensor(aps).mean()), float(
        torch.tensor(aucs).mean()), pos_out.sigmoid().cpu(), neg_out.sigmoid().cpu(), loss

m = torch.load("model_saved.pt")
memory, gnn, link_pred, neighbor_loader = m
memory.eval()
gnn.eval()
link_pred.eval()

if os.path.exists("val_ans_old.pt") is not True:
    graph_label = []
    all_loss = []
    start = time.time()
    for i in tqdm(range(1, 25)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(101, 125)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(201, 225)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(401, 425)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(501, 525)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    print(f"test cost time:{time.time() - start}")

    val_ans_old = [ all_loss, graph_label]
    torch.save(val_ans_old, "val_ans_old.pt")
else:
    val_ans = torch.load("val_ans_old.pt")
    loss_list = []
    for i in val_ans[0]:
        loss_list.append(i)
    threshold = max(loss_list)


def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    print('tn:',tn)
    print('fp:',fp)
    print('fn:',fn)
    print('tp:',tp)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    print("precision:",precision)
    print("recall:",recall)
    print("fscore:",fscore)
    print("accuracy:",accuracy)
    print("auc_val:",auc_val)
    return precision,recall,fscore,accuracy,auc_val


if os.path.exists("test_ans_old.pt") is not True:
    graph_label = []
    all_loss = []
    start = time.time()
    for i in tqdm(range(25, 100)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(125, 200)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(225, 300)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(300, 400)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(1)

    for i in tqdm(range(425, 500)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    for i in tqdm(range(525, 600)):
        path = "../data/graph_" + str(i) + ".TemporalData"
        test_graph = torch.load(path)
        test_ap, test_auc, pos_out_test, neg_out_test, loss_test = test_new(test_graph)
        print(f'Graph:{i}, Loss: {loss_test:.4f}')
        all_loss.append(loss_test)
        graph_label.append(0)

    print(f"test cost time:{time.time() - start}")
    test_ans_old = [all_loss, graph_label]
    torch.save(test_ans_old, "test_ans_old.pt")
else:
    labels = []
    preds = []

    test_ans = torch.load("test_ans_old.pt")
    test_loss_list = []
    index = 0
    for i in test_ans[0]:
        temp_loss = i
        label = test_ans[1][index]
        if temp_loss > threshold:
            pred = 1
        else:
            pred = 0

        labels.append(label)
        preds.append(pred)

        index += 1

        # If prediction is incorrect, print the information of the tested graph
        if pred != label:
            print(f"{index=} {temp_loss=} {label=} {pred=} {pred == label}")

    classifier_evaluation(labels, preds)








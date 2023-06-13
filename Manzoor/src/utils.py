import xxhash
import numpy as np


# 查找edge向量所对应的下标
def tensor_find(t, x):
    t = t.to('cpu')
    # x = x.to('cpu')
    t_np = t.numpy()
    idx = np.argwhere(t_np == x)
    return idx[0][0] + 1


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

#
from sklearn.metrics import average_precision_score, roc_auc_score
# # 导入plot_roc_curve,roc_curve和roc_auc_score模块
# from sklearn.metrics import plot_roc_curve, roc_curve, auc, roc_auc_score
# import torch
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix
#
#
#
# def classifier_evaluation(y_test, y_test_pred):
#     # groundtruth, pred_value
#     tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
#     #     tn+=100
#     #     print(clf_name," : ")
#     print('tn:', tn)
#     print('fp:', fp)
#     print('fn:', fn)
#     print('tp:', tp)
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     accuracy = (tp + tn) / (tp + tn + fp + fn)
#     fscore = 2 * (precision * recall) / (precision + recall)
#     auc_val = roc_auc_score(y_test, y_test_pred)
#     print("precision:", precision)
#     print("recall:", recall)
#     print("fscore:", fscore)
#     print("accuracy:", accuracy)
#     print("auc_val:", auc_val)
#     return precision, recall, fscore, accuracy, auc_val
#
#
# def minmax(data):
#     min_val = min(data)
#     max_val = max(data)
#     ans = []
#     for i in data:
#         ans.append((i - min_val) / (max_val - min_val))
#     return ans



import torch
import numpy as np


def std(t):
    t = np.array(t)
    return np.std(t)


def var(t):
    t = np.array(t)
    return np.var(t)


def mean(t):
    t = np.array(t)
    return np.mean(t)


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
    return loss_sum / count + 0.0000001


# (5104, 5.261780796052894)
val_ans = torch.load("val_ans_old.pt")
loss_list = []
for i in val_ans[0]:
    loss_list.append(i)
threshold = max(loss_list)



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
    index += 1
    if pred != label:
        print(f"{index=} {temp_loss=} {label=} {pred=} {pred == label}")


    # test_loss_list.append(cal_anomaly_loss(i))


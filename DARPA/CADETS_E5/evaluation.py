from sklearn.metrics import confusion_matrix
import logging

from kairos_utils import *
from config import *
from model import *


# Setting for logging
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'evaluation.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")
    return precision,recall,fscore,accuracy,auc_val

def ground_truth_label():
    labels = {}
    filelist = os.listdir(f"{artifact_dir}/graph_5_16")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_5_17")
    for f in filelist:
        labels[f] = 0

    attack_list = [
        '2019-05-16 09:20:32.093582942~2019-05-16 09:36:08.903494477.txt',
        '2019-05-16 09:36:08.903494477~2019-05-16 09:51:22.110949680.txt',
        '2019-05-16 09:51:22.110949680~2019-05-16 10:06:29.403713371.txt',
        '2019-05-16 10:06:29.403713371~2019-05-16 10:21:47.983513184.txt',

        # Here are the "fake" FP time windows described in Section 5.2 in the paper.
        # '2019-05-16 20:32:27.570220441~2019-05-16 20:48:38.072848659.txt',
        # '2019-05-16 21:19:00.930018779~2019-05-16 21:34:46.231624861.txt',
        # '2019-05-16 21:34:46.231624861~2019-05-16 21:49:46.992678639.txt',
        # '2019-05-16 21:49:46.992678639~2019-05-16 22:06:14.950154813.txt',
        # '2019-05-16 22:06:14.950154813~2019-05-16 22:21:40.662702391.txt',
        # '2019-05-16 22:21:40.662702391~2019-05-16 22:36:45.602858389.txt',
        # '2019-05-16 22:36:45.602858389~2019-05-16 22:51:51.220035024.txt',
        # '2019-05-16 22:51:51.220035024~2019-05-16 23:07:16.890296254.txt',
        # '2019-05-16 23:07:16.890296254~2019-05-16 23:22:54.052353000.txt',

        '2019-05-17 10:02:11.321524261~2019-05-17 10:17:26.881636687.txt',
        '2019-05-17 10:17:26.881636687~2019-05-17 10:32:38.131495470.txt',
        '2019-05-17 10:32:38.131495470~2019-05-17 10:48:02.091564015.txt'
    ]
    for i in attack_list:
        labels[i] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = [
            'nginx',
            # '128.55.12.167',
            # '4.21.51.250',
            # 'ocMain.py',
            'python',
            # '98.23.182.25',
            # '108.192.100.31',
            'hostname',
            'whoami',
            # 'cat /etc/passwd',
        ]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    attack_list = [
        '2019-05-16 09:20:32.093582942~2019-05-16 09:36:08.903494477.txt',
        '2019-05-16 09:36:08.903494477~2019-05-16 09:51:22.110949680.txt',
        '2019-05-16 09:51:22.110949680~2019-05-16 10:06:29.403713371.txt',
        '2019-05-16 10:06:29.403713371~2019-05-16 10:21:47.983513184.txt',
        '2019-05-17 10:02:11.321524261~2019-05-17 10:17:26.881636687.txt',
        '2019-05-17 10:17:26.881636687~2019-05-17 10:32:38.131495470.txt',
        '2019-05-17 10:32:38.131495470~2019-05-17 10:48:02.091564015.txt'
    ]
    for f in attack_list:
        if "05-16" in f:
            files.append(f"{artifact_dir}/graph_5_16/{f}")
        else:
            files.append(f"{artifact_dir}/graph_5_17/{f}")

    attack_edge_count = 0
    for fpath in (files):
        f = open(fpath)
        for line in f:
            if keyword_hit(line):
                attack_edge_count += 1
    logger.info(f"Num of attack edges: {attack_edge_count}")

if __name__ == "__main__":
    logger.info("Start logging.")

    # Validation date
    anomalous_queue_scores = []
    history_list = torch.load(f"{artifact_dir}/graph_5_12_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                # Plus 1 to ensure anomaly score is monotonically increasing
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []

        for i in hl:
            name_list.append(i['name'])
        # logger.info(f"Constructed queue: {name_list}")
        # logger.info(f"Anomaly score: {anomaly_score}")

        anomalous_queue_scores.append(anomaly_score)
    logger.info(f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n")


    # Evaluating the testing set
    pred_label = {}

    filelist = os.listdir(f"{artifact_dir}/graph_5_15/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_5_16/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_5_17/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{artifact_dir}/graph_5_15_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day15:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{artifact_dir}/graph_5_16_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day16:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{artifact_dir}/graph_5_17_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day17:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    # Calculate the metrics
    labels = ground_truth_label()
    y = []
    y_pred = []
    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label[i])
    classifier_evaluation(y, y_pred)
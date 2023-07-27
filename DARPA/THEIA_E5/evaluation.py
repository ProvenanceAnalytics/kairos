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
    filelist = os.listdir(f"{artifact_dir}/graph_5_14")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_5_15")
    for f in filelist:
        labels[f] = 0

    attack_list = [
        '2019-05-15 13:58:15.520482252~2019-05-15 14:13:37.257086895.txt',
        '2019-05-15 14:44:51.773840192~2019-05-15 15:00:26.765466538.txt',
    ]
    for i in attack_list:
        labels[i] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = [
            # 'sshd',
            'sshdlog',
            'shm',
            # 'python',
            # 'firefox',
            '189.141.204.211',
            '208.203.20.42',
        ]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    attack_list = [
        '2019-05-15 13:58:15.520482252~2019-05-15 14:13:37.257086895.txt',
        '2019-05-15 14:44:51.773840192~2019-05-15 15:00:26.765466538.txt',
    ]
    for f in attack_list:
        files.append(f"{artifact_dir}/graph_5_15/{f}")

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
    history_list = torch.load(f"{artifact_dir}/graph_5_11_history_list")
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

    filelist = os.listdir(f"{artifact_dir}/graph_5_14/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_5_15/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{artifact_dir}/graph_5_14_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day14:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

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
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")

    # Calculate the metrics
    labels = ground_truth_label()
    y = []
    y_pred = []
    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label[i])
    classifier_evaluation(y, y_pred)
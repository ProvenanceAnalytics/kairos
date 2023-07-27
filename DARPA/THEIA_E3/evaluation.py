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
    filelist = os.listdir(f"{artifact_dir}/graph_4_10")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_11")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{artifact_dir}/graph_4_12")
    for f in filelist:
        labels[f] = 0

    attack_list = [
        '2018-04-10 13:31:14.548738409~2018-04-10 13:46:36.161065223.txt',
        '2018-04-10 14:02:17.001271389~2018-04-10 14:17:34.001373488.txt',
        '2018-04-10 14:17:34.001373488~2018-04-10 14:33:18.350772859.txt',
        '2018-04-10 14:33:18.350772859~2018-04-10 14:48:47.320442910.txt',
        '2018-04-10 14:48:47.320442910~2018-04-10 15:03:54.307022037.txt',

        '2018-04-12 12:39:06.592684498~2018-04-12 12:54:44.001888457.txt',
        '2018-04-12 12:54:44.001888457~2018-04-12 13:09:55.026832462.txt',
        '2018-04-12 13:09:55.026832462~2018-04-12 13:25:06.588370709.txt',
        '2018-04-12 13:25:06.588370709~2018-04-12 13:40:07.178206094.txt',
    ]
    for i in attack_list:
        labels[i] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = [
            '/home/admin/clean',
            '/dev/glx_alsa_675',
            '/home/admin/profile',
            # '/var/log/mail',  # This will include the port scanning attacks, resulting a large number of the data.
            '/tmp/memtrace.so',
            '/var/log/xdev',
            '/var/log/wdev',
            'gtcache',
            # 'firefox',
            '161.116.88.72',
            '146.153.68.151',
            '145.199.103.57',
            '61.130.69.232',
            '5.214.163.155',
            '104.228.117.212',
            '141.43.176.203',
            '7.149.198.40',
            '5.214.163.155',
            '149.52.198.23',
        ]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    attack_list = [
        '2018-04-10 13:31:14.548738409~2018-04-10 13:46:36.161065223.txt',
        '2018-04-10 14:02:17.001271389~2018-04-10 14:17:34.001373488.txt',
        '2018-04-10 14:17:34.001373488~2018-04-10 14:33:18.350772859.txt',
        '2018-04-10 14:33:18.350772859~2018-04-10 14:48:47.320442910.txt',
        '2018-04-10 14:48:47.320442910~2018-04-10 15:03:54.307022037.txt',
        '2018-04-12 12:39:06.592684498~2018-04-12 12:54:44.001888457.txt',
        '2018-04-12 12:54:44.001888457~2018-04-12 13:09:55.026832462.txt',
        '2018-04-12 13:09:55.026832462~2018-04-12 13:25:06.588370709.txt',
        '2018-04-12 13:25:06.588370709~2018-04-12 13:40:07.178206094.txt',
    ]
    for f in attack_list:
        if "04-10" in f:
            files.append(f"{artifact_dir}/graph_4_10/{f}")
        else:
            files.append(f"{artifact_dir}/graph_4_12/{f}")

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
    history_list = torch.load(f"{artifact_dir}/graph_4_9_history_list")
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

    filelist = os.listdir(f"{artifact_dir}/graph_4_10/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_4_11/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{artifact_dir}/graph_4_12/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{artifact_dir}/graph_4_10_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day10:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{artifact_dir}/graph_4_11_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day11:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{artifact_dir}/graph_4_12_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day12:
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
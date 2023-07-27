import os
import re
import torch
from tqdm import tqdm
import hashlib


from config import *
from kairos_utils import *

filelist = [
    'ta1-theia-e3-official-6r.json.9',
    'ta1-theia-e3-official-1r.json.9',
    'ta1-theia-e3-official-6r.json.8',
    'ta1-theia-e3-official-6r.json.12',
    'ta1-theia-e3-official-1r.json.7',
    'ta1-theia-e3-official-6r.json.7',
    'ta1-theia-e3-official-6r.json.5',
    'ta1-theia-e3-official-1r.json.3',
    'ta1-theia-e3-official-6r.json',
    'ta1-theia-e3-official-1r.json.5',
    'ta1-theia-e3-official-6r.json.11',
    'ta1-theia-e3-official-1r.json.4',
    'ta1-theia-e3-official-1r.json.6',
    'ta1-theia-e3-official-5m.json',
    'ta1-theia-e3-official-1r.json.2',
    'ta1-theia-e3-official-6r.json.10',
    'ta1-theia-e3-official-6r.json.4',
    'ta1-theia-e3-official-6r.json.1',
    'ta1-theia-e3-official-3.json',
    'ta1-theia-e3-official-1r.json.8',
    'ta1-theia-e3-official-1r.json.1',
    'ta1-theia-e3-official-6r.json.6',
    'ta1-theia-e3-official-6r.json.2',
    'ta1-theia-e3-official-6r.json.3',
    'ta1-theia-e3-official-1r.json'
]


def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def store_netflow(file_path, cur, connect):
    # Parse data from logs
    netobjset = set()
    netobj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                    try:
                        res = re.findall('NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),', line)[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]

                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport  # 只关注向哪里发起的网络流 合理么？
                        hashstr = stringtomd5(nodeproperty)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []
    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(","))

    sql = '''insert into netflow_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return netobj2hash

def store_subject(file_path, cur, connect):
    # Parse data from logs
    subjectset = set()
    subject2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                    res = re.findall('Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"', line)[0]
                    try:
                        path_str = re.findall('"path":"(.*?)"', line)[0]
                        path = path_str
                    except:
                        path = "null"
                    nodeid = res[0]
                    cmdLine = res[2]
                    tgid = res[4]

                    nodeproperty = cmdLine + "," + tgid + "," + path
                    hashstr = stringtomd5(nodeproperty)
                    subject2hash[nodeid] = [hashstr, cmdLine, tgid, path]
                    subject2hash[hashstr] = nodeid
                    subjectset.add(hashstr)

    # Store into database
    datalist = []
    for i in subject2hash.keys():
        if len(i) != 64:
            datalist.append([i] + subject2hash[i])

    sql = '''insert into subject_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return subject2hash

def store_file(file_path, cur, connect):
    fileset = set()
    file2hash = {}

    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                    try:
                        res = re.findall('FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"', line)[0]
                        nodeid = res[0]
                        filepath = res[2]
                        nodeproperty = filepath
                        hashstr = stringtomd5(nodeproperty)
                        file2hash[nodeid] = [hashstr, nodeproperty]
                        file2hash[hashstr] = nodeid
                        fileset.add(hashstr)
                    except:
                        pass

    datalist = []
    for i in file2hash.keys():
        if len(i) != 64:
            datalist.append([i] + file2hash[i])

    sql = '''insert into file_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

    return file2hash

def create_node_list(cur, connect):
    node_list = {}

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["file", i[-1]]
    file_uuid2hash = {}
    for i in records:
        file_uuid2hash[i[0]] = i[1]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["subject", i[-1]]
    subject_uuid2hash = {}
    for i in records:
        subject_uuid2hash[i[0]] = i[1]

    # netflow
    sql = """
    select * from netflow_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["netflow", i[-2] + ":" + i[-1]]
    net_uuid2hash = {}
    for i in records:
        net_uuid2hash[i[0]] = i[1]

    node_list_database = []
    node_index = 0
    for i in node_list:
        node_list_database.append([i] + node_list[i] + [node_index])
        node_index += 1

    sql = '''insert into node2id
                         values %s
            '''
    ex.execute_values(cur, sql, node_list_database, page_size=10000)
    connect.commit()

    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg

def store_event(file_path, cur, connect, reverse, nodeid2msg, subject2hash, file2hash, netobj2hash):
    datalist = []
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                    time = re.findall('"timestampNanos":(.*?),', line)[0]
                    time = int(time)  # 将时间转成秒为单位
                    subjectid = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)[0]
                    objectid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"', line)[0]
                    relation_type = re.findall('"type":"(.*?)"', line)[0]

                    if subjectid in subject2hash.keys():
                        subjectid = subject2hash[subjectid][0]
                    if objectid in subject2hash.keys():
                        objectid = subject2hash[objectid][0]
                    if objectid in file2hash.keys():
                        objectid = file2hash[objectid][0]
                    if objectid in netobj2hash.keys():
                        objectid = netobj2hash[objectid][0]
                    if len(subjectid) == 64 and len(objectid) == 64:
                        if relation_type in reverse:
                            datalist.append(
                                (objectid, nodeid2msg[objectid], relation_type, subjectid, nodeid2msg[subjectid], time))
                        else:
                            datalist.append(
                                (subjectid, nodeid2msg[subjectid], relation_type, objectid, nodeid2msg[objectid], time))

                        if len(datalist) >= 10000:
                            try:
                                sql = '''insert into event_table
                                                         values %s
                                            '''
                                ex.execute_values(cur, sql, datalist, page_size=100000)
                                connect.commit()
                                datalist = []
                            except:
                                print("Failed to write database！")
                                connect.rollback()


if __name__ == "__main__":
    cur, connect = init_database_connection()

    # There will be 186100 netflow nodes stored in the table
    print("Processing netflow data")
    netobj2hash = store_netflow(file_path=raw_dir, cur=cur, connect=connect)

    # There will be 279369 subject nodes stored in the table
    print("Processing subject data")
    subject2hash = store_subject(file_path=raw_dir, cur=cur, connect=connect)

    # There will be 793899 file nodes stored in the table
    print("Processing file data")
    file2hash = store_file(file_path=raw_dir, cur=cur, connect=connect)

    # There will be 828312 entities stored in the table
    print("Extracting the node list")
    nodeid2msg = create_node_list(cur=cur, connect=connect)

    # There will be 44400000 events stored in the table
    print("Processing the events")
    store_event(
        file_path=raw_dir,
        cur=cur,
        connect=connect,
        reverse=edge_reversed,
        nodeid2msg=nodeid2msg,
        subject2hash=subject2hash,
        file2hash=file2hash,
        netobj2hash=netobj2hash
    )
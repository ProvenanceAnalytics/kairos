import hashlib
def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

################### 进程节点 ################### 

# 构建进程节点的唯一标识 多个uuid 对一个hash值，  多对一的关系
subjectset=set()
subject2hash={}
entitycount=0

datalist=[]
with open("./node_eventset","r") as f:
    for line in tqdm(f):
        # 转JSON  需要try 包起来，有的不是标准的json串
        if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line :
            res=re.findall('Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',line)[0]
            try:
                path_str=re.findall('"path":"(.*?)"',line)[0] 
                path=path_str
            except:
                path="null"
            nodeid=res[0]
            cmdLine=res[2]
            tgid=res[4]
            nodeproperty=cmdLine+","+tgid+","+path
            hashstr=stringtomd5(nodeproperty)
            subject2hash[nodeid]=[hashstr,nodeproperty]
            subject2hash[hashstr]=nodeid
            subjectset.add(hashstr)
            
            
            
#             print(line)
#             print(nodeproperty)
# for i in subject2hash.keys():
#     if len(i)!=64:
#         datalist.append([i]+subject2hash[i])

# #写入数据库


# sql = '''insert into subject_node_table
#                      values %s
#         '''
# ex.execute_values(cur,sql, datalist,page_size=10000)
# connect.commit()  #需要手动提交


################### 文件节点 ################### 
# 构建文件节点的唯一标识 多个uuid 对一个hash值，  多对一的关系
fileset=set()
file2hash={}# 丢弃了 绝对路径为空的实体，  那么在构建边的时候，遇到不存在的key 就应该跳过当前的event了
nullcount=0
datalist=[]
with open("./node_eventset","r") as f:
    for line in tqdm(f):
        # 转JSON  需要try 包起来，有的不是标准的json串
        if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line :
            try:
            
                res=re.findall('FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"',line)[0]
                nodeid=res[0]
                filepath=res[2]
                nodeproperty=filepath
                hashstr=stringtomd5(nodeproperty)
                file2hash[nodeid]=[hashstr,nodeproperty]
                file2hash[hashstr]=nodeid
                fileset.add(hashstr)
            except:
                # 228786个文件节点是没有路径的，那么关于这种事件应该如何处理呢？ 直接扔掉?
                pass
# for i in file2hash.keys():
#     if len(i)!=64:
#         datalist.append([i]+file2hash[i])

# #写入数据库


# sql = '''insert into file_node_table
#                      values %s
#         '''

# ex.execute_values(cur,sql, datalist,page_size=10000)
# connect.commit()  #需要手动提交


# subject2hash 进程 转hash
# file2hash   文件转hash
# netobj2hash 网络流转hash
datalist=[]
# firsttime=1522764134000000000
with open("./relation_eventset","r") as f:
    for line in tqdm(f):
        time=re.findall('"timestampNanos":(.*?),',line)[0]
        time=int(time) # 将时间转成秒为单位
        subjectid=re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"',line)[0]
        objectid=re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"',line)[0]
        relation_type=re.findall('"type":"(.*?)"',line)[0]
        
      
        if subjectid in subject2hash.keys():
            # 该实体是存在的
            subjectid=subject2hash[subjectid][0]# 将uuid转成节点的hash值
        if objectid in subject2hash.keys():
            objectid=subject2hash[objectid][0]# 将uuid转成节点的hash值
        if objectid in file2hash.keys():
            objectid=file2hash[objectid][0]# 将uuid转成节点的hash值
        if objectid in netobj2hash.keys():
            objectid=netobj2hash[objectid][0]# 将uuid转成节点的hash值
        if len(subjectid)==64 and len(objectid) == 64: # uuid都转成hashid才能被认作成为一条合法的事件 能够作为关系存下来
            if relation_type in ['EVENT_READ','EVENT_READ_SOCKET_PARAMS','EVENT_RECVFROM','EVENT_RECVMSG']:
                # 数据流入到subject中，因此process作为流入节点
                datalist.append((objectid,node2id[objectid],relation_type,subjectid,node2id[subjectid],time))
            else:
                datalist.append((subjectid,node2id[subjectid],relation_type,objectid,node2id[objectid],time))
            
            # 写入数据库       
            if len(datalist)>=10000:
                try:                
                    sql = '''insert into event_table
                                         values %s
                            '''
                    ex.execute_values(cur,sql, datalist,page_size=100000)
                    connect.commit()  #需要手动提交
                    datalist=[]
                except:
                    print("数据库写入失败！")
                    connect.rollback()
                    
        

# # 写入数据库       
# try:                
#     sql = '''insert into event_table
#                          values %s
#             '''
#     ex.execute_values(cur,sql, datalist,page_size=10000)
#     connect.commit()  #需要手动提交 
#     datalist=[]
# except:
#     print("数据库写入失败！")
#     connect.rollback()
    
        
        



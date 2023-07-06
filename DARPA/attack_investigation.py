
# 攻击调查模块

# 一些常见的路径替换  需要根据不同数据集进行调整
replace_dic={
    '/run/shm/':'/run/shm/*',
    '/home/admin/.cache/mozilla/firefox/':'/home/admin/.cache/mozilla/firefox/*',
    '/home/admin/.mozilla/firefox':'/home/admin/.mozilla/firefox*',    
    '/data/replay_logdb/':'/data/replay_logdb/*', 
    '/home/admin/.local/share/applications/':'/home/admin/.local/share/applications/*', 
    '/usr/share/applications/':'/usr/share/applications/*', 
    '/lib/x86_64-linux-gnu/':'/lib/x86_64-linux-gnu/*',     
    '/proc/':'/proc/*', 
    '/stat':'*/stat', 
    '/etc/bash_completion.d/':'/etc/bash_completion.d/*', 
    '/usr/bin/python2.7':'/usr/bin/python2.7/*', 
    '/usr/lib/python2.7':'/usr/lib/python2.7/*', 
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    return path_name


# 异常检测发现的时间窗口
attack_list=[
    # 这里存放有问题的时间窗口的edge list
    #例如 'graph_4_10_without_neg_edge/2018-04-10 13:31:14.548738409~2018-04-10 13:46:36.161065223.txt', 
  ]

# 生成加权图
original_edges_count=0
graphs=[]
gg=nx.DiGraph()
count=0
for path in tqdm(attack_list):
    if ".txt" in path:
        line_count=0
        node_set=set()
        tempg=nx.DiGraph()
        f=open(path,"r")       
        edge_list=[]
        for line in f:
            count+=1
            l=line.strip()
            jdata=eval(l)
            edge_list.append(jdata)
            
        edge_list = sorted(edge_list, key=lambda x:x['loss'],reverse=True) 
        original_edges_count+=len(edge_list)
        
        loss_list=[]
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean=mean(loss_list)
        loss_std=std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr=loss_mean+1.5*loss_std
        print("thr:",thr)
        for e in edge_list:
            if e['loss']>thr:    
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),str(hashgen(replace_path_name(e['dstmsg']))),loss=e['loss'],srcmsg=e['srcmsg'],dstmsg=e['dstmsg'],edge_type=e['edge_type'],time=e['time'])
        

# 社区发现算法
import datetime
import community as community_louvain
starttime = datetime.datetime.now()
#long running
partition = community_louvain.best_partition(gg.to_undirected())
#do something other
endtime = datetime.datetime.now()
print("社区发现计算完毕，消耗时间:{:d}".format((endtime - starttime).seconds))

# 根据社区构建candidate subgraph
communities={}
max_partition=0
for i in partition:
    if partition[i]>max_partition:
        max_partition=partition[i]
for i in range(max_partition+1):
    communities[i]=nx.DiGraph()
for e in gg.edges:
#     if partition[e[0]]==partition[e[1]]:
    communities[partition[e[0]]].add_edge(e[0],e[1])
    communities[partition[e[1]]].add_edge(e[0],e[1])


# 定义attack node，将对应的edge和node在绘制时染色
def attack_edge_flag(msg):
    attack_edge_type=[
        '/home/admin/clean',
        '/dev/glx_alsa_675',
        '/home/admin/profile',
          '/var/log/xdev',
    '/etc/passwd',
    '161.116.88.72',
    '146.153.68.151',
        '/var/log/mail',
        '/tmp/memtrace.so',
        '/var/log/xdev',
         '/var/log/wdev',
        'gtcache',
        'firefox',
#         '/var/log',
    ]
    flag=False
    for i in attack_edge_type:
        if i in msg:
            flag=True
            break
    return flag


# 绘制和渲染candidate subgraph

from graphviz import Digraph


graph_index=0


for c in communities:
    # 将文件、 进程、 网络连接的信息放到节点上面去
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'
    
    for e in communities[c].edges:
        try:
            temp_edge=gg.edges[e]
            srcnode=e['srcnode']
            dstnode=e['dstnode']
        except:
            pass        

        if True:
            subgraph_loss_sum+=temp_edge['loss']
            # 源节点     
            if "'subject': '" in temp_edge['srcmsg']:
                src_shape='box'
            elif "'file': '" in temp_edge['srcmsg']:
                src_shape='oval'
            elif "'netflow': '" in temp_edge['srcmsg']:
                src_shape='diamond'
            if attack_edge_flag(temp_edge['srcmsg']):
                src_node_color='red'
                attack_node_count+=1
            else:
                src_node_color='blue'
            dot.node( name=str(hashgen(replace_path_name_draw(temp_edge['srcmsg']))),label=str(replace_path_name_draw(temp_edge['srcmsg'])+str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])), color=src_node_color,shape = src_shape)

            #目的节点
            if "'subject': '" in temp_edge['dstmsg']:
                dst_shape='box'
            elif "'file': '" in temp_edge['dstmsg']:
                dst_shape='oval'
            elif "'netflow': '" in temp_edge['dstmsg']:
                dst_shape='diamond'
            if attack_edge_flag(temp_edge['dstmsg']):
                dst_node_color='red'
                attack_node_count+=1
            else:
                dst_node_color='blue'
            dot.node( name=str(hashgen(replace_path_name_draw(temp_edge['dstmsg']))),label=str(replace_path_name_draw(temp_edge['dstmsg'])+str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])), color=dst_node_color,shape = dst_shape)

            if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']):
                edge_color='red'
                attack_edge_count+=1
            else:
                edge_color='blue'
            dot.edge(str(hashgen(replace_path_name_draw(temp_edge['srcmsg']))),str(hashgen(replace_path_name_draw(temp_edge['dstmsg']))), label= temp_edge['edge_type'] , color=edge_color)#+ "  loss: "+str(temp_edge['loss']) + "  time: "+str(temp_edge['time'])

    print("开始渲染图像···")
    dot.render('./graph_visual/subgraph_'+str(graph_index), view=False)
    graph_index+=1

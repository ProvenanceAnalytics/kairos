# Environment setting

## Prerequisite Software
1. Anaconda
2. PostgresSQL (installation guild for Ubuntu: https://www.cherryservers.com/blog/how-to-install-and-setup-postgresql-server-on-ubuntu-20-04)

## Setting the database
### CADETS
```commandline
# enter the psql with postgres
sudo -u postgres psql

# create the database
postgres=# create database tc_cadet_dataset_db;

# switch to the created database
postgres=# \connect tc_cadet_dataset_db;

# create the event table and grant the privileges to postgres
tc_cadet_dataset_db=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
tc_cadet_dataset_db=# alter table event_table owner to postgres;
tc_cadet_dataset_db=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
tc_cadet_dataset_db=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_cadet_dataset_db=# alter table file_node_table owner to postgres;

# create the netflow table
tc_cadet_dataset_db=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_cadet_dataset_db=# alter table netflow_node_table owner to postgres;

# create the subject table
tc_cadet_dataset_db=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
tc_cadet_dataset_db=# alter table subject_node_table owner to postgres;

# create the node2id table
tc_cadet_dataset_db=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
tc_cadet_dataset_db=# alter table node2id owner to postgres;
tc_cadet_dataset_db=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```

### THEIA
```commandline
# enter the psql with postgres
sudo -u postgres psql

# create the database
postgres=# create database tc_theia_dataset_db;

# switch to the created database
postgres=# \connect tc_theia_dataset_db;

# create the event table and grant the privileges to postgres
tc_theia_dataset_db=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
tc_theia_dataset_db=# alter table event_table owner to postgres;
tc_theia_dataset_db=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
tc_theia_dataset_db=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_theia_dataset_db=# alter table file_node_table owner to postgres;

# create the netflow table
tc_theia_dataset_db=# create table netflow_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    src_addr  varchar,
    src_port  varchar,
    dst_addr  varchar,
    dst_port  varchar,
    constraint netflow_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_theia_dataset_db=# alter table netflow_node_table owner to postgres;

# create the subject table
tc_theia_dataset_db=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
tc_theia_dataset_db=# alter table subject_node_table owner to postgres;

# create the node2id table
tc_theia_dataset_db=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
tc_theia_dataset_db=# alter table node2id owner to postgres;
tc_theia_dataset_db=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```





```commandline
conda create -n kairos python=3.9
conda activate kairos
conda install psycopg2          # Note that it may fail to install using "pip install psycopg2"
conda install tqdm
pip install scikit-learn==1.2.0     # There may be a problem when using Feature Hashing functions with the version 1.2.2
pip install networkx==2.8.7
pip install pandas


# Pytorch CPU version
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install torch_geometric==2.0.0   # Note that pyg has added new way to load data since version 2.0.4, so current kairos codes won't work starting from this version
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# Pytorch GPU version
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.0.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html



```




# Problems&Solutions:
1. When executing psycopg2.connect(), OperationalError: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  Peer authentication failed for user "postgres".

**Solution**: please refer to https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge

2. When executing psycopg2.connect(), OperationalError: could not connect to server: No such file or directory the server running locally and accepting nections on Unix domain socket "/XXX/.s.PGSQL.5432"?

**Solution steps**:<br>
a) Check if postgres is running. If not start it and run the codes and see if the problem still exists.<br>
b) If the problem still exists when postgres is running, find the location of the file ".s.PGSQL.5432".<br>
c) Set a parameter in psycopg2.connect() with ```host=/the/location/of/the/file/```. Then the problem should be fixed.



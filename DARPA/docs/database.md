# Setting the database
Links to create the corresponding database and tables:

[1. CADETS E3](#CADETS-E3)

[2. THEIA E3](#THEIA-E3)

[3. ClearScope E3](#ClearScope-E3)

[4. CADETS E5](#CADETS-E5)

[5. THEIA E5](#THEIA-E3)

[6. ClearScope E5](#ClearScope-E5)

[7. OpTC](#OpTC)




## CADETS E3 <a name="CADETS-E3"></a>
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


## THEIA E3 <a name="THEIA-E3"></a>
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


## ClearScope E3 <a name="ClearScope-E3"></a>
TBA


## CADETS E5 <a name="CADETS-E5"></a>
```commandline
# enter the psql with postgres
sudo -u postgres psql

# create the database
postgres=# create database tc_e5_cadets_dataset_db;

# switch to the created database
postgres=# \connect tc_e5_cadets_dataset_db;

# create the event table and grant the privileges to postgres
tc_e5_cadets_dataset_db=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
tc_e5_cadets_dataset_db=# alter table event_table owner to postgres;
tc_e5_cadets_dataset_db=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
tc_e5_cadets_dataset_db=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_e5_cadets_dataset_db=# alter table file_node_table owner to postgres;

# create the netflow table
tc_e5_cadets_dataset_db=# create table netflow_node_table
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
tc_e5_cadets_dataset_db=# alter table netflow_node_table owner to postgres;

# create the subject table
tc_e5_cadets_dataset_db=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
tc_e5_cadets_dataset_db=# alter table subject_node_table owner to postgres;

# create the node2id table
tc_e5_cadets_dataset_db=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
tc_e5_cadets_dataset_db=# alter table node2id owner to postgres;
tc_e5_cadets_dataset_db=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```


## THEIA E5 <a name="THEIA-E5"></a>
```commandline
# enter the psql with postgres
sudo -u postgres psql

# create the database
postgres=# create database tc_e5_theia_dataset_db;

# switch to the created database
postgres=# \connect tc_e5_theia_dataset_db;

# create the event table and grant the privileges to postgres
tc_e5_theia_dataset_db=# create table event_table
(
    src_node      varchar,
    src_index_id  varchar,
    operation     varchar,
    dst_node      varchar,
    dst_index_id  varchar,
    timestamp_rec bigint,
    _id           serial
);
tc_e5_theia_dataset_db=# alter table event_table owner to postgres;
tc_e5_theia_dataset_db=# create unique index event_table__id_uindex on event_table (_id); grant delete, insert, references, select, trigger, truncate, update on event_table to postgres;

# create the file table
tc_e5_theia_dataset_db=# create table file_node_table
(
    node_uuid varchar not null,
    hash_id   varchar not null,
    path      varchar,
    constraint file_node_table_pk
        primary key (node_uuid, hash_id)
);
tc_e5_theia_dataset_db=# alter table file_node_table owner to postgres;

# create the netflow table
tc_e5_theia_dataset_db=# create table netflow_node_table
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
tc_e5_theia_dataset_db=# alter table netflow_node_table owner to postgres;

# create the subject table
tc_e5_theia_dataset_db=# create table subject_node_table
(
    node_uuid varchar,
    hash_id   varchar,
    exec      varchar
);
tc_e5_theia_dataset_db=# alter table subject_node_table owner to postgres;

# create the node2id table
tc_e5_theia_dataset_db=# create table node2id
(
    hash_id   varchar not null
        constraint node2id_pk
            primary key,
    node_type varchar,
    msg       varchar,
    index_id  bigint
);
tc_e5_theia_dataset_db=# alter table node2id owner to postgres;
tc_e5_theia_dataset_db=# create unique index node2id_hash_id_uindex on node2id (hash_id);
```






## ClearScope E5 <a name="ClearScope-E5"></a>
TBA


## OpTC <a name="OpTC"></a>
TBA


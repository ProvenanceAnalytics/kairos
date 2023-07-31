# Create the database

Execute SQL: `CREATE DATABASE manzoor_db;`


# Create the table

Execute SQL: 
`create table IF NOT EXISTS raw_data
(
    source_id        varchar,
    source_type      varchar,
    destination_id   varchar,
    destination_type varchar,
    edge_type        varchar,
    graph_id         integer,
    _id              serial
        constraint raw_data_pk
            primary key
);

alter table raw_data
    owner to psql;
`






# StreamSpot

## Environment settings
Please follow the description in the [environment settings](../../DARPA/settings/environment-settings.md) to set up the required environment for Kairos.

## Create the database
Please follow the instructions below to create the database for StreamSpot dataset

```commandline
# execute the psql with postgres user
sudo -u postgres psql

# create the database
postgres=# create database streamspot;

# switch to the created database
postgres=# \connect streamspot;

# create the table and grant the privileges to postgres
streamspot=# create table IF NOT EXISTS raw_data
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

streamspot=# alter table raw_data owner to postgres;
```


## Instructions to run experiments on StreamSpot dataset
TBA







# StreamSpot

## Environment Settings
Follow the description in the [environment settings](../../DARPA/settings/environment-settings.md) to set up the required environment for Kairos.

## Create the database
Use the following commands to create the database for the StreamSpot dataset

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

## Running StreamSpot experiments
Run the commands below to obtain the reconstruction and detection results in `results.txt`. 
> Make sure your machine has enough memory (at least 64GB) to run the experiments.

```commandline
conda activate kairos
cd src
python preprocess.py
python train.py
python test.py > results.txt
python test.py >> results.txt
```

Instead of training the StreamSpot model, 
you can use our pre-trained model [here](https://drive.google.com/drive/u/0/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C)
and skip to testing directly.

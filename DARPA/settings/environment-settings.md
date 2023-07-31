# Environmental settings

## Prerequisites
Although other versions of the applications might also work, we recommend users to use the same versions for better reproducing the experiments. 
1. OS Version: 5.19.0-46-generic #47~22.04.1-Ubuntu
2. Anaconda: 23.3.1
3. PostgresSQL 
   1. Version: 15.3, Ubuntu 15.3-1.pgdg22.04+1  
   2. Installation guild for Ubuntu: https://www.cherryservers.com/blog/how-to-install-and-setup-postgresql-server-on-ubuntu-20-04)
4. GraphViz: 2.43.0 
5. (For reference) GPU info:
   1. Driver Version: 530.41.03
   2. CUDA Version: 12.1

## Python & Libraries
We recommend users to run the instructions below to install the corresponding libraries, but our requirements.txt is also provided for users to check the versions of some libraries.
```commandline
conda create -n kairos python=3.9
conda activate kairos
conda install psycopg2          # Note that it may fail to install using "pip install psycopg2"
conda install tqdm
pip install scikit-learn==1.2.0     # There may be a problem when using Feature Hashing functions with the version 1.2.2
pip install networkx==2.8.7
pip install xxhash==3.2.0
pip install graphviz==0.20.1

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



# DEMO (DARPA CADETS E3)
Here we use CADETS E3 as the demo to help user get familiar with the pipeline of Kairos and reproduce the experiments.

1. Refer to the descriptions in [Environmental settings](docs/Environmental_settings.md) and set up the environments for Kairos.

2. Refer to the [database settings](docs/database.md). Follow the instructions in the CADETS E3 section and create the database for it. 

3. Open the config.py in the CADETS_E3 folder, set the variable ```raw_dir``` as the absolute path of the folder where your raw CADETS E3 data is. In addition, change the database-related variables (e.g. username, password, etc.) based on your database configurations.

4. Run the Kairos pipeline based on the following scripts
```commandline
cd CADETS_E3
make pipeline
```

5. Once the pipeline is finished, the artifacts will be stored in the ```CADETS_E3/artifact/``` folder. The folder structure is shown as below:

- artifact/
    - graphs/
    - graph_4_3
    - graph_4_4/
    - graph_4_5/
    - graph_4_6/
    - graph_4_7/
    - graph_visual/
    - models/
    - embedding.log
    - training.log
    - reconstruction.log
    - anomalous_queue.log
    - evaluation.log
    - some other artifacts

```graphs/```folder contains all the vectorized graphs.

```graph_4_*``` folders contain the reconstruction results of the tested date.

```graph_visual``` folder contains all the summary graphs for attack investigation.

```embedding.log``` records some statistics, e.g., the number of edges in the graphs, during graph vectorization.

```training.log``` records the losses during model training.

```reconstruction.log``` records some reconstruction statistics during testing.

```anomalous_queue.log``` records the anomalous time windows flagged by Kairos.

```evaluation.log``` records the evaluation results for CADETS E3 dataset.



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

```graph_4_*``` folders contain the reconstruction results of the graphs.

```graph_visual``` folder contains all the summary graphs for attack investigation.

```embedding.log``` records some statistics, e.g., the number of edges in the graphs, during graph vectorization.

```training.log``` records the losses during model training.

```reconstruction.log``` records some reconstruction statistics during testing.

```anomalous_queue.log``` records the anomalous time windows flagged by Kairos.

```evaluation.log``` records the evaluation results for CADETS E3 dataset.

6. Note: Kairos detection performance relies on the quality of the trained GNN models. Due to the different initialized weights of the models, GNN models might not be converged using our provided hyperparameters. If the detection results are different from those in the paper, we recommend users to either of each:
   1. Adjust the hyperparameters for model training and adjust the thresholds to achieve the same performance as in the paper.
   2. Use the models we trained for evaluations. To do so, users need to change the path into the path of the downloaded model in the ```test.py```, line 170. Then run the following scripts:
   ```commandline
   make test
   make anomaly_detection
   ```
   Then, users should be able to obtain the same results in the paper with our pre-set thresholds.

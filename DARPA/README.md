# Demo (DARPA CADETS E3)
In this demo, 
we use the CADETS E3 dataset to demonstrate Kairos' end-to-end workflow.
Running this pipeline will reproduce the experimental results reported in our paper.

1. Follow the description in the [environment settings](settings/environment-settings.md) to set up the required environment for Kairos.

2. Follow the description in the [CADETS E3 database settings](settings/database.md#cadets-e3) to create a database for the workload. 

3. Edit CADETS E3's [config.py](CADETS_E3/config.py) to set the variable `raw_dir` as the absolute path of the folder in which your raw CADETS E3 data is located. 
In addition, change the database-related variables (e.g. username, password, etc.) based on your database configurations.

4. Run the Kairos workflow using the commands:
```commandline
cd CADETS_E3
make pipeline
```

5. Once the execution is finished, the artifacts will be stored in the `CADETS_E3/artifact/` folder. The folder structure looks like this:

```
- artifact/
    - graphs/
    - graph_4_3/
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
```
where
* `graphs/` contains all the vectorized graphs.

* `graph_4_*/` contains the reconstruction results of the graphs.

* `graph_visual/` contains all the summary graphs for attack investigation.

* `embedding.log` records some statistics, e.g., the number of edges in the graphs, during graph vectorization.

* `training.log` records the losses during model training.

* `reconstruction.log` records some reconstruction statistics during testing.

* `anomalous_queue.log` records the anomalous time windows flagged by Kairos.

* `evaluation.log` records the evaluation results for the CADETS E3 dataset.


### Using the Pre-trained Model

As expected, 
Kairos detection performance relies on the quality of the trained GNN models,
but model training takes a significant amount of time.
You can skip training and directly use our pre-trained models
for quick evaluations.
To do so,
simply provide the file path of the pre-trained model
(which you can download from [here](https://drive.google.com/drive/u/0/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C))
in `test.py` (in [this](https://github.com/ProvenanceAnalytics/kairos/blob/37044bfd30393c0a0543d3b98f2049cd039cc013/DARPA/CADETS_E3/test.py#L170) line of code) and then run:
```commandline
make test
make anomaly_detection
```

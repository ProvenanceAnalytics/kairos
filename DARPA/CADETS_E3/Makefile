prepare:
	mkdir -p ./artifact/

create_database:
	python create_database.py

embed_graphs:
	python embedding.py

train:
	python train.py

test:
	python test.py

anomalous_queue:
	python anomalous_queue_construction.py

evaluation:
	python evaluation.py

attack_investigation:
	python attack_investigation.py

preprocess: prepare create_database embed_graphs

deep_graph_learning: train test

anomaly_detection: anomalous_queue evaluation

pipeline: preprocess deep_graph_learning anomaly_detection attack_investigation


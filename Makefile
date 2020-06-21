install:
	pip install -r requirements.txt

run-demo:
	python sacsann.py train_and_predict data/features mm --labels_path data/labels --train_chromosomes 1	--test_chromosomes 2

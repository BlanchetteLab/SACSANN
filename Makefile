install:
	pip install -r requirements.txt

run-demo:
	python sacsann.py data/labels data/features mm --train_chromosomes 1 --test_chromosomes 2

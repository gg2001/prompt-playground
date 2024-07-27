test:
	PYTHONPATH=. pytest tests/test_embeddings.py -s

env:
	conda env export > environment.yml

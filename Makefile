test:
	PYTHONPATH=. pytest tests/test_embeddings.py -s

env:
	conda env export --no-builds | sed 's|^name: .*|name: prompt-playground|' | grep -v "^prefix: " > environment.yml

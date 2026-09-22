.PHONY: run charts test clean

run:
	python -m rlsub.train

charts:
	python -m rlsub.charts

test:
	python -m pytest tests/ -q

clean:
	rm -rf charts __pycache__ */__pycache__ .pytest_cache

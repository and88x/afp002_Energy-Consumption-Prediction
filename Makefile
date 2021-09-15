PY=python 
.PHONY:
    run
    all
    typehint
    test
    lint
    black
    #clean
    install

run:
	#$(PY) -i src/main.py 
	jupyter-lab

all:
	@+make black
	@+make typehint
	@+make lint
	#@+make test
	#@+make clean
	@+make run

typehint:
	mypy --ignore-missing-imports src/

test:
	# pytest tests/
	$(PY) src/runner.py

lint:
	pylint src/

black:
	black -l 79 src/*.py

clean:
	find . -type f -name "*.pyc" | xargs rm -fr 
	find . -type d -name __pycache__ | xargs rm -fr

install:
	pip install -U mypy black pylint pytest pandas pyarrow matplotlib plotly==5.1.0 kaleido dash==1.21.0 statsmodels==0.12.2 "jupyterlab>=3" "ipywidgets>=7.6" scipy cesium==0.9.12 xgboost==1.4.2 "jupyterlab-kite>=2.0.2" pandas-datareader
	# solve the issue with https://github.com/cesium-ml/cesium/issues/275
	conda install ipykernel
	$(PY) -m ipykernel install --user --name afp002

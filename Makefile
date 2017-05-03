author=$(Ge Yang)
author_email=$(yangge1987@gmail.com)

default:
	make install
	make setup-vis-server
	make train
install:
	pip install -r requirement.txt
setup-vis-server:
	python -m visdom.server > visdom.log 2>&1 &
	sleep 0.5s
	open http://localhost:8097/env/Variational-Autoencoder-experiment
train:
	python vae_mnist.py
evaluate:
	python vae_mnist.py

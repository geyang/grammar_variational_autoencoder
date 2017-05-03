# Grammar Variational Autoencoder (implementation in pyTorch) [![](https://img.shields.io/badge/link_on-GitHub-brightgreen.svg?style=flat-square)](https://github.com/episodeyang/grammar_variational_autoencoder)

### Todo

- [ ] data 
- [ ] model
- [ ] train 
- [ ] think of a demo
- [ ] closer look at the paper

## Usage (To Run)
1. install dependencies via
    ```bash
    pip install -r requirement.txt
    ```
2. Fire up a `visdom` server instance to show the visualizations. Run in a dedicated prompt to keep this alive.
    ```bash
    python -m visdom.server
    ```
3. In a new prompt run
    ```bash
    python vae_mnist.py
    ```

Or for with a quick shortcut, you can just run `make`. You can take a look at
the [`./Makefile`](./Makefile) for more details.
    
## Grammar Variational Autoencoder (VAE) and Variational Bayesian methods




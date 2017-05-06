# Grammar Variational Autoencoder (implementation in pyTorch) [![](https://img.shields.io/badge/link_on-GitHub-brightgreen.svg?style=flat-square)](https://github.com/episodeyang/grammar_variational_autoencoder)

### Todo

- [ ] what type of accuracy metric do we use?
- [ ] train 
    - [ ] encoder convolution exact configuration
    - [ ] read dynamic convolutional network 
        - [ ] what are the evaluation metrics in DCNN?
            - [ ] sentiment analysis
            - [ ] 
- [ ] think of a demo
- [ ] closer look at the paper

#### Done
- [x] data 
- [x] model

## Usage (To Run)

All of the script bellow are included in the [`./Makefile`](./Makefile). To install and run training, 
you can just run `make`. For more details, take a look at the `./Makefile`.

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
    python grammar_vae.py
    ```
    
## Grammar Variational Autoencoder (VAE) and Variational Bayesian methods




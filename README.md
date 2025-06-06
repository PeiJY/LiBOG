# LiBOG

[![ArXiv](https://img.shields.io/badge/arXiv-2505.13025-b31b1b.svg)](https://arxiv.org/abs/2505.13025)

Official implementation of our paper **"LiBOG: Lifelong Learning for Black-Box Optimizer Generation"**, which is accepted at *IJCAI* 2025. In this work, we tackle the problem of **lifelong learning from a sequence of BBO problem distributions to generate high-performance BBO optimizer** using a novel **LiBOG**, which achieves **good forward transferring, mild catastrophic forgetting and stable learning**.


![LiBOG](./LiBOG.png)



## Citing

LiBOG: Lifelong Learning for Black-Box Optimizer Generation, Jiyuan Pei, Yi Mei, Jialin Liu and Mengjie Zhang. Accepted at 34th International Joint Conference on Artificial Intelligence (IJCAI) 2025.

```
@inproceedings{LiBOG,
author={Pei, Jiyuan and Mei, Yi, and Liu, Jialin and Zhang, Mengjie},
title={LiBOG: Lifelong Learning for Black-Box Optimizer Generation},
booktitle = {34th International Joint Conference on Artificial Intelligence},
year={2025},
}
```





## Repository Structure

```
├── run_lifelong.py         # Entry point for training/testing
├── options.py              # Running options for training/testing
├── execute/                # Function code for training/testing
├── model/                  # Model definitions
├── expr/                   # Code for expression
├── dataset/                # Dataset loading and preprocessing of BBO problems
├── env/                    # RL environment of BBO optimizer for solving problems
├── utils/                  # Utility functions and helpers
├── population/             # About solution population
├── pbo_env/                # Classic BBO optimizers
├── requirements.txt        # Python dependencies
└── README.md               # This file
```


**Built upon**: [Symbol](https://github.com/MetaEvo/Symbol) – used and extended with MIT License

## Requirements

The dependencies of this project are listed in requirements.txt. You can install them using the following command.
```
pip install -r requirements.txt
```

## Quick Start
```
python run_lifelong.py --train --ll_training_method LiBOG --run_name test_code
```

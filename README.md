# Intrusion Detection System using a custom Neural Network
This project implements a custom neural network using PyTorch and trains it on [NSL-KDD dataset](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD).

## Setup and Installation

- Clone the repository.
- This project using [uv](https://github.com/astral-sh/uv) for dependency and package management. Your machine must have it installed as a prerequisite. Once the repository is cloned, you can setup the project at once by running the command `uv sync` inside the directory.
- To run the training pipeline, run the command `uv run src/train.py` from the project root directory.
- You can set the hyperparameters and models to be tested in `src/constants.py`.
- To test the model, run the command `uv run src/test.py` from the project root directory.

The model trained with current hyperparameters is able to achieve a F1 Score of **0.7** on the test split.

Any kind of PRs and suggestions for improvements are extremely welcomed.
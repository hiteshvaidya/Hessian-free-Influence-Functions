# An approximation of Influence function in Deep Neural Networks using Fisher Information Matrix
Influence functions are used to obtain the importance of an input sample in learning an input-output mapping by neural networks. Through this value, we can estimate the effect over performance of a network after removing a certain sample from the training set. Thus, helping us to train a neural network on only the most significant/important data samples. This can be used to store the most important input samples in data buffers in continual learning or even, removing biases or unlearning in Large Language Models.

## Reference
- [https://arxiv.org/pdf/2405.03869](https://arxiv.org/pdf/2405.03869)
- [https://proceedings.mlr.press/v89/karakida19a/karakida19a.pdf](https://proceedings.mlr.press/v89/karakida19a/karakida19a.pdf)

## Objective
Influence functions are calculated using inverse of Hessian matrix. For a large parameter space, this leads to higher computational complexity. Therefore, we need an efficient way to approximate the influence function without having to calculate the hessian matrix. Through this project, we test the accuracy and robustness of an approximation of Influence function using *Fisher Information Matrix* instead of a Hessian matrix.


## Code
Create a [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment and install all the dependencies as follows,
```
conda create -n if python=3.12
conda activate if
pip install -r requirements.txt
python -m ipykernel install --user --name=if --display-name="Python-if"
```

open the jupyter notebooks and ensure that you are using the `if` conda environment created earlier in the form of python kernel.

## Datasets
- [penn treebank](https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html) - [paperswithcode](https://paperswithcode.com/dataset/penn-treebank)
- [text-8](https://www.kaggle.com/datasets/gupta24789/text8-word-embedding)

## Files (under construction)
- training a simple LSTM on penn treebank dataset - [train_lstm_penn_treebank.ipynb](train_lstm_penn_treebank.ipynb)
- Testing Influence functions with Fisher Information matrix - [fisher_if.ipynb](fisher_if.ipynb)


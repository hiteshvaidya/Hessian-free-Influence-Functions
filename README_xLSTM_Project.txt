xLSTM Language Model – Grid Search and Extended Training
=========================================================

This project implements a custom LSTM-based language model (xLSTMPyTorch) trained on the Penn Treebank (PTB) dataset. It performs grid search over various hyperparameter configurations to identify the best-performing model and continues training the best one for additional epochs. The project is designed to support downstream applications including influence function and Fisher Information Matrix analysis.

Features
--------
- Custom xLSTM implementation with optional LayerNorm
- Word-level tokenization using TorchText and the PTB dataset
- Grid search over 27 hyperparameter combinations:
    - embed_size ∈ [64, 128, 256]
    - hidden_size ∈ [128, 256, 512]
    - learning_rate ∈ [0.001, 0.005, 0.01]
- Evaluation metrics per epoch: Accuracy, Cross-entropy Loss, Perplexity
- Checkpointing of best models by accuracy
- Final continued training of best model for 50 additional epochs
- Designed for compatibility with Influence Function and Fisher Information Matrix analysis

Dependencies
------------
- Python 3.10+
- PyTorch
- TorchText
- tqdm

Install dependencies:
    pip install torch torchtext tqdm

Dataset
-------
- Penn Treebank (PTB) from TorchText
- Preprocessed using basic English tokenizer
- Sequence length: 30
- No separate validation/test split is included (can be added if needed)

File Structure
--------------
- NLP_Project_27x5_1x50.py: Main training script
  - Performs grid search (5 epochs × 27 configurations)
  - Saves best model from each config as best_xlstm_embed{e}_hidden{h}_lr{lr}.pth
  - Selects best overall configuration and trains it for 50 more epochs
  - Final model is saved as final_continued_model.pth

Usage
-----
Run the training script:
    python NLP_Project_27x5_1x50.py

After completion:
- The top-performing model from grid search is saved as .pth files.
- The final continued training model is saved as final_continued_model.pth.

Output
------
Each run prints:
- Epoch-wise training metrics
- Summary of best parameters from grid search
- Evaluation metrics for each continued training epoch

Example:
    Epoch 1 | Loss: 4.5221 | Accuracy: 0.1952 | Perplexity: 91.78
    ...
    Final model saved as: final_continued_model.pth

Next Steps: Influence Function & FIM
------------------------------------
This code is ready for influence function analysis and Fisher Information Matrix-based evaluation:
- loss.backward(create_graph=True) is already enabled
- Model checkpoints can be loaded and used for gradient/Hessian estimation
- Final weights are saved for reproducibility and downstream robustness testing

License
-------
This project is for educational and research purposes only. No license is currently specified.
# English-Vietnamese-Machine-Translation
## 1. Train RNN models
### 1.1. How to train the RNN-based Encoder Decoder:
- Step 1: Upload the notebook `rnn-based-encoder-decoder.ipynb` to Kaggle. We recommend you to use GPU P100 for training.
- Step 2: Add the input data `phomt-dl-2023` to the notebook.
- Step 3: Set the parameters of the dictionary `config` as follow:
    - `train_mode : True`
    - `translate_mode: False`
    - `eval_mode: False`
    - `small_train_data`: the number of data you want to train, we recommend to set 20000.
    - `epoch:` the number of epoch you want to train.
    - `batch_size`: the batch size you want to use. If the `small_train_data` is 20000, the `batch_size` should be 128.
    - `initial_model`: the path of the initial model to load the old model for continuing training. If you train from the the beginning, set = `None`.


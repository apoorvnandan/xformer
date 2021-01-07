# xformer
Simple and Flexible API for training transformer models on sequence to sequence tasks in Pytorch.

## Installation

```text
pip install git+https://github.com/apoorvnandan/xformer.git
```

## Contents
- [Purpose](#Purpose)
- [Current Scope](#Current-Scope)
- [Minimal Examples](#Minimal-Examples)
- [Design](#Design)
- [Custom Callback](#Custom-Callback)
- [Dev Notes](#Dev-Notes)
- [Acknowledgements](#Acknowledgements)

## Purpose: 
Have a simple interface built on top of pytorch for modelling any sequence to sequence problems with transformer models. The models have built in functions for training and generating outputs. (only with greedy decoding at the moment) And we have callbacks for use case specific customizations.

Check our minimal examples below!

## Current scope:

Input:  
- Case 1: Input is a sequence of token indices (shape: (batch_size,input_len), type: long)
- Case 2: Input is a sequence of feature vectors (shape: (batch_size,input_len,input_dim), type: float)

Output:
- Output is a sequence of token indices (shape: (batch_size,input_len), type: long)
    
This covers a lot of popular applications of sequence to sequence models:

- Machine Translation
- Abstractive Summarisation
- Automatic Speeech Recognition
- Video captioning
- Chatbots

We also provide built-in callbacks that help with the above use cases. 
You can also write a custom callback to execute your code at any point in the training loop.

## Minimal Examples

Machine Translation:
```python
from my_data import (
    src_tokenizer, 
    trg_tokenizer, 
    train_loader, 
    test_loader
)

from xformer import Transformer
from xformer.callbacks import BleuScore

model = Transformer.from_config(
    num_classes=trg_tokenizer.size(),
    embed_input=True,
    input_vocab_size=src_tokenizer.size(),
    src_pad_idx=src_tokenizer.pad_token_idx,
    trg_pad_idx=trg_tokenizer.pad_token_idx
)
bleu_cb = BleuScore(
    trg_tokenizer, test_loader, trg_tokenizer.bos_token_idx, trg_tokenizer.eos_token_idx
)
model.fit(train_loader, n_epochs=2, callbacks=[bleu_cb])
```
```text
Epoch 1/2
1070/1070 [==============================] - 279s 261ms/step - loss: 0.7809
test bleu score: 84.12
Epoch 2/2
1070/1070 [==============================] - 266s 248ms/step - loss: 0.1499
test bleu score: 88.49
```
Speech to text: (work in progress)
```python
from my_data import ( 
    trg_tokenizer, 
    train_loader, 
    test_loader,
    input_dim
)
from xformer import Transformer
from xformer.callbacks import TokenErrorRate

model = Transformer.from_config(
    num_classes=trg_tokenizer.size(),
    input_dim=input_dim
)
cb = TokenErrorRateCallback(test_loader, trg_tokenizer.bos_token_idx, trg_tokenizer.eos_token_idx)
model.fit(train_loader, n_epochs=2, callback=[cb])
```
```text
Epoch 1/2
200/200 [==============================] - 279s 261ms/step - loss: 0.7809
test token error rate: 40.47
Epoch 2/2
200/200 [==============================] - 266s 248ms/step - loss: 0.1499
test token error rate: 33.24
```

## Design
Heavily inspired by huggingface, keras and built on top of pytorch. The idea is to only write the necessary details about your model and data, and train a transformer model without any extra code. Callbacks provide an easy way to make the training code flexible. 

All the models contain the entire architecture as a `nn.Module` so that you can easily use them in other setups. (eg. using the encoder part to extract input representations and using them for a classification task)

The progress bars have been coded using `tf.keras.utils.Progbar`.

## Custom Callback
Callbacks allow you to execute your code at the following points in the training loop.
- Before the training loop starts - by overriding `on_train_start(self, model)`
- At the start of every epoch - by overriding `on_epoch_start(self, model)`
- At the end of every epoch - by overriding `on_epoch_end(self, model)`
- When the training ends - by overriding `on_fit_end(self, model)`

To create a callback, simply extend the base `xformer.callbacks.Callback` class, and override the appropriate methods. Each of these methods have an argument: `model`. This argument contains the model object, so you can use it for everything you can use the original model object for.

Example: Callback for printing out the output of few inputs at the every epoch.

```python
# Step 1: Extend base Callback class
from xformer.callbacks import Callback

class ExactMatchAccuracy(Callback):

# Step 2: Initialise with necessary objects to execute custom code
    def __init__(self, input_batch, trg_tokenizer):
        self.batch = input_batch
        self.tokenizer = trg_tokenizer
        
# Step 3: Override appropriate method to execute your code in the training loop
# The argument 'model' is the transformer model object being trained.
    def on_epoch_end(self, model):
        model.model.eval()  # model.model contains the `nn.Module`
        enc_inp = self.batch['enc_inp']
        dec_out = self.batch['dec_out']
        bs = batch['enc_inp'].shape[0]
        trg_bos_idx = self.tokenizer.trg_bos_idx
        trg_eos_idx = self.tokenizer.trg_eos_idx
        for i in range(bs):
            preds,_ = model.generate(enc_inp[i,:], trg_bos_idx, trg_eos_idx)
            pred_text = self.tokenizer.decode(preds)
            target_text = self.tokenizer.decode(dec_out[i,:].numpy())
            pad_token = self.tokenizer.idx_to_token[self.tokenizer.pad_token_idx]
            target_text = target_text.replace(pad_token,'').strip()  # remove pads
            print('output:', pred_text)
            print('target:', target_text)
```
## Dev Notes
- [x] Write minimum usable version with core functionality
- [x] Test machine translation with small Eng-French dataset.
- [ ] Write API Reference
- [ ] Test speech to text
- [ ] Improve code quality and docstrings
- [ ] Put package on pip
- [ ] Multi GPU support
- [ ] Support for different schedulers and optimizers
- [ ] Upload some pretrained models

## Acknowledgements
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop)
- [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
- [Attention is all you need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [🤗 Transformers](https://github.com/huggingface/transformers)
- [Keras](https://github.com/keras-team/keras)

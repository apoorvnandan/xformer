# xformer
Simple and Flexible API for training transformer models on sequence to sequence tasks.

## Installation

```text
pip install git+https://github.com/apoorvnandan/xformer.git
```

## Contents
- [Purpose](##Purpose)
- [Current Scope](##Current-Scope)
- [Minimal Examples](##Minimal-Examples)
- [Design](##Design)
- [Custom Callback](##Custom-Callback)
- [Dev Notes](##Dev-Notes)
- [Acknowledgements](##Acknowledgements)

## Purpose: 
Have a simple interface built on top of pytorch for modelling any sequence to sequence problems with transformer models. Check our minimal examples below!

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
Heavily inspired by huggingface, keras and sklearn. The idea is to only write the necessary details about your model and data, and train a transformer model without any extra code. Callbacks provide an easy way to make the training code flexible. The progress bars displayed during training look exactly like those in keras `model.fit` because they have been coded using `tf.keras.utils.Progbar`.

## Custom Callback
```python
# Step 1: Extend base Callback class
from xformer.callbacks import Callback

class ExactMatchAccuracy(Callback):

# Step 2: Initialise with necessary objects to execute custom code
    def __init__(self, test_dl, trg_bos_idx, trg_eos_idx):
        self.test_dl = test_dl
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        
# Step 3: Override appropriate method to execute custom code in the training loop
#    The argument 'model' is the transformer model object being trained.
    def on_epoch_end(self, model):
        model.model.eval()
        test_dl = self.test_dl
        correct = 0
        total = 0
        for idx, batch in enumerate(test_dl):
            enc_inp = batch['enc_inp']
            bs = batch['enc_inp'].shape[0]
            N = batch['dec_out'].shape[1]
            
            for i in range(bs):
                preds,_ = model.generate(enc_inp[i,:], self.trg_bos_idx, self.trg_eos_idx)
                preds = torch.tensor(preds, dtype=torch.long)
                targets = batch['dec_out'][i,:].cpu().numpy()
                flag = True
                if targets.shape != preds.shape:
                    flag = False
                else:
                    for t in range(N):
                        if targets[t] != -100 and targets[t] != preds[t]:
                            flag = False
                            break
                if flag:
                    correct += 1
                total += 1
        print(f'test acc: {correct/total:.4f} (exact sequence match)')
```
## Dev Notes
- [x] Write minimum usable version with core functionality
- [x] Test machine translation with small Eng-French dataset.
- [ ] Write API Reference
- [ ] Test speech to text with LibriSpeech
- [ ] Improve code quality and docstrings
- [ ] Put package on pip
- [ ] Multi GPU suppor
- [ ] Upload some pretrained models

## Acknowledgments
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop)
- [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
- [Attention is all you need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [🤗 Transformers](https://github.com/huggingface/transformers)
- [Keras](https://github.com/keras-team/keras)

import torch

class Callback:
    def on_train_start(self, model): pass
    
    def on_epoch_end(self, model): pass
    
    def on_fit_end(self, model): pass
    
    def on_epoch_start(self, model): pass
    
    
class ExactMatchAccuracy(Callback):
    """This callback can be used to calculate exact match accuracy at the end of every
    epoch. This works only when the input is a sequence of token indices and not a
    a sequence of feature vector.
    Arguments:
    - test_dl: test data loader
    - trg_bos_idx: start token index from target vocabulary
    - trg_eos_idx: end token index from target vocabulary
    """
    def __init__(self, test_dl, trg_bos_idx, trg_eos_idx):
        self.test_dl = test_dl
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        
    def on_epoch_end(self, model):
        model.model.eval()
        test_dl = self.test_dl
        correct = 0
        total = 0
        for idx, batch in enumerate(test_dl):
            enc_inp = batch['enc_inp'] #.to(model.device)
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
        
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
        
class BleuScore(Callback):
    def __init__(self, tokenizer, test_dl, trg_bos_idx, trg_eos_idx):
        self.tokenizer = tokenizer
        self.test_dl = test_dl
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        
    def bleu_score(self, candidate, references):
        ref_words = [references.split()]
        can_words = candidate.split()
        return sentence_bleu(ref_words, can_words)
    
    def on_epoch_end(self, model):
        model.model.eval()
        test_dl = self.test_dl
        score_list = []
        for idx, batch in enumerate(test_dl):
            enc_inp = batch['enc_inp'] #.to(model.device)
            dec_out = batch['dec_out']
            bs = batch['enc_inp'].shape[0]
            N = batch['dec_out'].shape[1]
            
            for i in range(bs):
                preds,_ = model.generate(enc_inp[i,:], self.trg_bos_idx, self.trg_eos_idx)
                pred_text = self.tokenizer.decode(preds)
                target_text = self.tokenizer.decode(dec_out[i,:].numpy())
                pad_token = self.tokenizer.idx_to_token[self.tokenizer.pad_token_idx]
                target_text = target_text.replace(pad_token,'').strip()
                bleu = self.bleu_score(pred_text, target_text)
                score_list.append(bleu)
        print(f'test bleu score: {np.mean(score_list)*100:.2f}')
        
        
class DisplayFewOutputs(Callback):
    def __init__(self, input_batch, tokenizer, trg_bos_idx, trg_eos_idx, n=5):
        self.input_batch = input_batch
        self.n = n
        self.tokenizer = tokenizer
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        
    def on_epoch_end(self, model):
        model.model.eval()
        batch = self.input_batch
        enc_inp = batch['enc_inp'] #.to(model.device)
        dec_out = batch['dec_out']
        bs = batch['enc_inp'].shape[0]
        N = batch['dec_out'].shape[1]
        
        for i in range(self.n):
            preds,_ = model.generate(enc_inp[i,:], self.trg_bos_idx, self.trg_eos_idx)
            pred_text = self.tokenizer.decode(preds)
            target_text = self.tokenizer.decode(dec_out[i,:].numpy())
            pad_token = self.tokenizer.idx_to_token[self.tokenizer.pad_token_idx]
            target_text = target_text.replace(pad_token,'').strip()
            print('output:', pred_text)
            print('target:', target_text)
            print()

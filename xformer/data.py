import numpy as np
import random
import torch

class AdditionDataset:
    def __init__(self, ndigit, split, len=10000, bs=128, start_end_tokens=True):
        self.start_end_tokens = start_end_tokens
        self.len = len
        self.bs = bs
        self.split = split # train/test
        self.ndigit = ndigit
        self.vocab_size = 10 # 10 possible digits 0..9
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = ndigit + ndigit + ndigit + 1 - 1
        
        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        random.seed(42) 
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.len#self.ixes.size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError
        # given a problem index idx, first recover the associated a + b
        X = []
        Y = []
        for bi in range(self.bs):
            rand_idx = random.randrange(0,self.ixes.size)
            idx = self.ixes[rand_idx]
            nd = 10**self.ndigit
            a = idx // nd
            b = idx %  nd
            c = a + b

            render_x = f'%0{self.ndigit}d%0{self.ndigit}d' % (a,b)
            render_y = f'%0{self.ndigit+1}d' % c
            int_x = [int(s) for s in render_x]
            int_y = [11] + [int(s) for s in render_y] + [12]

            X.append(torch.tensor(int_x, dtype=torch.long))
            Y.append(torch.tensor(int_y, dtype=torch.long))


        return {
            "enc_inp": torch.stack(X),
            "dec_out": torch.stack(Y)
        }

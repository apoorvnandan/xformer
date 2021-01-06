class CharTokenizer:
    def __init__(self):
        self.idx_to_token = ['[pad]','[bos]','[eos]','[unk]']
        self.pad_token_idx = 0
        self.bos_token_idx = 1
        self.eos_token_idx = 2
        self.unk_token_idx = 3
        self.create_token_idx_map()
        
    def size(self):
        return len(self.idx_to_token)
        
    def create_token_idx_map(self):
        self.token_to_idx = {}
        for i,t in enumerate(self.idx_to_token):
            self.token_to_idx[t] = i
    
    def fit(self, texts):
        for t in texts:
            for ch in t:
                if ch not in self.idx_to_token:
                    self.idx_to_token.append(ch)
        self.create_token_idx_map()
    
    def encode(self, t, add_boundary_tags=True):
        if type(t) == list:
            ret = []
            for _ in t:
                ret.append(self.encode_str(_, add_boundary_tags))
            return ret
        elif type(t) == str:
            return self.encode_str(t, add_boundary_tags)
        else:
            raise ValueError(f"arg should be either List[str] or str")
        
    def encode_str(self, s, add_boundary_tags):
        if add_boundary_tags:
            ret = [self.bos_token_idx]
        else:
            ret = []
        for ch in s:
            ret.append(self.token_to_idx.get(ch, self.unk_token_idx))
        if add_boundary_tags:
            ret.append(self.eos_token_idx)
        return ret
    
    def decode(self, t):
        if len(t) == 0:
            return []
        if type(t[0]) == list:
            return [self.decode_line(s) for s in t]
        return self.decode_line(t)
    
    def decode_line(self, s):
        ret = ''
        for tok in s:
            ret += self.idx_to_token[tok]
        return ret
    
class WordTokenizer:
    def __init__(self):
        self.idx_to_token = ['[pad]','[bos]','[eos]','[unk]']
        self.pad_token_idx = 0
        self.bos_token_idx = 1
        self.eos_token_idx = 2
        self.unk_token_idx = 3
        self.create_token_idx_map()
        
    def size(self):
        return len(self.idx_to_token)
        
    def create_token_idx_map(self):
        self.token_to_idx = {}
        for i,t in enumerate(self.idx_to_token):
            self.token_to_idx[t] = i
    
    def fit(self, texts):
        for t in texts:
            t = t.strip()
            for ch in t.split():
                if ch not in self.idx_to_token:
                    self.idx_to_token.append(ch)
        self.create_token_idx_map()
    
    def encode(self, t, add_boundary_tags=True):
        if type(t) == list:
            ret = []
            for _ in t:
                ret.append(self.encode_str(_, add_boundary_tags))
            return ret
        elif type(t) == str:
            return self.encode_str(t, add_boundary_tags)
        else:
            raise ValueError(f"arg should be either List[str] or str")
        
    def encode_str(self, s, add_boundary_tags):
        s = s.strip()
        if add_boundary_tags:
            ret = [self.bos_token_idx]
        else:
            ret = []
        for ch in s.split():
            ret.append(self.token_to_idx.get(ch, self.unk_token_idx))
        if add_boundary_tags:
            ret.append(self.eos_token_idx)
        return ret
    
    def decode(self, t):
        if len(t) == 0:
            return []
        if type(t[0]) == list:
            return [self.decode_line(s) for s in t]
        return self.decode_line(t)
    
    def decode_line(self, s):
        ret = []
        for tok in s:
            ret.append(self.idx_to_token[tok])
        return ' '.join(ret)

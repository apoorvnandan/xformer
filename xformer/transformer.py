import tensorflow as tf


class BaseModel:
    def on_train_start(self): pass
    
    def on_epoch_end(self): pass
    
    def on_fit_end(self): pass
    
    def on_epoch_start(self): pass
                        
    def fit(self, dl, n_epochs=1, callbacks=None):
        self.dl = dl
        self.n_epochs = n_epochs
        self.on_train_start()
        if callbacks is not None:
            for cb in callbacks:
                cb.on_train_start(self)
        for epoch in range(n_epochs):
            self.on_epoch_start()
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_start(self)
            self.epoch = epoch
            self.n_batches = len(dl)
            print(f'Epoch {epoch+1}/{n_epochs}')
            pbar = tf.keras.utils.Progbar(target=self.n_batches)
            for idx, batch in enumerate(dl):
                self.batch_idx = idx
                loss_dict = self.train_step(epoch, idx, batch) 
                pbar.update(idx, values=list(loss_dict.items()))
            pbar.update(self.n_batches, values=None)
            self.on_epoch_end()
            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(self)
        self.on_fit_end()
        if callbacks is not None:
            for cb in callbacks:
                cb.on_fit_end(self)
    
    
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hid_dim, 
        n_layers, 
        n_heads, 
        pf_dim,
        dropout, 
        device,
        max_length = 100,
        token_embedding=True,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.device = device
        if token_embedding:
            self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        else:
            self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        #src = One of: [batch size, src len], [batch size, src len, input dim]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, src len]
        src_emb = self.tok_embedding(src)
        src = self.dropout((src_emb * self.scale) + self.pos_embedding(pos))
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
        #src = [batch size, src len, hid dim]     
        return src
    
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention
    
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
    
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = One of : [batch size, src len], [batch size, src len, input dim]
        if len(src.shape) == 2:
            src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        elif len(src.shape) == 3:
            src_mask = (src[:,:,0] != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f'src has {len(src.shape)} dims')
        
        return src_mask #src_mask = [batch size, 1, 1, src len]
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = One of : [batch size, src len], [batch size, src len, input dim]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    
    
class TransformerModel(BaseModel):
    def __init__(self, model, device, trg_pad_idx):
        super().__init__()
        self.model = model
        self.model.to(device)  # :TODO: multi gpu termination
        self.device = device
        self.loss = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_step(self, epoch, idx, batch):
        self.model.train()
        x = batch['enc_inp'].to(self.device)
        y = batch['dec_out'].to(self.device)
        output, _ = self.model(x, y[:,:-1])
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        y = y[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = self.loss(output, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"loss": loss.item()}
    
    def generate(self, x, trg_bos_idx, trg_eos_idx, max_len=512):
        """ Returns list of output token indices """
        # x = One of [input_len] or [input len, input dim]
        device = self.device
        self.model.eval()
        src_indexes = x
        if len(src_indexes.shape) == 1:
            src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        else:
            src_tensor = torch.tensor(src_indexes, dtype=torch.float).unsqueeze(0).to(device)
        src_mask = self.model.make_src_mask(src_tensor)#.to(device)
        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_bos_idx]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = self.model.make_trg_mask(trg_tensor).to(device)
            with torch.no_grad():
                output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            output = output.cpu()
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == trg_eos_idx:
                break
        return trg_indexes, output.numpy()
    
    def _generate_batch(x, trg_bos_idx, trg_eos_idx, max_len=512):
        # :TODO: batch inference
        pass

        
class Transformer:
    """ This class can be used to initialize a transformer sequence to sequence model.
    The model can be used in the following cases:
    Input:
    Case 1: Input is a sequence of token indices (shape: (batch_size,input_len), type: long)
    Case 2: Input is a sequence of feature vectors (shape: (batch_size,input_len,input_dim), type: float)
    Output:
    - Output is a sequence of token indices (shape: (batch_size,input_len), type: long)
    
    Method: from_config(*args, **kwrags)
    Arguments:
    ----------
    - num_classes (int): number of possible classes at each time step of output sequence
    - hidden_dim (int, default=64): hidden_dim for transformer
    - embed_input (bool, default=False): Set to True if input is a sequence of token indices.
    - input_vocab_size (int, default=0): Set to vocab size of input tokens in Case 1 (above)
    - input_dim (int, default=32): Size of each time step in input sequence in Case 2 (above)
    - max_len (int, default=512): Max length of input sequence
    - src_pad_idx (int, default=-100): Set to the index of pad token 
        in input sequence in Case 1 (above) and pad value for input tensor in Case 2 (above)
    - trg_pad_idx (int, default=-100): Set to the index of pad token 
        in output sequence
        
    Inputs batch: The batch should be a dictionary with two keys:
    - enc_inp: tensor containing input sequence batch
    - dec_out: tensor containing output sequence batch
    """
    @staticmethod
    def from_config(
        num_classes,
        hidden_dim=64,
        num_enc_layers=2,
        num_dec_layers=2,
        num_enc_heads=2,
        num_dec_heads=2,
        enc_fc_dim=128,
        dec_fc_dim=128,
        enc_dropout=0.1,
        dec_dropout=0.1,
        embed_input=False,
        input_vocab_size=0,
        input_dim=32,
        max_len=512,
        src_pad_idx=-100,
        trg_pad_idx=-100
    ):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        enc = Encoder(
            input_dim=input_vocab_size if embed_input else input_dim,
            hid_dim=hidden_dim, 
            n_layers=num_enc_layers, 
            n_heads=num_enc_heads, 
            pf_dim=enc_fc_dim,
            dropout=enc_dropout, 
            device=device,
            max_length=max_len,
            token_embedding=embed_input,
        )

        dec = Decoder(
            output_dim=num_classes, 
            hid_dim=hidden_dim, 
            n_layers=num_dec_layers, 
            n_heads=num_dec_heads, 
            pf_dim=dec_fc_dim, 
            dropout=dec_dropout, 
            device=device,
            max_length=max_len
        )

        model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, device)
        return TransformerModel(model, device, trg_pad_idx)

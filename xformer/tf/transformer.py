import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.__version__
"""
## Transformer Input Layer
This layer computes the sum of position embeddings and feature embeddings to feed to
the transformer layers.
"""


class TransformerInput(layers.Layer):
    def __init__(
        self, input_type="tokens", nvocab=1000, nhid=64, nff=128, maxlen=100,
    ):
        super().__init__()
        self.input_type = input_type
        if input_type == "tokens":
            self.emb = tf.keras.layers.Embedding(nvocab, nhid)
        elif input_type == "feats":
            self.emb = tf.keras.layers.Dense(nhid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=nhid)

    def call(self, x):
        if self.input_type == "tokens":
            maxlen = tf.shape(x)[-1]
        elif self.input_type == "feats":
            maxlen = tf.shape(x)[1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


"""
## Transformer Encoder Layer
"""


class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Transformer Decoder Layer
"""


class TransformerDecoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.drop = layers.Dropout(rate)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """
        Mask the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, trg):
        input_shape = tf.shape(trg)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        trg_att = self.self_att(trg, trg, attention_mask=causal_mask)
        trg_norm = self.ln1(trg + self.drop(trg_att))
        enc_out = self.enc_att(trg_norm, enc_out)
        enc_out_norm = self.ln2(self.drop(enc_out) + trg_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.ln3(enc_out_norm + self.drop(ffn_out))
        return ffn_out_norm


"""
## Complete Transformer Model
"""
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def masked_loss(real, pred):
    """ assuming pad token index = 0 """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


class Transformer(keras.Model):
    def __init__(
        self,
        input_type="tokens",
        nvocab=1000,
        ninp=32,
        nhid=64,
        nhead=2,
        nff=128,
        src_maxlen=100,
        trg_maxlen=100,
        nlayers=2,
        nclasses=10,
    ):
        super().__init__()
        self.nlayers = nlayers
        self.trg_maxlen = trg_maxlen
        self.input_type = input_type

        self.enc_input = TransformerInput(
            input_type=input_type, nvocab=1000, nhid=64, maxlen=src_maxlen
        )
        self.dec_input = TransformerInput(
            input_type="tokens", nvocab=nclasses, nhid=nhid, maxlen=trg_maxlen
        )
        for i in range(nlayers):
            setattr(self, f"enc_layer_{i}", TransformerEncoderLayer(nhid, nhead, nff))
            setattr(self, f"dec_layer_{i}", TransformerDecoderLayer(nhid, nhead, nff))

        self.final = layers.Dense(nclasses)

    def encode_src(self, src):
        x = self.enc_input(src)
        for i in range(self.nlayers):
            x = getattr(self, f"enc_layer_{i}")(x)
        return x

    def decode(self, enc_out, trg):
        y = self.dec_input(trg)
        for i in range(self.nlayers):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        src = inputs[0]
        trg = inputs[1]
        x = self.encode_src(src)
        y = self.decode(x, trg)
        return self.final(y)

    def train_step(self, batch):
        """ Process one batch inside model.fit() """
        src = batch["src"]
        trg = batch["trg"]
        dec_inp = trg[:, :-1]
        dec_trg = trg[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([src, dec_inp])
            loss = masked_loss(dec_trg, preds)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}

    def generate(self, src, trg_start_token_idx):
        """ Return a batch of predicted token indices with greedy deecoding """
        bs = tf.shape(src)[0]
        dec_inp = tf.ones((bs, self.trg_maxlen), dtype=tf.int32) * trg_start_token_idx
        for i in range(self.trg_maxlen):
            preds = self([src, dec_inp])
            pred_idx = tf.argmax(preds, axis=-1, output_type=tf.int32)
            current_pred = tf.expand_dims(pred_idx[:, i], axis=-1)
            if i < self.trg_maxlen - 1:
                future_pad = tf.ones((bs, self.trg_maxlen - (i + 2)), dtype=tf.int32)
                dec_inp = tf.concat(
                    [dec_inp[:, : i + 1], current_pred, future_pad], axis=-1
                )
            else:
                dec_inp = tf.concat([dec_inp[:, : i + 1], current_pred], axis=-1)
        return pred_idx


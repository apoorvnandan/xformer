from tensorflow import keras
import tensorflow as tf

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(self, batch, idx_to_token, trg_start_token_idx=27, trg_end_token_idx=28):
        """ Displays a batch of outputs after every epoch 
        Arguments:
        - batch: test batch containing the keys "src" and "trg"
        - idx_to_token: a List containing the vocabulary tokens corresponding to their indices
        - trg_start_token_idx: start token index in the target vocabulary
        - trg_end_token_idx: end token index in the target vocabulary
        """
        self.batch = batch
        self.trg_start_token_idx = trg_start_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        src = self.batch["src"]
        trg = self.batch["trg"].numpy()
        bs = tf.shape(src)[0]
        preds = self.model.generate(src, self.trg_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target = ""
            for idx in trg[i, :]:
                target += self.idx_to_char[idx]
            prediction = "<"
            over = False
            for idx in preds[i, :]:
                if over:  # Add padding token once end token has beeen predicted
                    prediction += "-"
                    continue
                if idx == 28:
                    over = True
                prediction += self.idx_to_char[idx]
            print(f"target:     {target}")
            print(f"prediction: {prediction}")
            print()


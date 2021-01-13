import random
import os



import tensorflow as tf
def char_to_idx(ch):
    if ch == "<":
        return 27  # start token
    if ch == ">":
        return 28  # end token
    if ch == "-":
        return 0  # pad token
    if ch == " ":
        return 29  # space token
    return ord(ch) - 96  # a->1, b->2, etc


def filename_to_label(f):
    fsplit = f.split('_+_')
    f1 = fsplit[0]
    f2 = fsplit[1]
    m = {
        "0": [char_to_idx(ch) for ch in "<zero>-"],
        "1": [char_to_idx(ch) for ch in "<one>--"],
        "2": [char_to_idx(ch) for ch in "<two>--"],
        "3": [char_to_idx(ch) for ch in "<three>"],
        "4": [char_to_idx(ch) for ch in "<four>-"],
        "5": [char_to_idx(ch) for ch in "<five>-"],
        "6": [char_to_idx(ch) for ch in "<six>--"],
        "7": [char_to_idx(ch) for ch in "<seven>"],
        "8": [char_to_idx(ch) for ch in "<eight>"],
        "9": [char_to_idx(ch) for ch in "<nine>-"],
    }
    label = m[f1.split("/")[-1][0]] + m[f2.split("/")[-1][0]]
    pads = []
    nonpads = []
    for i in label:
        if i == 0:
            pads.append(i)
        else:
            nonpads.append(i)
    label = nonpads + pads
    clean = []
    i = 0
    while i < len(label):
        if i+1 < len(label):
            if label[i] == 28 and label[i+1] == 27:
                i += 2
                clean.append(29)
                continue
        clean.append(label[i])
        i += 1
    return clean


def fpath_to_logmelspec(f):
    fsplit = tf.strings.split(f,'_+_')
    f1 = fsplit[0]
    f2 = fsplit[1]
    sample_rate = 8000
    audio1 = tf.io.read_file(f1)
    audio1, _ = tf.audio.decode_wav(audio1, 1, sample_rate)
    audio2 = tf.io.read_file(f2)
    audio2, _ = tf.audio.decode_wav(audio2, 1, sample_rate)
    audio = tf.concat([audio1,audio2], axis=0)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.expand_dims(audio, axis=0)
    stfts = tf.signal.stft(
        audio, frame_length=1024, frame_step=256, fft_length=1024
    )  # A 1024-point STFT with frames of 64 ms and 75% overlap.
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 2000.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return tf.squeeze(log_mel_spectrograms, axis=0)


def create_dataset(flist, bs=4):
    label_data = [filename_to_label(f) for f in flist]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(fpath_to_logmelspec)
    label_ds = tf.data.Dataset.from_tensor_slices(label_data)
    ds = tf.data.Dataset.zip((audio_ds, label_ds))
    ds = ds.map(lambda x, y: {"src": x, "trg": y})
    return ds.batch(bs)


def get_data():
    data_path = keras.utils.get_file(
        "spoken_digit.tar.gz",
        "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.9.tar.gz",
    )
    command = f"tar -xvf {data_path} --directory ."
    os.system(command)

    root = "free-spoken-digit-dataset-1.0.9/recordings"
    flist = [os.path.join(root,x) for x in os.listdir(root)]
    random.shuffle(flist)

    new_flist = []
    for i,f in enumerate(flist):
        if i+1 == len(flist):
            break
        g = flist[i+1]
        new_f = f"{f}_+_{g}"
        new_flist.append(new_f)
        
    ds = create_dataset(new_flist[:1900])
    val_ds = create_dataset(new_flist[1900:])
    return ds, val_ds

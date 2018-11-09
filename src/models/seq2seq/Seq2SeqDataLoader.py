import tensorflow as tf
import numpy as np

from hbconfig import Config
from sklearn.model_selection import train_test_split
from models.seq2seq.Seq2SeqHelper import *

class DataLoader():
    def __init__(self):
        embed_file = Config.data.embed_file
        src_data_file = Config.data.data_prefix+Config.data.src_suffix
        tgt_data_file = Config.data.data_prefix+Config.data.tgt_suffix
        self._preprocess(src_data_file, tgt_data_file, embed_file)        
        self._make_train_and_test()
            
    def make_batch(self, src_idx, tgt_idx):
        def _parse(src, tgt_in, tgt_out, src_len, tgt_len):
            return {"source": src, 
                    "source_sequence_length": src_len}, {
                    "target_input": tgt_in,
                    "target_output": tgt_out,
                    "target_sequence_length": tgt_len}

        dataset = tf.data.Dataset.from_generator(
            lambda: self._next_batch(src_idx, tgt_idx),
            (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None])))
        dataset = dataset.map(_parse)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def predict_input_fn(self, xs):
        def infer_inps(str_li):
            xs_len = [len(s.split()) for s in str_li]
            max_len = max(xs_len)
            xs = [self.text_to_id(s.split()) for s in str_li]
            xs = tf.keras.preprocessing.sequence.pad_sequences(xs, max_len, padding='post')
            xs_len = np.array(xs_len)
            return xs,xs_len

        x,x_len = infer_inps(xs)
        return tf.estimator.inputs.numpy_input_fn(
            x = {'source':x,'source_sequence_length':x_len},
            shuffle = False)

    def text_to_id(self, texts):
        return [self.word2idx.get(char, UNK_ID) for char in texts]

    def id_to_text(self, ids):
        return [self.idx2word.get(id, UNK) for id in ids]

    def _next_batch(self, src_idx, tgt_idx):
        for i in range(0, len(src_idx), Config.model.batch_size):
            src_batch = src_idx[i: i+Config.model.batch_size]
            tgt_batch = tgt_idx[i: i+Config.model.batch_size]
            
            src_len,tgt_in,tgt_out,tgt_len = [],[],[],[]
            tgt_batch_in,tgt_batch_out = [],[]

            for src in src_batch:
                src_len.append(len(src))

            for tgt in tgt_batch:
                tgt_len.append(len(tgt)+1)
                tgt_batch_in.append([SOS_ID]+tgt)
                tgt_batch_out.append(tgt+[EOS_ID])

            src = self._pad_sent_batch(src_batch)
            tgt_in = self._pad_sent_batch(tgt_batch_in)
            tgt_out = self._pad_sent_batch(tgt_batch_out)

            yield src, tgt_in, tgt_out, src_len, tgt_len

    def _pad_sent_batch(self, sent_batch):
        max_sent_len = max([len(sent) for sent in sent_batch])
        padded_seqs = [(sent + [PAD_ID]*(max_sent_len - len(sent))) for sent in sent_batch]
        return padded_seqs

    def _make_train_and_test(self):
        src_lines = self.src_data.split('\n')
        tgt_lines = self.tgt_data.split('\n')

        X = [self.text_to_id(line.split()) for line in src_lines]
        y = [self.text_to_id(line.split()) for line in tgt_lines]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def _preprocess(self, src_data_file, tgt_data_file, embed_file):
        self.src_data = read_data(src_data_file)
        self.tgt_data = read_data(tgt_data_file)
        words = load_embed(embed_file)
        self.idx2word, self.word2idx = build_map(words)
        self.vocab_size = len(self.word2idx)
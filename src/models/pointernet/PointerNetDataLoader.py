import tensorflow as tf
import numpy as np

from hbconfig import Config
from sklearn.model_selection import train_test_split
from models.seq2seq.Seq2SeqDataLoader import DataLoader as Seq2SeqDataLoader
from models.seq2seq.Seq2SeqHelper import *

class DataLoader(Seq2SeqDataLoader):
    def __init__(self):
        super(DataLoader,self).__init__()
            
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

    def _next_batch(self, src, tgt):
        batch_src = []
        batch_src_len = []
        batch_tgt_in = []
        batch_tgt_out = []
        batch_tgt_len = []
            
        for src_line,tgt_line in zip(src,tgt):
            single_src = self.text_to_id(src_line)+[EOS_ID]
            single_src_len = len(single_src)
            single_tgt = self.text_to_id(tgt_line)+[EOS_ID]
            single_tgt = [single_src.index(t) for t in single_tgt]
            single_tgt_len = len(single_tgt)
            batch_src.append(single_src)
            batch_src_len.append(single_src_len)
            batch_tgt_in.append(single_tgt)
            batch_tgt_out.append(single_tgt)
            batch_tgt_len.append(single_tgt_len)
            if len(batch_src) == Config.model.batch_size:
                pad_src = self._pad_sent_batch(batch_src)
                pad_tgt_in = self._pad_sent_batch(batch_tgt_in)
                pad_tgt_out = self._pad_sent_batch(batch_tgt_out)
                yield pad_src,pad_tgt_in,pad_tgt_out,batch_src_len,batch_tgt_len
                batch_src = []
                batch_src_len = []
                batch_tgt_in = []
                batch_tgt_out = []
                batch_tgt_len = []

    def _make_train_and_test(self):
        src_lines = self.src_data.split('\n')
        tgt_lines = self.tgt_data.split('\n')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(src_lines, tgt_lines, test_size=0.2)
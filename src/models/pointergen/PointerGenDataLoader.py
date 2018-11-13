import tensorflow as tf
import numpy as np

from hbconfig import Config
from sklearn.model_selection import train_test_split
from models.seq2seq.Seq2SeqDataLoader import DataLoader as Seq2SeqDataLoader
from models.seq2seq.Seq2SeqHelper import *

class DataLoader(Seq2SeqDataLoader):
    def __init__(self):
        super(DataLoader,self).__init__()
            
    def make_batch(self, src, tgt):
        def _parse(src, tgt_in, tgt_out, src_len, tgt_len, src_max_oov):
            return {"source": src, 
                    "source_sequence_length": src_len,
                    "source_max_oov":src_max_oov}, {
                    "target_input": tgt_in,
                    "target_output": tgt_out,
                    "target_sequence_length": tgt_len}

        dataset = tf.data.Dataset.from_generator(
            lambda: self._next_batch(src, tgt),
            (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape(None)))
        dataset = dataset.map(_parse)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def _next_batch(self, src, tgt):
        batch_src = []
        batch_src_oovs = []
        batch_src_len = []
        batch_tgt_in = []
        batch_tgt_out = []
        batch_tgt_len = []
            
        for src_line,tgt_line in zip(src,tgt):
            single_src,single_src_oovs,single_src_len = self.source2idx(src_line.strip())
            single_tgt,single_tgt_len = self.target2idx(tgt_line.strip(),single_src_oovs)
            batch_src.append(single_src)
            batch_src_oovs.append(single_src_oovs)
            batch_src_len.append(single_src_len)
            batch_tgt_in.append([SOS_ID]+single_tgt)
            batch_tgt_out.append(single_tgt+[EOS_ID])
            batch_tgt_len.append(single_tgt_len+1)
            if len(batch_src) == Config.model.batch_size:
                pad_src = self._pad_sent_batch(batch_src)
                pad_tgt_in = self._pad_sent_batch(batch_tgt_in)
                pad_tgt_out = self._pad_sent_batch(batch_tgt_out)
                max_oovs = max(len(x) for x in batch_src_oovs)
                yield pad_src,pad_tgt_in,pad_tgt_out,batch_src_len,batch_tgt_len,max_oovs
                batch_src = []
                batch_src_oovs = []
                batch_src_len = []
                batch_tgt_in = []
                batch_tgt_out = []
                batch_tgt_len = []
    
    def source2idx(self, sent):
        tokens = sent.split()
        single_src = []
        single_src_oovs = []
        single_src_len = len(tokens)
        for token in tokens:
            if token not in self.word2idx:
                if token not in single_src_oovs:
                    single_src_oovs.append(token)
                single_src.append(self.vocab_size + single_src_oovs.index(token))
            else:
                single_src.append(self.word2idx[token])
        return single_src, single_src_oovs, single_src_len

    def target2idx(self, sent, oovs):
        tokens = sent.split()
        single_tgt = []
        single_tgt_len = len(tokens)
        for token in tokens:
            if token not in self.word2idx:
                if token not in oovs:
                    single_tgt.append(UNK_ID)
                else:
                    single_tgt.append(self.vocab_size + oovs.index(token))
            else:
                single_tgt.append(self.word2idx[token])
        return single_tgt, single_tgt_len

    def _make_train_and_test(self):
        src_lines = self.src_data.split('\n')
        tgt_lines = self.tgt_data.split('\n')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(src_lines, tgt_lines, test_size=0.2)
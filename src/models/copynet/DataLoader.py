import tensorflow as tf
import numpy as np

from hbconfig import Config
from sklearn.model_selection import train_test_split
from models.attention.DataLoader import DataLoader as AttentionDataLoader
from utils.Seq2SeqHelper import *

class DataLoader(AttentionDataLoader):
    def __init__(self):
        super(DataLoader,self).__init__() 
            
    def get_batch(self, src, tgt, scope='train'):
        def _parse(src, src_len, src_oovs, max_src_oovs_len, tgt_in, tgt_out, tgt_len):
            return {"source": src, 
                    "source_sequence_length": src_len,
                    "source_oovs":src_oovs,
                    "max_source_oovs_length":max_src_oovs_len}, {
                    "target_input": tgt_in,
                    "target_output": tgt_out,
                    "target_sequence_length": tgt_len}

        dataset = tf.data.Dataset.from_generator(
            lambda: self._make_batch(src, tgt),
            (tf.int32, tf.int32, tf.string, tf.int32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape(None), 
             tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None])))
        dataset = dataset.map(_parse)
        if scope == 'train':
            dataset = dataset.shuffle(buffer_size=10000)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def _make_batch(self, src, tgt):
        batch_src = []
        batch_src_len = []
        batch_src_oovs = []
        batch_tgt_in = []
        batch_tgt_out = []
        batch_tgt_len = []            
        for src_line,tgt_line in zip(src,tgt):
            single_src,single_src_len,single_src_oovs = self._source2idx(src_line.strip())
            single_tgt,single_tgt_len = self._target2idx(tgt_line.strip(), single_src_oovs)
            batch_src.append(single_src)
            batch_src_len.append(single_src_len)
            batch_src_oovs.append(single_src_oovs)
            batch_tgt_in.append([SOS_ID]+single_tgt)
            batch_tgt_out.append(single_tgt+[EOS_ID])
            batch_tgt_len.append(single_tgt_len+1)
            if len(batch_src) == Config.model.batch_size:
                pad_src = self._pad_sent_batch(batch_src)
                pad_src_oovs = self._pad_sent_batch(batch_src_oovs,['<PAD>'])
                max_src_oovs_len = max(len(x) for x in batch_src_oovs)
                pad_tgt_in = self._pad_sent_batch(batch_tgt_in)
                pad_tgt_out = self._pad_sent_batch(batch_tgt_out)
                yield pad_src,batch_src_len,pad_src_oovs,max_src_oovs_len,pad_tgt_in,pad_tgt_out,batch_tgt_len
                batch_src = []
                batch_src_len = []
                batch_src_oovs = []
                batch_tgt_in = []
                batch_tgt_out = []
                batch_tgt_len = []

    def get_infer_batch(self, src):
        def _parse(src, src_len, src_oovs, max_src_oovs_len):
            return {"source": src, 
                    "source_sequence_length": src_len,
                    "source_oovs":src_oovs,
                    "max_source_oovs_length":max_src_oovs_len}

        dataset = tf.data.Dataset.from_generator(
            lambda: self._make_infer_batch(src),
            (tf.int32, tf.int32, tf.string, tf.int32),
            (tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape(None)))
        dataset = dataset.map(_parse)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features
        
    def _make_infer_batch(self, src):
        batch_src = []
        batch_src_len = []
        batch_src_oovs = []            
        for src_line in src:
            single_src,single_src_len,single_src_oovs = self._source2idx(src_line.strip())
            batch_src.append(single_src)
            batch_src_len.append(single_src_len)
            batch_src_oovs.append(single_src_oovs)
        pad_src = self._pad_sent_batch(batch_src)
        pad_src_oovs = self._pad_sent_batch(batch_src_oovs,['<PAD>'])
        max_src_oovs_len = max(len(x) for x in batch_src_oovs)
        yield pad_src, batch_src_len, pad_src_oovs, max_src_oovs_len
    
    def _source2idx(self, src_sent):
        src_tokens = src_sent.split()
        single_src = []
        single_src_oovs = []
        single_src_len = len(src_tokens)
        for token in src_tokens:
            if token not in self.word2idx:
                if token not in single_src_oovs:
                    single_src_oovs.append(token)
                single_src.append(self.vocab_size + single_src_oovs.index(token))
            else:
                single_src.append(self.word2idx[token])

        return single_src, single_src_len, single_src_oovs
    
    def _target2idx(self, tgt_sent, single_src_oovs):        
        tgt_tokens = tgt_sent.split()
        single_tgt = []
        single_tgt_len = len(tgt_tokens)
        
        for token in tgt_tokens:
            if token not in self.word2idx:
                if token not in single_src_oovs:
                    single_tgt.append(UNK_ID)
                else:
                    single_tgt.append(self.vocab_size + single_src_oovs.index(token))
            else:
                single_tgt.append(self.word2idx[token])

        return single_tgt, single_tgt_len
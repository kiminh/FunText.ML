import tensorflow as tf
import numpy as np

from hbconfig import Config
from sklearn.model_selection import train_test_split
from utils.Seq2SeqHelper import *

class DataLoader():
    def __init__(self):
        embed_file = Config.data.embed_file
        src_data_file = Config.data.data_prefix+Config.data.src_suffix
        tgt_data_file = Config.data.data_prefix+Config.data.tgt_suffix
        src_data,tgt_data = self._preprocess(src_data_file, tgt_data_file, embed_file)        
        self._make_train_and_test(src_data,tgt_data)

    def _preprocess(self, src_data_file, tgt_data_file, embed_file):
        src_data = read_data(src_data_file)
        tgt_data = read_data(tgt_data_file)
        words,self.word_embs = load_embed(embed_file)
        self.idx2word, self.word2idx = build_map(words)
        self.vocab_size = len(self.word2idx)
        return src_data,tgt_data

    def _make_train_and_test(self,src_data,tgt_data):
        X = src_data.split('\n')
        y = tgt_data.split('\n')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
            
    def get_batch(self, src, tgt, scope='train'):
        def _parse(src, src_len, tgt_in, tgt_out, tgt_len):
            return {"source": src, 
                    "source_sequence_length": src_len}, {
                    "target_input": tgt_in,
                    "target_output": tgt_out,
                    "target_sequence_length": tgt_len}

        dataset = tf.data.Dataset.from_generator(
            lambda: self._make_batch(src, tgt),
            (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([None, None]), tf.TensorShape([None]), 
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
        batch_tgt_in = []
        batch_tgt_out = []
        batch_tgt_len = []
            
        for src_line,tgt_line in zip(src,tgt):
            single_src,single_src_len = self._source2idx(src_line.strip())
            single_tgt,single_tgt_len = self._target2idx(tgt_line.strip())
            batch_src.append(single_src)
            batch_src_len.append(single_src_len)
            batch_tgt_in.append([SOS_ID]+single_tgt)
            batch_tgt_out.append(single_tgt+[EOS_ID])
            batch_tgt_len.append(single_tgt_len+1)
            if len(batch_src) == Config.model.batch_size:
                pad_src = self._pad_sent_batch(batch_src)
                pad_tgt_in = self._pad_sent_batch(batch_tgt_in)
                pad_tgt_out = self._pad_sent_batch(batch_tgt_out)
                yield pad_src,batch_src_len,pad_tgt_in,pad_tgt_out,batch_tgt_len
                batch_src = []
                batch_src_len = []
                batch_tgt_in = []
                batch_tgt_out = []
                batch_tgt_len = []
    
    def get_infer_batch(self, src):
        def _parse(src, src_len):
            return {"source": src, 
                    "source_sequence_length": src_len}

        dataset = tf.data.Dataset.from_generator(
            lambda: self._make_infer_batch(src),
            (tf.int32, tf.int32),
            (tf.TensorShape([None, None]), tf.TensorShape([None])))
        dataset = dataset.map(_parse)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features
        
    def _make_infer_batch(self, src):
        batch_src = []
        batch_src_len = []
            
        for src_line in src:
            single_src,single_src_len = self._source2idx(src_line.strip())
            batch_src.append(single_src)
            batch_src_len.append(single_src_len)
        pad_src = self._pad_sent_batch(batch_src)
        yield pad_src,batch_src_len
    
    def _source2idx(self, src_sent):
        src_tokens = src_sent.split()
        single_src = []
        single_src_len = len(src_tokens)

        for token in src_tokens:
            if token not in self.word2idx:
                single_src.append(UNK_ID)
            else:
                single_src.append(self.word2idx[token])

        return single_src, single_src_len
    
    def _target2idx(self, tgt_sent):        
        tgt_tokens = tgt_sent.split()
        single_tgt = []
        single_tgt_len = len(tgt_tokens)
        
        for token in tgt_tokens:
            if token not in self.word2idx:
                single_tgt.append(UNK_ID)
            else:
                single_tgt.append(self.word2idx[token])

        return single_tgt, single_tgt_len

    def _pad_sent_batch(self, sent_batch, pads=[PAD_ID]):
        max_sent_len = max([len(sent) for sent in sent_batch])
        padded_seqs = [(sent + pads*(max_sent_len - len(sent))) for sent in sent_batch]
        return padded_seqs
import tensorflow as tf
import numpy as np
import nltk

from hbconfig import Config
from models.seq2seq.Seq2SeqGenerator import Generator as Seq2SeqGenerator
from models.seq2seq.Seq2SeqHelper import *
from models.copynet.CopyNetHelper import CopyNetWrapper, CopyNetWrapperState

class Generator(Seq2SeqGenerator):
    def __init__(self, dl):
        super(Generator,self).__init__(dl)   
    
    def predict_fn(self, text, estimator):
        res = []      
        xs = text.split('\n')
        preds = list(estimator.predict(lambda: self.dl.predict_input_fn(xs)))
        for x, pred in zip(xs, preds):
            pred_str = ' '.join(self.dl.id_to_text(pred))
            print('IN: {}'.format(x))
            print('OUT: {}'.format(pred_str))
            res.append(pred_str) 

        return '\n'.join(res)
        
    def _init_placeholder(self, features, labels):
        self._source = features['source']
        self._source_sequence_length = features['source_sequence_length']
        self._source_max_oov = features['source_max_oov']
        self._batch_size = tf.shape(self._source)[0]

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            self._target_input = labels['target_input']
            self._target_output = labels['target_output']
            self._target_sequence_length = labels['target_sequence_length']
            self._max_target_sequence_length = tf.reduce_max(self._target_sequence_length)
        else:
            max_source_sequence_length = tf.reduce_max(self._source_sequence_length)
            self._max_target_sequence_length = tf.to_int32(tf.round(tf.to_float(max_source_sequence_length) * 2.0))

    def _build_decoder(self, encoder_outputs, encoder_state):        
        with tf.variable_scope('decoder'):
            cell, initial_state = self._build_decoder_cell(encoder_state, encoder_outputs)
            
            cell = CopyNetWrapper(cell, self._source, self._source_max_oov, encoder_outputs, self._output_layer, self._vocab_size)
            
            self._output_layer = None
            initial_state = cell.zero_state(self._batch_size,
                tf.float32).clone(cell_state=initial_state)

            if self._mode != tf.estimator.ModeKeys.PREDICT:
                return self._build_decoder_train_eval(cell, initial_state)
            else:
                return self._build_decoder_infer(cell, initial_state)
    
        
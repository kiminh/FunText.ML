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

    def _build_decoder(self, encoder_outputs, encoder_state):        
        with tf.variable_scope('decoder'):
            cell, initial_state = self._build_decoder_cell(encoder_state, encoder_outputs)
            
            cell = CopyNetWrapper(cell, self._source, 100, encoder_outputs, self._output_layer, self._vocab_size)
            
            self._output_layer = None
            initial_state = cell.zero_state(self._batch_size,
                tf.float32).clone(cell_state=initial_state)

            if self._mode != tf.estimator.ModeKeys.PREDICT:
                return self._build_decoder_train_eval(cell, initial_state)
            else:
                return self._build_decoder_infer(cell, initial_state)
    
        
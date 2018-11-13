import tensorflow as tf
import numpy as np
import nltk

from hbconfig import Config
from models.seq2seq.Seq2SeqGenerator import Generator as Seq2SeqGenerator
from models.seq2seq.Seq2SeqHelper import *
from models.pointernet.PointerNetHelper import PointerGeneratorDecoder, PointerGeneratorGreedyEmbeddingHelper, PointerGeneratorBahdanauAttention,PointerGeneratorAttentionWrapper

class Generator(Seq2SeqGenerator):
    def __init__(self, dl):
        super(Generator,self).__init__(dl)
        
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
            
    
    def _build_decoder_train_eval(self, cell, initial_state):
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs = self._decoder_emb_inp,
            sequence_length = self._target_sequence_length)

        decoder = PointerGeneratorDecoder(
            source_extend_tokens = self._source,
            source_oov_words = self._source_max_oov,
            coverage = Config.model.coverage,
            cell = cell,
            helper = helper,
            initial_state = initial_state,
            output_layer = self._output_layer)

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,            
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=self._max_target_sequence_length,
            swap_memory=True)

        logits = decoder_output.rnn_output
        sample_id = decoder_output.sample_id

        logits_length = tf.shape(logits)[1]
        label_length = self._max_target_sequence_length
        pad_size = label_length - logits_length
        logits = tf.pad(logits, [[0, 0], [0, pad_size], [0, 0]])
        predictions = tf.argmax(logits, axis=2)
        
        return logits, predictions
    
    def _build_decoder_infer(self, cell, initial_state):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self._embedding_decoder,
            start_tokens=tf.tile([SOS_ID], [self._batch_size]),
            end_token=EOS_ID
        )

        decoder = PointerGeneratorDecoder(
            source_extend_tokens = self._source,
            source_oov_words = self._source_max_oov,
            coverage = Config.model.coverage,
            cell=cell,
            helper=helper,
            initial_state=initial_state,
            output_layer=self._output_layer
        )

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,            
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=self._max_target_sequence_length)
        
        logits = tf.no_op()
        predictions = tf.expand_dims(dec_outputs.sample_id, -1)
        predictions = tf.transpose(predictions, perm=[0, 2, 1])
        return logits, predictions

    def _build_decoder_cell(self, encoder_state, encoder_outputs):
        memory = encoder_outputs
        memory_sequence_length = self._source_sequence_length
        cell_state = encoder_state

        attention = PointerGeneratorBahdanauAttention(
            num_units = Config.model.num_units,
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            coverage = Config.model.coverage)
        
        cell = PointerGeneratorAttentionWrapper(
            cell = create_rnn_cell(self._num_layers,self._mode),
            attention_mechanism = attention,
            attention_layer_size = Config.model.num_units,
            alignment_history = True,
            coverage = Config.model.coverage)

        initial_state = cell.zero_state(self._batch_size, tf.float32).clone(cell_state=cell_state)

        #指导文件
        # # setup initial state of decoder
        # initial_state = [self.decode_initial_state for i in range(
        #     self.config.decode_layer_num)]
        # # initial state for attention cell
        # attention_cell_state = decode_cell[0].zero_state(
        #     dtype=tf.float32, batch_size=batch_size)
        # initial_state[0] = attention_cell_state.clone(
        #     cell_state=initial_state[0])
        # initial_state = tuple(initial_state)
        
        return cell, initial_state
    
        
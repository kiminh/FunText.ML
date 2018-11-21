import tensorflow as tf
import numpy as np
import nltk

from hbconfig import Config
from models.attention.Generator import Generator as AttentionGenerator
from utils.Seq2SeqHelper import *

class Generator(AttentionGenerator):
    def __init__(self, dl):
        super(Generator,self).__init__(dl)

    def model_fn(self, features, labels, mode):
        self._mode = mode   
        self.loss, self.train_op,  self.metrics, self.predictions = None, None, None, None

        self.logits, self.predictions = self._build_graph(features, labels, mode)
        
        if mode != tf.estimator.ModeKeys.PREDICT:
            self.loss = self._build_loss()
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.train_op = self._build_optimizer()
        
        if mode == tf.estimator.ModeKeys.EVAL:
            self.metrics = self._build_metric()

        predictions = {
            "predictions": self.predictions,
            "oovs": self._source_oovs
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions=predictions)
    
    def predict_fn(self, text, estimator):
        res = []
        xs = text.split('\n')
        predict_input_fn = lambda: self.dl.get_infer_batch(xs)
        preds =list(estimator.predict(predict_input_fn))
        for ii in range(len(xs)):
            pred_str_list = []
            for word_id in preds[ii]['predictions']:
                if(word_id==EOS_ID):
                    break
                if word_id < self._vocab_size:
                    pred_str_list.append(self.dl.idx2word[word_id])
                else:
                    pred_str_list.append(preds[ii]['oovs'][word_id - self._vocab_size].decode())            
            pred_str = ' '.join(pred_str_list)
            x = xs[ii]
            print('IN : {}'.format(x))
            print('OUT: {}'.format(pred_str))
            res.append(pred_str)

        return '\n'.join(res)
        
    def _build_graph(self, features, labels, mode):
        self._mode = mode
        self._init_placeholder(features, labels)        
        self._init_embeddings()
        encoder_outputs, encoder_state = self._build_encoder()
        return self._build_decoder(encoder_outputs, encoder_state)
    
    def _init_placeholder(self, features, labels):
        self._source = features['source']
        self._source_sequence_length = features['source_sequence_length']
        self._max_source_sequence_length = tf.shape(self._source)[1]
        self._source_mask = tf.sequence_mask(self._source_sequence_length, self._max_source_sequence_length)
        self._source_oovs = features['source_oovs']
        self._max_source_oovs_length = features['max_source_oovs_length']

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            self._target_input = labels['target_input']
            self._target_output = labels['target_output']
            self._target_sequence_length = labels['target_sequence_length']
            self._max_target_sequence_length = tf.reduce_max(self._target_sequence_length)
            self._target_mask = tf.sequence_mask(self._target_sequence_length, self._max_target_sequence_length)
        else:
            self._target_input = self._source
            self._target_output = self._source
            self._target_sequence_length = self._source_sequence_length
            self._max_target_sequence_length = tf.reduce_max(self._target_sequence_length)
            self._target_mask = tf.sequence_mask(self._target_sequence_length, self._max_target_sequence_length)

        self._batch_size = tf.shape(self._source)[0]
    
    def _init_embeddings(self):
        with tf.variable_scope("embeddings", dtype=self._dtype):
            self._embedding = tf.get_variable(name='embedding_share', shape=[self._vocab_size, Config.model.embed_dim], dtype=tf.float32)

            self._encoder_emb_inp = tf.nn.embedding_lookup(
                params=self._embedding,
                ids=tf.where(condition=tf.less(self._source, self._vocab_size),
                             x=self._source,
                             y=tf.ones_like(self._source) * self._vocab_size-1))

            self._decoder_emb_inp = tf.nn.embedding_lookup(
                params=self._embedding,
                ids=tf.where(condition=tf.less(self._target_input, self._vocab_size),
                             x=self._target_input,
                             y=tf.ones_like(self._target_input) * self._vocab_size-1))
    
    def _build_decoder(self, encoder_outputs, encoder_state):        
        with tf.variable_scope('decoder'):
            cell, initial_state = self._build_decoder_cell(encoder_state, encoder_outputs)
            return self._build_decoder_copy(encoder_outputs, cell, initial_state)

    def _build_decoder_copy(self, encoder_outputs, cell, initial_state):
        self.weights_copy = tf.get_variable(
            "weights_copy", shape=[Config.model.num_units * 2, Config.model.num_units])
        self.weights_generate = tf.get_variable(
            "weights_generate", shape=[Config.model.num_units, self._vocab_size])
        self.decoder_cell = cell

        def cond(time, state, copy_pro, max_len, output_prob_list):
            return time < max_len

        def body(time, state, copy_pro, max_len, output_prob_list):
            # selective read
            this_decoder_emb = self._decoder_emb_inp[:, time, :]
            this_decoder_data = self._target_input[:, time]
            selective_mask = tf.cast(tf.equal(self._source, tf.expand_dims(this_decoder_data, axis=1)),
                                        dtype=tf.float32)  # batch * encoder_max_len
            selective_mask_sum = tf.reduce_sum(selective_mask, axis=1)
            rou = tf.where(tf.less(selective_mask_sum, 1e-10),
                                        selective_mask, selective_mask / tf.expand_dims(selective_mask_sum, 1))

            selective_read = tf.einsum("ijk,ij->ik", encoder_outputs, rou)

            this_decoder_final = tf.concat([this_decoder_emb, selective_read], axis=1)
            this_decoder_output, state = self.decoder_cell(this_decoder_final, state)  # batch * hidden_dim

            # generate mode
            generate_score = tf.matmul(
                this_decoder_output, self.weights_generate, name="generate_score")  # batch * vocab_size

            # copy mode
            copy_score = tf.einsum("ijk,km->ijm", encoder_outputs, self.weights_copy)
            copy_score = tf.nn.tanh(copy_score)
            copy_score = tf.einsum("ijm,im->ij", copy_score, this_decoder_output)
            copy_score = self._mask_logits(self._source_mask, copy_score)

            mix_score = tf.concat([generate_score, copy_score], axis=1)  # batch * (vocab_size + encoder_max_len)
            probs = tf.cast(tf.nn.softmax(mix_score), tf.float32)
            prob_g = probs[:, :self._vocab_size]
            prob_c = probs[:, self._vocab_size:]

            encoder_inputs_one_hot = tf.one_hot(
                indices=self._source,
                depth=self._vocab_size + self._max_source_oovs_length)
            prob_c = tf.einsum("ijn,ij->in", encoder_inputs_one_hot, prob_c)

            # if encoder inputs has intersection words with vocab dict,
            # move copy mode probability to generate mode probability

            prob_g = prob_g + prob_c[:, :self._vocab_size]
            prob_c = prob_c[:, self._vocab_size:]
            prob_final = tf.concat([prob_g, prob_c], axis=1) + 1e-10  # batch * (vocab_size + oovs_size)

            output_prob_list = output_prob_list.write(time, prob_final)

            return time + 1, state, prob_c, max_len, output_prob_list

        self.output_prob_list = tf.TensorArray(dtype=tf.float32, size=self._max_target_sequence_length, name="logits_list")
        _, _, _, _, self.output_prob_list = tf.while_loop(
            cond, body,
            loop_vars=[0,
                        initial_state,
                        tf.zeros([self._batch_size, self._max_source_oovs_length], dtype=tf.float32),
                        self._max_target_sequence_length,
                        self.output_prob_list
                        ]
        )
        logits = self.output_prob_list.stack()  # decoder_max_len * batch * (vocab_size + oovs_size)
        logits = tf.transpose(logits, perm=[1, 0, 2])  # batch * decoder_max_len * (vocab_size + oovs_size)
        sample_id = tf.argmax(logits, 2)
        return logits, sample_id

    def _build_loss(self):
        target_output_one_hot = tf.one_hot(
            self._target_output, self._vocab_size + self._max_source_oovs_length)
        crossent = - tf.reduce_sum(
            target_output_one_hot * tf.log(self.logits), -1)
        nonzeros = tf.count_nonzero(self._target_mask)
        self._target_mask = tf.cast(self._target_mask, dtype=tf.float32)
        loss = (tf.reduce_sum(crossent * self._target_mask) /
                            tf.cast(nonzeros, tf.float32))
        return loss

    def _mask_logits(self, seq_mask, scores):
        '''
        to do softmax, assign -inf value for the logits of padding tokens
        '''
        score_mask_values = -1e10 * tf.ones_like(scores, dtype=tf.float32)
        return tf.where(seq_mask, scores, score_mask_values)
    
        
import tensorflow as tf
import numpy as np
import nltk

from hbconfig import Config
from models.seq2seq.Seq2SeqGenerator import Generator as Seq2SeqGenerator
from models.seq2seq.Seq2SeqHelper import *

class Generator(Seq2SeqGenerator):
    def __init__(self, dl):
        super(Generator,self).__init__(dl)

    def model_fn(self, features, labels, mode, params):
        logits = self.forward(features)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=tf.argmax(logits, -1))
            
        if mode != tf.estimator.ModeKeys.PREDICT:
            max_len = tf.argmax(labels['target_sequence_length'])
            print(logits)
            print(labels['target_input'])
            loss_op = tf.contrib.seq2seq.sequence_loss(
                logits = logits,
                targets = labels['target_input'],
                weights = tf.sequence_mask(labels['target_sequence_length'], max_len, dtype=tf.float32))
            train_op = tf.train.AdamOptimizer().apply_gradients(
                self.clip_grads(loss_op),
                global_step = tf.train.get_global_step())
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)
    
    def forward(self, features):
        inputs = features['source']
        enc_seq_len = features['source_sequence_length']
        max_len = tf.shape(enc_seq_len)[0]
        batch_sz = tf.shape(inputs)[0]
        masks = tf.to_float(tf.sign(inputs))
        
        with tf.variable_scope('Encoder'):
            embedding = tf.get_variable('lookup_table',
                                        [self._vocab_size, Config.model.embed_dim])
            enc_inp = tf.nn.embedding_lookup(embedding, inputs)
            enc_rnn_out, enc_rnn_state = tf.nn.dynamic_rnn(self.rnn_cell(),
                                                        enc_inp,
                                                        enc_seq_len,
                                                        dtype=tf.float32)
            
        with tf.variable_scope('Decoder'):
            outputs = []
            
            dec_cell = self.rnn_cell()
            W1 = tf.layers.Dense(Config.model.num_units, use_bias=False)
            W2 = tf.layers.Dense(Config.model.num_units, use_bias=False)
            v = tf.get_variable('v', [Config.model.num_units])
            
            state = enc_rnn_state
            starts = tf.fill([batch_sz], SOS_ID)
            inp = tf.nn.embedding_lookup(embedding, starts)
            
            for _ in range(max_len):
                _, state = dec_cell(inp, state)
                output = self.attention(state, enc_rnn_out, masks, W1, W2, v)
                outputs.append(output)
                idx = tf.argmax(output, -1, output_type=tf.int32)
                inp = self.point(idx, batch_sz, enc_inp)
        
        outputs = tf.stack(outputs, 1)
        return outputs

    def clip_grads(self, loss):
        variables = tf.trainable_variables()
        grads = tf.gradients(loss, variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, Config.model.max_gradient_norm)
        return zip(clipped_grads, variables)


    def rnn_cell(self):
        return tf.nn.rnn_cell.GRUCell(Config.model.num_units,
                                    kernel_initializer=tf.orthogonal_initializer())

    def point(self, idx, batch_size, enc_inp):
        return tf.gather_nd(enc_inp, tf.concat([
            tf.expand_dims(tf.range(batch_size), 1),
            tf.expand_dims(idx, 1)],
            axis=1))

    def attention(self, query, keys, masks, W1, W2, v):
        query = tf.expand_dims(query, 1)
        align = v * tf.tanh(W1(query) + W2(keys))
        align = tf.reduce_sum(align, [2])
        align *= masks
        return align
    
        
import tensorflow as tf
import numpy as np
import nltk

from hbconfig import Config
from models.seq2seq.Seq2SeqHelper import *

class Generator():
    def __init__(self, dl):
        self.dl = dl
        self._dtype = tf.float32
        self._vocab_size = dl.vocab_size        
        self._num_layers = Config.model.num_layers

    def model_fn(self, features, labels, mode):
        self._mode = mode   
        self.loss, self.train_op,  self.metrics, self.predictions = None, None, None, None

        logits, self.predictions = self._build_graph(features, labels, mode)
        
        if mode != tf.estimator.ModeKeys.PREDICT:
            self.loss = self._build_loss(logits)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.train_op = self._build_optimization()
        
        if mode == tf.estimator.ModeKeys.EVAL:
            self.metrics = self._build_metric()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions=self.predictions)
    
    def predict_fn(self, text, estimator):
        res = []      
        xs = text.split('\n')
        preds = list(estimator.predict(self.dl.predict_input_fn(xs)))
        for x, pred in zip(xs, preds):
            pred_str = ' '.join(self.dl.id_to_text(pred))
            print('IN: {}'.format(x))
            print('OUT: {}'.format(pred_str))
            res.append(pred_str)

        return '\n'.join(res)
        
    def _build_graph(self, features, labels, mode):
        self._mode = mode
        self._init_placeholder(features, labels)
        self._init_embeddings()
        self._build_projection()
        encoder_outputs, encoder_state = self._build_encoder()
        logits, sample_id = self._build_decoder(encoder_outputs, encoder_state)
        return logits, sample_id
        
    def _init_placeholder(self, features, labels):
        self._source = features['source']
        self._source_sequence_length = features['source_sequence_length']
        self._batch_size = tf.shape(self._source)[0]

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            self._target_input = labels['target_input']
            self._target_output = labels['target_output']
            self._target_sequence_length = labels['target_sequence_length']
            self._max_target_sequence_length = tf.reduce_max(self._target_sequence_length)
        else:
            max_source_sequence_length = tf.reduce_max(self._source_sequence_length)
            self._max_target_sequence_length = tf.to_int32(tf.round(tf.to_float(max_source_sequence_length) * 2.0))
            
    def _init_embeddings(self):
        with tf.variable_scope("embeddings", dtype=self._dtype):
            embedding = tf.get_variable("embedding_share", 
                                    [self._vocab_size, Config.model.embed_dim], self._dtype)
            self._embedding_encoder = embedding
            self._embedding_decoder = embedding

            self._encoder_emb_inp = tf.nn.embedding_lookup(
                self._embedding_encoder, self._source)

            if self._mode != tf.estimator.ModeKeys.PREDICT:
                self._decoder_emb_inp = tf.nn.embedding_lookup(
                    self._embedding_decoder, self._target_input)
            else:
                self._decoder_emb_inp = None

    def _build_encoder(self):
        with tf.variable_scope('encoder'):
            encoder_type = Config.model.encoder_type

            if encoder_type == "uni":
                cell = create_rnn_cell(self._num_layers,self._mode)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=cell, 
                                                            inputs=self._encoder_emb_inp, 
                                                            sequence_length=self._source_sequence_length, 
                                                            dtype=tf.float32,
                                                            swap_memory=True)
            elif encoder_type == "bi":                
                num_bi_layers = int(self._num_layers / 2)
                encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(num_bi_layers)

                if num_bi_layers == 1:
                    encoder_state  = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % self._encoder_type)

            return encoder_outputs, encoder_state
    
    def _build_bidirectional_rnn(self, num_bi_layers):
        """Create and call biddirectional RNN cells."""
        fw_cell = create_rnn_cell(num_bi_layers,self._mode)
        bw_cell = create_rnn_cell(num_bi_layers,self._mode)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            self._encoder_emb_inp,
            dtype=self._dtype,
            sequence_length=self._source_sequence_length,
            swap_memory=True)

        return tf.concat(bi_outputs, -1), bi_state
    
    def _build_projection(self):
        with tf.variable_scope("decoder/output_projection"):
            self._output_layer = tf.layers.Dense(
                self._vocab_size, use_bias=False, name="output_projection")

    def _build_decoder(self, encoder_outputs, encoder_state):        
        with tf.variable_scope('decoder'):
            cell, initial_state = self._build_decoder_cell(encoder_state, encoder_outputs)

            if self._mode != tf.estimator.ModeKeys.PREDICT:
                return self._build_decoder_train_eval(cell, initial_state)
            else:
                return self._build_decoder_infer(cell, initial_state)
    
    def _build_decoder_train_eval(self, cell, initial_state):
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs = self._decoder_emb_inp,
            sequence_length = self._target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = cell,
            helper = helper,
            initial_state = initial_state,
            output_layer = self._output_layer)

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            maximum_iterations = self._max_target_sequence_length)

        logits = decoder_output.rnn_output
        sample_id = decoder_output.sample_id

        logits_length = tf.shape(logits)[1]
        label_length = self._max_target_sequence_length
        pad_size = label_length - logits_length
        logits = tf.pad(logits, [[0, 0], [0, pad_size], [0, 0]])
        predictions = tf.argmax(logits, axis=2)
        
        return logits, predictions
    
    def _build_decoder_infer(self, cell, initial_state):
        start_tokens = tf.fill([self._batch_size], SOS_ID)
        end_token = EOS_ID

        if Config.infer.infer_mode == "beam_search":
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=self._embedding_decoder,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=initial_state,
                beam_width=Config.infer.beam_width,
                output_layer=self._output_layer,
                length_penalty_weight=Config.infer._length_penalty_weight)
        elif Config.infer.infer_mode == "greedy":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self._embedding_decoder, 
                start_tokens=start_tokens, 
                end_token=end_token)
        else:
            raise ValueError("Unknown infer_mode '%s'", Config.infer.infer_mode)
        
        if Config.infer.infer_mode != "beam_search":
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell, 
                helper, 
                initial_state,
                output_layer=self._output_layer)

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,            
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=self._max_target_sequence_length)

        if Config.infer.infer_mode == "beam_search":
            logits = tf.no_op()
            sample_id = decoder_output.predicted_ids[:, :, 0]
        else:
            logits = decoder_output.rnn_output
            sample_id = decoder_output.sample_id

        return logits, sample_id

    def _build_decoder_cell(self, encoder_state, encoder_outputs):
        if self._mode == tf.estimator.ModeKeys.PREDICT and Config.infer.infer_mode == "beam_search":
            memory = tf.contrib.seq2seq.tile_batch(encoder_outputs, Config.infer.beam_width)
            memory_sequence_length = tf.contrib.seq2seq.tile_batch(self._source_sequence_length, Config.infer.beam_width)
            cell_state = tf.contrib.seq2seq.tile_batch(encoder_state, Config.infer.beam_width)
        else:
            memory = encoder_outputs
            memory_sequence_length = self._source_sequence_length
            cell_state = encoder_state

        attention = tf.contrib.seq2seq.LuongAttention(
            num_units = Config.model.num_units,
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            scale=True)
        
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell = create_rnn_cell(self._num_layers,self._mode),
            attention_mechanism = attention,
            attention_layer_size = Config.model.num_units)

        if self._mode == tf.estimator.ModeKeys.PREDICT and Config.infer.infer_mode == "beam_search":
            initial_state = cell.zero_state(self._batch_size * Config.infer.beam_width, tf.float32).clone(
                        cell_state=cell_state)
        else:
            initial_state = cell.zero_state(self._batch_size, tf.float32).clone(cell_state=cell_state)
        
        return cell, initial_state
    
    def _build_loss(self, logits):
        weight_masks = tf.sequence_mask(
            self._target_sequence_length, 
            self._max_target_sequence_length,
            dtype=logits.dtype)

        loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=self._target_output,
                weights=weight_masks,
                name="loss")
        return loss

    def _build_optimization(self):
        train_op = tf.contrib.layers.optimize_loss(
            self.loss, 
            tf.train.get_global_step(),
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss', 'learning_rate'],
            name="train_op")
        return train_op

    def _build_metric(self):
        def blue_score(labels, predictions,
                       weights=None, metrics_collections=None,
                       updates_collections=None, name=None):
            def _nltk_blue_score(labels, predictions):
                # slice after <eos>
                predictions = predictions.tolist()
                for i in range(len(predictions)):
                    prediction = predictions[i]
                    if EOS_ID in prediction:
                        predictions[i] = prediction[:prediction.index(EOS_ID)+1]

                labels = [
                    [[w_id for w_id in label if w_id != PAD_ID]]
                    for label in labels.tolist()]
                predictions = [
                    [w_id for w_id in prediction]
                    for prediction in predictions]

                return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

            score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
            return tf.metrics.mean(score * 100)

        metrics = {
            "accuracy": tf.metrics.accuracy(self._target_output, self.predictions),
            "bleu": blue_score(self._target_output, self.predictions)
        }
        return metrics

    
        
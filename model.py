#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
FLAGS = tf.flags.FLAGS


class SummarizationModel(object):
    """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, None], name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [FLAGS.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, None], name='enc_padding_mask')

        # review part
        self._review_batch = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.review_num, None],
                                            name='review_batch')
        self._review_lens = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.review_num], name='review_lens')
        self._review_padding_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.review_num, None],
                                                   name='review_padding_mask')
        self._review_num_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.review_num], name='review_num_mask')
        self._review_scores = tf.placeholder(tf.float32, [hps.batch_size, FLAGS.review_num], name='review_scores')

        self._review_review_edge_mask = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.review_num, FLAGS.review_num],
                                            name='review_review_edge_mask')

        # qa pair
        self._qa_question_batch = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.qa_pair_num, None],
                                            name='qa_question_batch')
        self._qa_question_lens = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.qa_pair_num], name='qa_question_lens')
        self._qa_question_padding_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.qa_pair_num, None],
                                                   name='qa_question_padding_mask')

        self._qa_answer_batch = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.qa_pair_num, None],
                                            name='qa_answer_batch')
        self._qa_answer_lens = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.qa_pair_num], name='qa_answer_lens')
        self._qa_answer_padding_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.qa_pair_num, None],
                                                   name='qa_answer_padding_mask')
        
        # attribbute part
        self._attr_key_batch = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.attribute_num, FLAGS.attribute_len],
                                              name='attr_key_batch')
        self._attr_key_lens = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.attribute_num], name='attribute_key_lens')
        self._attr_key_padding_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.attribute_num, None],
                                                   name='attribute_key_padding_mask')
        
        self._attr_value_batch = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.attribute_num, FLAGS.attribute_len],
                                                name='attr_value_batch')
        self._attr_value_lens = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.attribute_num], name='attribute_value_lens')
        self._attr_value_padding_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.attribute_num, None],
                                                   name='attribute_value_padding_mask')
       
        self._attr_num_mask = tf.placeholder(tf.bool, [hps.batch_size, FLAGS.attribute_num], name='attr_num_mask')

        self._attribute_attribute_edge_mask = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.attribute_num, FLAGS.attribute_num],
                                                       name='attribute_attribute_edge_mask')
        self._review_attribute_edge_mask = tf.placeholder(tf.int32, [hps.batch_size, FLAGS.review_num, FLAGS.attribute_num],
                                                       name='review_attribute_edge_mask')

        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [FLAGS.batch_size, None],
                                                          name='enc_batch_extend_vocab')
            self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # decoder part
        self._dec_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [FLAGS.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, hps.max_dec_steps],
                                                name='dec_padding_mask')

        if "decode" in FLAGS.mode and FLAGS.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [FLAGS.batch_size, None], name='prev_coverage')

    def _make_feed_dict(self, batch, just_enc=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the encoder.
        """
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

        feed_dict[self._review_batch] = batch.review_batch
        feed_dict[self._review_lens] = batch.review_lens
        feed_dict[self._review_padding_mask] = batch.review_padding_mask
        feed_dict[self._review_num_mask] = batch.review_num_mask
        feed_dict[self._review_scores] = batch.review_scores
        feed_dict[self._attr_key_batch] = batch.attr_key_batch
        feed_dict[self._attr_key_lens] = batch.attr_key_lens
        feed_dict[self._attr_key_padding_mask] = batch.attr_key_padding_mask
        feed_dict[self._attr_value_batch] = batch.attr_value_batch
        feed_dict[self._attr_value_lens] = batch.attr_value_lens
        feed_dict[self._attr_value_padding_mask] = batch.attr_value_padding_mask
        feed_dict[self._attr_num_mask] = batch.attr_num_mask

        feed_dict[self._qa_question_batch] = batch.qa_question_batch
        feed_dict[self._qa_question_lens] = batch.qa_question_lens
        feed_dict[self._qa_question_padding_mask] = batch.qa_question_padding_mask
        feed_dict[self._qa_answer_batch] = batch.qa_answer_batch
        feed_dict[self._qa_answer_lens] = batch.qa_answer_lens
        feed_dict[self._qa_answer_padding_mask] = batch.qa_answer_padding_mask

        feed_dict[self._review_review_edge_mask] = batch.review_review_edge_mask
        feed_dict[self._review_attribute_edge_mask] = batch.review_attribute_edge_mask
        feed_dict[self._attribute_attribute_edge_mask] = batch.attribute_attribute_edge_mask

        if FLAGS.pointer_gen:
            feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self._max_art_oovs] = batch.max_art_oovs
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                                dtype=tf.float32,
                                                                                sequence_length=seq_len,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
        return encoder_outputs, fw_st, bw_st

    def _add_attr_encoder(self, attr_key_enc_input, attr_value_enc_input):
        """
        把原本k和v都是单个词的表示整合成一个向量化表示, 使用self attention对k或者v中的每个词得到一个权重，然后对每个词的embedding加权求和
        :param attr_key_enc_input: [batch, attr_num, attr_len, emb_dim]
        :param attr_value_enc_input:  [batch, attr_num, attr_len, emb_dim]
        :return:  attr_k_mem [batch, attr_num, emb_dim] , attr_v_mem [batch, attr_num, emb_dim]
        """
        with tf.variable_scope('attr_encoder'):
            kw1 = tf.get_variable('kw1', [FLAGS.emb_dim, FLAGS.hidden_dim])
            kw2 = tf.get_variable('kw2', [FLAGS.hidden_dim, 1])
            vw1 = tf.get_variable('vw1', [FLAGS.emb_dim, FLAGS.hidden_dim])
            vw2 = tf.get_variable('vw2', [FLAGS.hidden_dim, 1])
            k = tf.reshape(attr_key_enc_input,
                           [FLAGS.batch_size * FLAGS.attribute_num * FLAGS.attribute_len, FLAGS.emb_dim])
            v = tf.reshape(attr_value_enc_input,
                           [FLAGS.batch_size * FLAGS.attribute_num * FLAGS.attribute_len, FLAGS.emb_dim])
            k_logits = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(k, kw1)), kw2),
                                  [FLAGS.batch_size, FLAGS.attribute_num, FLAGS.attribute_len, 1])
            v_logits = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(v, vw1)), vw2),
                                  [FLAGS.batch_size, FLAGS.attribute_num, FLAGS.attribute_len, 1])
            k_self_attention = tf.nn.softmax(k_logits, 2)  # [batch, attr_num, attr_len, 1]
            v_self_attention = tf.nn.softmax(v_logits, 2)  # [batch, attr_num, attr_len, 1]
            attr_k_mem = tf.reduce_sum(k_self_attention * attr_key_enc_input, 2)  # [batch, attr_num, emb_dim]
            attr_v_mem = tf.reduce_sum(v_self_attention * attr_value_enc_input, 2)  # [batch, attr_num, emb_dim]
            return attr_k_mem, attr_v_mem

    def attribute_encoder(self, attribute_key_input, attribute_key_len, attribute_value_input, attribute_value_len):
        """
        encode attribute to hidden states
        :param attribute_input: [batch, attribute_num, attribute_len, emd_dim]
        :param attribute_len:  [batch, attribute_num]
        :return: encoder_outputs [batch, attribute_num, attribute_len, 2*hidden_dim], fw_st, bw_st
        """
        with tf.variable_scope('attribute_key_encoder'):
            cell_fw_key = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw_key = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            attribute_key_input = tf.reshape(attribute_key_input,
                                      [FLAGS.batch_size * FLAGS.attribute_num, FLAGS.attribute_len, FLAGS.emb_dim])
            attribute_key_len = tf.reshape(attribute_key_len, [-1])
            (encoder_outputs_key, (fw_st_key, bw_st_key)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_key, cell_bw_key, attribute_key_input,
                                                                                dtype=tf.float32,
                                                                                sequence_length=attribute_key_len,
                                                                                swap_memory=True)
            encoder_outputs_key = tf.concat(axis=2, values=encoder_outputs_key)  # concatenate the forwards and backwards states
            encoder_outputs_key = tf.reshape(encoder_outputs_key,
                                         [FLAGS.batch_size, FLAGS.attribute_num, FLAGS.attribute_len, 2 * FLAGS.hidden_dim])

        with tf.variable_scope('attribute_value_encoder'):

            cell_fw_value = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw_value = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            attribute_value_input = tf.reshape(attribute_value_input,
                                      [FLAGS.batch_size * FLAGS.attribute_num, FLAGS.attribute_len, FLAGS.emb_dim])
            attribute_value_len = tf.reshape(attribute_value_len, [-1])
            (encoder_outputs_value, (fw_st_value, bw_st_value)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_value, cell_bw_value, attribute_value_input,
                                                                                dtype=tf.float32,
                                                                                sequence_length=attribute_value_len,
                                                                                swap_memory=True)
            encoder_outputs_value = tf.concat(axis=2, values=encoder_outputs_value)  # concatenate the forwards and backwards states
            encoder_outputs_value = tf.reshape(encoder_outputs_value,
                                         [FLAGS.batch_size, FLAGS.attribute_num, FLAGS.attribute_len, 2 * FLAGS.hidden_dim])

        return encoder_outputs_key, fw_st_key, bw_st_key, encoder_outputs_value, fw_st_value, bw_st_value

    def review_encoder(self, review_input, review_len):
        """
        encode reviews to hidden states
        :param review_input: [batch, review_num, review_len, emd_dim]
        :param review_len:  [batch, review_num]
        :return: encoder_outputs [batch, review_num, review_len, 2*hidden_dim], fw_st, bw_st
        """
        with tf.variable_scope('review_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            review_input = tf.reshape(review_input,
                                      [FLAGS.batch_size * FLAGS.review_num, FLAGS.review_len, FLAGS.emb_dim])
            review_len = tf.reshape(review_len, [-1])
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, review_input,
                                                                                dtype=tf.float32,
                                                                                sequence_length=review_len,
                                                                                swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
            encoder_outputs = tf.reshape(encoder_outputs,
                                         [FLAGS.batch_size, FLAGS.review_num, FLAGS.review_len, 2 * FLAGS.hidden_dim])
        return encoder_outputs, fw_st, bw_st

    def qa_question_encoder(self, qa_question_input, qa_question_len):
        """
        encode question to hidden states
        :param qa_question_input: [batch, qa_num, qa_question_len, emd_dim]
        :param qa_question_len:  [batch, qa_num]
        :return: encoder_outputs [batch, qa_num, 2*hidden_dim]
        """
        with tf.variable_scope('qa_question_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            qa_question_input = tf.reshape(qa_question_input,
                                      [FLAGS.batch_size * FLAGS.qa_pair_num, FLAGS.max_enc_steps, FLAGS.emb_dim])
            qa_question_len = tf.reshape(qa_question_len, [-1])
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, qa_question_input,
                                                                                dtype=tf.float32,
                                                                                sequence_length=qa_question_len,
                                                                                swap_memory=True)
            question_vector = tf.concat([fw_st.c, bw_st.c], axis=1)  # [batch_size*qa_num, 2*hidden_dim]
            question_vector = tf.reshape(question_vector,
                                         [FLAGS.batch_size, FLAGS.qa_pair_num, 2 * FLAGS.hidden_dim])
        return question_vector

    def qa_answer_encoder(self, qa_answer_input, qa_answer_len):
        """
        encode answer to hidden states
        :param qa_answer_input: [batch, qa_num, qa_answer_len, emd_dim]
        :param qa_answer_len:  [batch, qa_num]
        :return: encoder_outputs [batch, qa_num, 2*hidden_dim]
        """
        with tf.variable_scope('qa_answer_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            qa_answer_input = tf.reshape(qa_answer_input,
                                      [FLAGS.batch_size * FLAGS.qa_pair_num, FLAGS.max_dec_steps, FLAGS.emb_dim])
            qa_answer_len = tf.reshape(qa_answer_len, [-1])
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, qa_answer_input,
                                                                                dtype=tf.float32,
                                                                                sequence_length=qa_answer_len,
                                                                                swap_memory=True)
            answer_vector = tf.concat([fw_st.c, bw_st.c], axis=1)  # [batch_size*qa_num, 2*hidden_dim]
            answer_vector = tf.reshape(answer_vector,
                                         [FLAGS.batch_size, FLAGS.qa_pair_num, 2 * FLAGS.hidden_dim])
        return answer_vector

    def qa_attetion(query_vector, qa_question_state, qa_answer_state):
    	'''
    	:param query_vector: [batch, 2*hidden_dim]
    	:param qa_question_state: [batch, qa_num, 2*hidden_dim]
    	:param qa_answer_state: [batch, qa_num, 2*hidden_dim]
    	return qa_memory: [batch, 2*hidden_dim]

    	'''
    	with tf.variable_scope('qa_attetion') as scope:
    		w_qa = tf.get_variable('w_qa', [2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])

    		query_vector_r = tf.reshape(query_vector, [FLAGS.batch_size, 1, 2 * FLAGS.hidden_dim])
    		qa_question_state_t = tf.transpose(qa_question_state, perm=[0, 2, 1]) # [batch_size, 2*hidden_dim, qa_num]
    		w_qa_tile = tf.reshape(tf.tile(w_qa, [FLAGS.batch_size, 1]), [FLAGS.batch_size, 2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
    		question_attention = tf.nn.softmax(tf.matmul(tf.matmul(query_vector_r, w_qa_tile), qa_question_state_t), axis=-1) # [batch_size, 1, qa_num]
    		qa_memory = tf.reshape(tf.matmul(question_attention, qa_answer_state), [FLAGS.batch_size, 2 * FLAGS.hidden_dim])

    		return qa_memory

    def query_aware_review_encoder(self, review_st, query_st, review_mask):
        """
        面向query编码review
        :param review_st: reviews hidden state [batch_size, review_num, review_len, 2*hidden_dim]
        :param query_st: query hidden state [batch_size, 4*hidden_dim]
        :param review_mask: mask [batch_size, review_num, review_len]
        :return: review vector [batch_size, review_num, 2*hidden_dim]
        """
        with tf.variable_scope('query_aware_review_encoder') as scope:
            w_q = tf.get_variable("w_q", [4 * FLAGS.hidden_dim, FLAGS.hidden_dim])
            w_r = tf.get_variable("w_r", [2 * FLAGS.hidden_dim, FLAGS.hidden_dim])
            v = tf.get_variable("v", [FLAGS.hidden_dim, 1])

            review_st_reshaped = tf.reshape(review_st, [FLAGS.batch_size * FLAGS.review_num * FLAGS.review_len, 2 * FLAGS.hidden_dim])
            query_st_reshaped = tf.reshape(tf.tile(query_st, [1, FLAGS.review_num * FLAGS.review_len]), [FLAGS.batch_size * FLAGS.review_num * FLAGS.review_len, 4 * FLAGS.hidden_dim])
            r_q_attention = tf.matmul(tf.tanh(tf.matmul(review_st_reshaped, w_r) + tf.matmul(query_st_reshaped, w_q)), v)
            r_q_attention = exp_mask(tf.reshape(r_q_attention, [FLAGS.batch_size, FLAGS.review_num, FLAGS.review_len]), review_mask)
            r_q_attention = tf.nn.softmax(r_q_attention, axis=-1)

            review_vector = review_st * tf.expand_dims(r_q_attention, dim=-1)  # [batch_size, review_num, review_len, 2*hidden_dim]
            review_vector = tf.reduce_sum(review_vector, axis=2)

        return review_vector

    def query_aware_attribute_encoder(self, query_vector, attr_value_st, attr_key_st, attr_num_mask):
        """
        面向query编码attribute
        :param query_vector: [batch_size, 4*hidden_dim]
        :param attr_value_st: [batch_size, attribute_num, 4*hidden_dim]
        :param attr_key_st: [batch_size, attribute_num, 4*hidden_dim]
        :param attr_key_mask: mask [batch_size, attribute_num]
        return attribute vector [batch_size, attribute_num, 4*hidden_dim]
        """

        with tf.variable_scope('query_aware_attribute_encoder') as scope:

            w_k = tf.get_variable("w_k", [4 * FLAGS.hidden_dim, 4 * FLAGS.hidden_dim])
            w_v = tf.get_variable("w_v", [4 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])

            attr_key_reshape = tf.matmul(tf.reshape(attr_key_st, [FLAGS.batch_size*FLAGS.attribute_num, 4 * FLAGS.hidden_dim]), w_k)
            attr_key_reshape = tf.reshape(attr_key_reshape, [FLAGS.batch_size, FLAGS.attribute_num, 4 * FLAGS.hidden_dim])

            attention_k = tf.squeeze(tf.matmul(attr_key_reshape, tf.expand_dims(query_vector, -1)))  # [batch_size, attribute_num]
            attention_k = exp_mask(attention_k, attr_num_mask)
            attention_k = tf.nn.softmax(attention_k, axis=-1)
            attention_k = tf.tile(tf.expand_dims(attention_k, -1), [1, 1, 4 * FLAGS.hidden_dim])  # [batch_size, attribute_num, 4*hidden_dim]

            attribute_vector = tf.multiply(attr_value_st, attention_k)
            attribute_vector = tf.matmul(tf.reshape(attribute_vector, [FLAGS.batch_size*FLAGS.attribute_num, 4 * FLAGS.hidden_dim]), w_v)
            attribute_vector = tf.reshape(attribute_vector, [FLAGS.batch_size, FLAGS.attribute_num, 2 * FLAGS.hidden_dim])

        return attribute_vector

    def review_fusion(self, review_vector, query_vector, review_num_mask, review_scores):
        """

        :param review_vector: [batch_size, review_num, 2*hidden_dim]
        :param query_vector: [batch_size, 2*hidden_dim]
        :param review_num_mask: [batch_size, review_num]
        :param review_scores: review的BM25权重 [batch_size, review_num]
        :return: fused_review [batch_size, 2*hidden_dim] review_gate [batch_size, review_num]
        """
        with tf.variable_scope('review_fusion') as scope:
            if FLAGS.learned_review_gates:  # 使用review和query之间的attention分数当做融合reviews的权重
                if FLAGS.review_fusion_matching:
                    query_vector_tiled = tf.tile(tf.expand_dims(query_vector, 1), [1, FLAGS.review_num, 1])
                    matching_1 = tf.concat([review_vector, query_vector_tiled],
                                           2)  # [batch_size, review_num, 4*hidden_dim]
                    matching_2 = review_vector * query_vector_tiled  # [batch_size, review_num, 2*hidden_dim]
                    matching_3 = tf.abs(review_vector - query_vector_tiled)  # [batch_size, review_num, 2*hidden_dim]
                    matching = tf.concat([matching_1, matching_2, matching_3], 2)
                    bw_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_dim)
                    fw_cell = tf.contrib.rnn.GRUCell(FLAGS.hidden_dim)
                    (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, matching,
                                                                                     dtype=tf.float32)
                    hidden = tf.concat([output_bw, output_fw], 2)  # [batch_size, review_num, 2*hidden_dim]
                    logits = tf.layers.dense(tf.reshape(hidden, [-1, 2 * FLAGS.hidden_dim]),
                                             1)  # [batch_size * review_num]
                    logits = tf.reshape(logits, [FLAGS.batch_size, FLAGS.review_num])
                    gates = tf.nn.softmax(logits, 1)
                else:
                    w = tf.get_variable('review_fusion', [2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    imd = tf.matmul(tf.reshape(review_vector, [-1, 2 * FLAGS.hidden_dim]), w)
                    # imd [batch_size * review_num, 2*hidden_dim]
                    imd = tf.reshape(imd, [FLAGS.batch_size, FLAGS.review_num, 2 * FLAGS.hidden_dim])
                    logits = tf.matmul(imd, tf.expand_dims(query_vector, axis=1), transpose_b=True)
                    # logits [batch_size, review_num, 1]
                    logits = exp_mask(tf.squeeze(logits), review_num_mask)  # [batch_size, review_num]
                    gates = tf.nn.softmax(logits, -1)  # [batch_size, review_num]
                    # distrib_gates = tf.distributions.Categorical(logits=logits)
                    # distrib_bm25_score = tf.distributions.Categorical(logits=review_scores)
                    # self.review_score_gates_kl = tf.reduce_mean(
                    #     tf.distributions.kl_divergence(distrib_gates, distrib_bm25_score))
                normalize_logits = tf.nn.l2_normalize(logits, 1)
                normalize_review_scores = tf.nn.l2_normalize(review_scores, 1)
                self.review_score_gates_kl = tf.losses.cosine_distance(normalize_review_scores, normalize_logits, 1)
                tf.summary.scalar('loss/review_score_gates_kl', self.review_score_gates_kl)
            else:  # 使用BM25的匹配分数当做融合reviews权重
                gates = review_scores
            fused_review = tf.reduce_sum(review_vector * tf.expand_dims(gates, axis=-1), axis=1)
        return fused_review, gates

    def attr_fusion(self, attr_k_mem, attr_v_mem, query_vector, attr_num_mask):
        """
        使用query向量去attention每个attr的key，然后得出一个关于每个attr的分布，然后用这个attention分布去加权求和attr的value
        :param attr_k_mem: [batch, attr_num, emb_dim]
        :param attr_v_mem: [batch, attr_num, emb_dim]
        :param query_vector: [batch_size, 2*hidden_dim]
        :param attr_num_mask: [batch, attr_num]
        :return: attr_attention [batch, attr_num], fused_attribute [batch, emb_dim]
        """
        with tf.variable_scope('attr_fusion'):
            w = tf.get_variable('w', [FLAGS.emb_dim, 2 * FLAGS.hidden_dim])
            k = tf.reshape(attr_k_mem, [FLAGS.batch_size * FLAGS.attribute_num, FLAGS.emb_dim])
            query_vector = tf.expand_dims(query_vector, 1)  # [batch_size, 1, 2*hidden_dim]
            tiled_query = tf.tile(query_vector,
                                  [1, FLAGS.attribute_num, 1])  # [batch_size, attribute_num, 2*hidden_dim]
            query_vector = tf.expand_dims(tiled_query, 2)  # [batch_size, attribute_num, 1, 2*hidden_dim]
            imd = tf.nn.relu(tf.matmul(k, w))  # [batch * attribute_num, 2*hidden_dim]
            imd = tf.reshape(imd, [FLAGS.batch_size, FLAGS.attribute_num, 2 * FLAGS.hidden_dim])
            imd = tf.expand_dims(imd, -1)  # [batch, attribute_num, 2*hidden_dim, 1]
            attr_attention = tf.squeeze(tf.matmul(query_vector, imd))  # [batch, attr_num]
            attr_attention = tf.nn.softmax(exp_mask(attr_attention, attr_num_mask), 1)  # [batch, attr_num]
            fused_attribute = tf.reduce_sum(tf.expand_dims(attr_attention, -1) * attr_v_mem, 1)
            return attr_attention, fused_attribute

    def _reduce_states(self, fw_st, bw_st, review_state):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = FLAGS.hidden_dim
        with tf.variable_scope('reduce_final_st'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 4, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 4, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c, review_state])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h, review_state])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def _reduce_states_node(self, fw_st, bw_st, node_state):
        """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = FLAGS.hidden_dim
        with tf.variable_scope('reduce_final_st_node'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 8, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 8, hidden_dim], dtype=tf.float32,
                                         initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            # Apply linear layer
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c, node_state])  # Concatenation of fw and bw cell
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h, node_state])  # Concatenation of fw and bw state
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def _add_decoder(self, inputs, fused_review=None, fused_attribute=None, reuse=False, node_states=None,
                     query_vector=None, node_mask=None, qa_memory_state=None):
        """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

        Args:
          inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

        Returns:
          outputs: List of tensors; the outputs of the decoder
          out_state: The final state of the decoder
          attn_dists: A list of tensors; the attention distributions
          p_gens: A list of scalar tensors; the generation probabilities
          coverage: A tensor, the current coverage vector
        """
        cell = tf.contrib.rnn.LSTMCell(FLAGS.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if "decode" in FLAGS.mode and FLAGS.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

        rr = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell,
                               initial_state_attention=("decode" in FLAGS.mode), pointer_gen=FLAGS.pointer_gen,
                               use_coverage=FLAGS.coverage, prev_coverage=prev_coverage, fused_review=fused_review,
                               fused_attribute=fused_attribute, node_states=node_states,
                               node_mask=node_mask, query_vector=query_vector, qa_memory_vector=qa_memory_state)
        outputs, out_state, attn_dists, p_gens, coverage, self.review_attn_dist = rr
        return outputs, out_state, attn_dists, p_gens, coverage

    def _calc_final_dist(self, vocab_dists, attn_dists):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

        Returns:
          final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
            attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            extended_vsize = self._vocab.size() + self._max_art_oovs  # the maximum (over the batch) size of the extended vocabulary
            extra_zeros = tf.zeros((FLAGS.batch_size, self._max_art_oovs))
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
                                    vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            batch_nums = tf.range(0, limit=FLAGS.batch_size)  # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
            indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
            shape = [FLAGS.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
                                    attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                           zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists

    def _add_emb_vis(self, embedding_var):
        """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
        https://www.tensorflow.org/get_started/embedding_viz
        Make the vocab metadata file, then make the projector config file pointing to it."""
        train_dir = os.path.join(FLAGS.log_root, "train")
        vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
        self._vocab.write_metadata(vocab_metadata_path)  # write metadata file
        summary_writer = tf.summary.FileWriter(train_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = vocab_metadata_path
        projector.visualize_embeddings(summary_writer, config)

    def subgraph_encoder(self, node1_embedding, node2_embedding, edge_mask):
        """
        node attention编码subgraph
        :param node1_embedding: [batch_size, node1_num, 2*hidden_dim]
        :param node2_embedding: [batch_size, node2_num, 2*hidden_dim]
        :param edge_mask: [batch_size, node1_num, node2_num]
        return node vector [batch_size, node1_num, 2*hidden_dim], [batch_size, node2_num, 2*hidden_dim]
        """
        w_n = tf.get_variable('w_n', [2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
        w_n_tile = tf.reshape(tf.tile(w_n, [FLAGS.batch_size, 1]), [FLAGS.batch_size, 2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])

        node2_t = tf.transpose(node2_embedding, perm=[0, 2, 1]) # [batch_size, 2*hidden_dim, node2_num]
        node1_attention = tf.matmul(tf.matmul(node1_embedding, w_n_tile), node2_t)
        node1_attention_mask = exp_mask(node1_attention, edge_mask)
        node1_attention_mask = tf.nn.softmax(node1_attention_mask, axis=-1) # [batch_size, node1_num, node2_num]
        node1_vector = tf.matmul(node1_attention_mask, node2_embedding) # [batch_size, node1_num, 2*hidden_dim]
        node1_fusion = tf.concat([tf.multiply(node1_embedding, node1_vector), node1_embedding, node1_vector], 2)
        # [batch_size, node1_num, 2*hidden_dim *3]

        node1_t = tf.transpose(node1_embedding, perm=[0, 2, 1]) # [batch_size, 2*hidden_dim, node1_num]
        node2_attention = tf.matmul(tf.matmul(node2_embedding, w_n_tile), node1_t)
        node2_attention_mask = exp_mask(node2_attention, tf.transpose(edge_mask, perm=[0, 2, 1]))
        node2_attention_mask = tf.nn.softmax(node2_attention_mask, axis=-1) # [batch_size, node2_num, node1_num]
        node2_vector = tf.matmul(node2_attention_mask, node1_embedding) # [batch_size, node2_num, 2*hidden_dim]
        node2_fusion = tf.concat([tf.multiply(node2_embedding, node2_vector), node2_embedding, node2_vector], 2)
        # [batch_size, node2_num, 2*hidden_dim *3]

        return node1_fusion, node2_fusion

    def subgraph_integration(self, query_vector, subgraph1_embedding, subgraph2_embedding):
        """
        subgraph_integration
        :param query_vector: [batch_size, 2*hidden_dim]
        :param subgraph1_embedding: [batch_size, node_num, 2*hidden_dim*3]
        :param subgraph2_embedding: [batch_size, node_num, 2*hidden_dim*3]
        return integration vector [batch_size, node_num, 2*hidden_dim*3]
        """
        w_s = tf.get_variable('w_s', [2 * FLAGS.hidden_dim, 3 * 2 * FLAGS.hidden_dim])
        w_s_tile = tf.reshape(tf.tile(w_s, [FLAGS.batch_size, 1]), [FLAGS.batch_size, 2 * FLAGS.hidden_dim, 3 * 2 * FLAGS.hidden_dim])
        b_s = tf.get_variable('b_s', [1])

        query_vector_expand = tf.expand_dims(query_vector, axis=1) # [batch_size, 1, 2*hidden_dim]

        subgraph1_embedding_t = tf.transpose(subgraph1_embedding, perm=[0,2,1]) # [batch_size, 2*hidden_dim*3, node_num]
        attention1 = tf.reshape(tf.matmul(query_vector_expand, tf.tanh(tf.matmul(w_s_tile, subgraph1_embedding_t) + b_s)), 
            [FLAGS.batch_size, -1, 1])
        subgraph2_embedding_t = tf.transpose(subgraph2_embedding, perm=[0,2,1]) # [batch_size, 2*hidden_dim*3, node_num]
        attention2 = tf.reshape(tf.matmul(query_vector_expand, tf.tanh(tf.matmul(w_s_tile, subgraph2_embedding_t) + b_s)), 
            [FLAGS.batch_size, -1, 1])
        subgraph_attention = tf.reshape(tf.concat([attention1, attention2], 2), [-1, 1, 2]) # [batch_size*node_num, 1, 2]
        subgraph_attention = tf.nn.softmax(subgraph_attention, axis=-1) 

        tmp_embedding = tf.reshape(tf.concat([subgraph1_embedding, subgraph2_embedding], 2), [-1, 2, 3 * 2 * FLAGS.hidden_dim])
        integration_vector = tf.reshape(tf.matmul(subgraph_attention, tmp_embedding), 
            [FLAGS.batch_size, -1, 3 * 2 * FLAGS.hidden_dim])
        # [batch_size, node_num, 2*hidden_dim*3]

        return integration_vector

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        vsize = self._vocab.size()  # size of the vocabulary

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-FLAGS.rand_unif_init_mag, FLAGS.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=FLAGS.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, FLAGS.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)
                if FLAGS.mode == "train": self._add_emb_vis(embedding)  # add to tensorboard
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
                emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]
                review_enc_input = tf.nn.embedding_lookup(embedding, self._review_batch)
                # qa_pair_question_input = tf.nn.embedding_lookup(embedding, self._qa_question_batch)
                # qa_pair_answer_input = tf.nn.embedding_lookup(embedding, self._qa_answer_batch)
                # [batch, review_num, review_len, emd_dim]
                attr_key_enc_input = tf.nn.embedding_lookup(embedding, self._attr_key_batch)
                attr_value_enc_input = tf.nn.embedding_lookup(embedding, self._attr_value_batch)
                self.truth_emb = tf.nn.embedding_lookup(embedding, self._dec_batch)

            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
            self.query_vector = tf.concat([fw_st, bw_st], axis=-1)  # [batch_size, 4*hidden_dim]
            # self.query_vector = tf.reduce_mean(enc_outputs, axis=1)  # [batch_size, 2*hidden_dim]

            # review
            self.review_hidden_state, _, _ = self.review_encoder(review_enc_input, self._review_lens)
            review_embedding = self.query_aware_review_encoder(self.review_hidden_state, self.query_vector, self._review_padding_mask)

            # attr
            _, key_fw_st, key_bw_st, _, value_fw_st, value_bw_st = self.attribute_encoder(attr_key_enc_input, self._attr_key_lens, attr_value_enc_input, self._attr_value_lens)
            attr_key_vector = tf.reshape(tf.concat([key_fw_st, key_bw_st], axis=-1), [FLAGS.batch_size, FLAGS.attribute_num, 4 * FLAGS.hidden_dim])
            attr_value_vector = tf.reshape(tf.concat([value_fw_st, value_bw_st], axis=-1), [FLAGS.batch_size, FLAGS.attribute_num, 4 * FLAGS.hidden_dim])

            attribute_embedding = self.query_aware_attribute_encoder(self.query_vector, attr_key_vector, attr_value_vector, self._attr_num_mask)

            for layer in range(FLAGS.graph_layer):

                # rr
                with tf.variable_scope('subgraph_rr'):

                    if layer > 0:
                        tf.variable_scope.get_variable_scope().reuse_variables()

                    w_rr = tf.get_variable('w_rr', [2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    w_rr_tile = tf.reshape(tf.tile(w_rr, [FLAGS.batch_size, 1]), [FLAGS.batch_size, 2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    review_embedding_project_rr = tf.matmul(review_embedding, w_rr_tile)
                    review_embedding_rr, _ = self.subgraph_encoder(review_embedding_project_rr,
                        review_embedding_project_rr, self._review_review_edge_mask)

                # aa
                with tf.variable_scope('subgraph_aa'):

                    if layer > 0:
                        tf.variable_scope.get_variable_scope().reuse_variables()

                    w_aa = tf.get_variable('w_aa', [2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    w_aa_tile = tf.reshape(tf.tile(w_aa, [FLAGS.batch_size, 1]), [FLAGS.batch_size, 2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    attribute_embedding_project_aa = tf.matmul(attribute_embedding, w_aa_tile)
                    attribute_embedding_aa, _ = self.subgraph_encoder(attribute_embedding_project_aa,
                        attribute_embedding_project_aa, self._attribute_attribute_edge_mask)

                # ra
                with tf.variable_scope('subgraph_ra'):

                    if layer > 0:
                        tf.variable_scope.get_variable_scope().reuse_variables()

                    w_ra = tf.get_variable('w_ra', [2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    w_ra_tile = tf.reshape(tf.tile(w_ra, [FLAGS.batch_size, 1]), [FLAGS.batch_size, 2 * FLAGS.hidden_dim, 2 * FLAGS.hidden_dim])
                    attribute_embedding_project_ra = tf.matmul(attribute_embedding, w_ra_tile)
                    review_embedding_project_ra = tf.matmul(review_embedding, w_ra_tile)
                    review_embedding_ra, attribute_embedding_ra = self.subgraph_encoder(review_embedding_project_ra,
                        attribute_embedding_project_ra, self._review_attribute_edge_mask)

                # subgraph integration
                with tf.variable_scope('subgraph_integration_review') as scope:
                    if layer > 0:
                        tf.variable_scope.get_variable_scope().reuse_variables()
                    review_node_embedding = self.subgraph_integration(self.query_vector, review_embedding_rr, review_embedding_ra)
                with tf.variable_scope('subgraph_integration_attribute') as scope:
                    if layer > 0:
                        tf.variable_scope.get_variable_scope().reuse_variables()
                    attribute_node_embedding = self.subgraph_integration(self.query_vector, attribute_embedding_aa, attribute_embedding_ra)

                review_embedding = review_node_embedding
                attribute_embedding = attribute_node_embedding

            node_embedding = tf.concat([review_node_embedding, attribute_node_embedding], 1) # [batch_size, review_num+attribute_num, 2*hidden_dim*3]
            node_mask_map = tf.concat([self._review_num_mask, self._attr_num_mask], 1)
            

            self._enc_states = enc_outputs

            node_embedding_initial = tf.reduce_sum(node_embedding, 1) # [batch, 2* hidden*3]
            self._dec_in_state = self._reduce_states_node(fw_st, bw_st, node_embedding_initial)

            decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(
                emb_dec_inputs, node_states=node_embedding, query_vector=self.query_vector, node_mask=node_mask_map)

            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w', [FLAGS.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
                vocab_scores = []
                for i, output in enumerate(decoder_outputs):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer

                vocab_dists = [tf.nn.softmax(s) for s in
                               vocab_scores]  # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
            else:  # final distribution is just vocabulary distribution
                final_dists = vocab_dists

            if FLAGS.mode in ['train', 'eval']:
                # Calculate the loss
                with tf.variable_scope('loss'):
                    if FLAGS.pointer_gen:
                        # Calculate the loss per step
                        # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                        loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                        batch_nums = tf.range(0, limit=FLAGS.batch_size)  # shape (batch_size)
                        for dec_step, dist in enumerate(final_dists):
                            targets = self._target_batch[:, dec_step]
                            indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
                            self.gold_probs = tf.gather_nd(dist,
                                                      indices)  # shape (batch_size). prob of correct words on this step
                            losses = -tf.log(self.gold_probs + 1e-10)
                            loss_per_step.append(losses)

                        # Apply dec_padding_mask and get loss
                        self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

                    else:  # baseline model
                        self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
                                                                      self._target_batch,
                                                                      self._dec_padding_mask)  # this applies softmax internally

                    # Calculate coverage loss from the attention distributions
                    if FLAGS.coverage:
                        with tf.variable_scope('coverage_loss'):
                            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
                            tf.summary.scalar('coverage_loss', self._coverage_loss)
                        self._total_loss = self._loss + FLAGS.cov_loss_wt * self._coverage_loss

        if "decode" in FLAGS.mode:
            # We run decode beam search mode one decoder step at a time
            # print(final_dists)
            assert len(
                final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(final_dists,
                                                     FLAGS.batch_size * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
            self._topk_log_probs = tf.log(topk_probs)

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        tvars = tf.trainable_variables()
        # g_variables = [t for t in tvars if not t.name.startswith('seq2seq/decoder/discriminator')]
        # d_variables = [t for t in tvars if t.name.startswith('seq2seq/decoder/discriminator')]
        # tf.logging.info('g_variables %d, d_variables %d', len(g_variables), len(d_variables))
        tf.summary.scalar('loss/perplexity', tf.exp(self._loss))

        # tf.summary.scalar('loss/d_scores', self.discriminator_score_d)
        # tf.summary.scalar('loss/g_scores', self.discriminator_score_g)
        # loss_to_minimize += tf.stop_gradient(self.discriminator_score_g)
        # loss_to_minimize += self.discriminator_score_g
        # d_gradients = tf.gradients(self.discriminator_score_d, d_variables,
        #                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        # d_gradients, d_global_norm = tf.clip_by_global_norm(d_gradients, self._hps.max_grad_norm)
        # self._d_train_op = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(d_gradients, d_variables),
        #                                                                 name='d_train_step')
        # self._d_clip_op = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_variables]

        tf.summary.scalar('loss/minimize_loss', loss_to_minimize)
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        if FLAGS.plot_gradients:
            for grad, var in zip(gradients, tvars):
                tf.summary.histogram(var.name + '/gradient', grad)

        # Clip the gradients
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
            tf.summary.scalar('loss/global_norm', global_norm)

        learning_rate = tf.train.polynomial_decay(FLAGS.lr, self.global_step,
                                                  FLAGS.dataset_size / FLAGS.batch_size * 5,
                                                  FLAGS.lr / 10)
        tf.summary.scalar('loss/learning_rate', learning_rate)
        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=FLAGS.adagrad_init_acc)
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        tf.logging.info('Building graph...')
        t0 = time.time()
        self._add_placeholders()
        self.global_epoch = tf.get_variable('epoch_num', [], initializer=tf.constant_initializer(1, tf.int32),
                                            trainable=False, dtype=tf.int32)
        self.add_epoch_op = tf.assign_add(self.global_epoch, 1)
        with tf.device("/gpu:%d" % FLAGS.device if FLAGS.device != '' and FLAGS.device >= 0 else "/cpu:0"):
            self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch, summary=False):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'loss': self._loss,
            'global_step': self.global_step,
            'global_epoch': self.global_epoch,
            'gold_probs': self.gold_probs
        }
        # to_return['d_train_op'] = self._d_train_op
        # to_return['d_clip_op'] = self._d_clip_op
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        if summary:
            to_return['summaries'] = self._summaries

        result = sess.run(to_return, feed_dict)
        return result

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }
        if self._hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch (for beam search)

        Returns:
          enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders
        fetches = {
            '_enc_states': self._enc_states,
            '_dec_in_state': self._dec_in_state,
            'global_step': self.global_step,
        }
        fetches.update({
            'review_states': self.review_hidden_state,
            'query_vector': self.query_vector,
        })
        fetched = sess.run(fetches, feed_dict)
        enc_states = fetched['_enc_states']
        dec_in_state = fetched['_dec_in_state']
        global_step = fetched['global_step']
        review_states = fetched['review_states']
        query_vector = fetched['query_vector']
        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        return enc_states, dec_in_state, query_vector, review_states

    def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, fused_review=None,
                       fused_attribute=None, review_states=None, query_vector=None):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          batch: Batch object containing single example repeated across the batch
          latest_tokens: Tokens to be fed as input into the decoder for this timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
        """

        beam_size = len(dec_init_states)

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed = {
            self._enc_states: enc_states,
            self._enc_padding_mask: batch.enc_padding_mask,
            self._dec_in_state: new_dec_in_state,
            self._dec_batch: np.transpose(np.array([latest_tokens])),
            self._review_num_mask: batch.review_num_mask,
            self._review_review_edge_mask: batch.review_review_edge_mask,
            self._review_attribute_edge_mask: batch.review_attribute_edge_mask,
            self._attribute_attribute_edge_mask: batch.attribute_attribute_edge_mask,
            self._attr_key_batch: batch.attr_key_batch,
            self._attr_key_lens: batch.attr_key_lens,
            self._attr_key_padding_mask: batch.attr_key_padding_mask,
            self._attr_value_batch: batch.attr_value_batch,
            self._attr_value_lens: batch.attr_value_lens,
            self._attr_value_padding_mask: batch.attr_value_padding_mask,
            self._attr_num_mask: batch.attr_num_mask,
        }

        feed[self.review_hidden_state] = review_states
        feed[self._review_padding_mask] = batch.review_padding_mask
        feed[self.query_vector] = query_vector
        # feed[self.fused_attribute] = fused_attribute

        to_return = {
            "ids": self._topk_ids,
            "probs": self._topk_log_probs,
            "states": self._dec_out_state,
            "attn_dists": self.attn_dists
        }

        if FLAGS.pointer_gen:
            feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed[self._max_art_oovs] = batch.max_art_oovs
            to_return['p_gens'] = self.p_gens

        if self._hps.coverage:
            feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(beam_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if FLAGS.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(beam_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == beam_size
        else:
            new_coverage = [None for _ in range(beam_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage

    def discriminator(self, fake_decoder_outputs, decoder_outputs, fused_review, fused_attribute):
        """
        a CNN based discriminator
        :param fake_decoder_outputs: list of [batch_size, hidden_dim]
        :param decoder_outputs: list of [batch_size, hidden_dim]
        :param fused_review: [batch_size, 2*hidden_dim]
        :param fused_attribute: [batch, emb_dim]
        :return: score []
        """
        with tf.variable_scope('discriminator') as scope:
            tf.logging.info(scope.name)
            fake_decoder_outputs = tf.concat([tf.expand_dims(t, 1) for t in fake_decoder_outputs], 1)
            decoder_outputs = tf.concat([tf.expand_dims(t, 1) for t in decoder_outputs], 1)
            # [batch_size, max_len, hidden_dim]

            # 1D Convolutional
            # fake_features = tf.layers.conv1d(fake_decoder_outputs, FLAGS.hidden_dim, [2], padding='same',
            #                                  activation=tf.nn.relu, name='d_conv')
            # decoder_features = tf.layers.conv1d(decoder_outputs, FLAGS.hidden_dim, [2], padding='same',
            #                                     activation=tf.nn.relu, name='d_conv', reuse=True)
            # [batch_size, max_len, hidden_dim]

            # 2D Convolutional
            kernel_size = 2  # means capture 2-gram features
            bias = tf.get_variable('bias', [1, FLAGS.hidden_dim])
            review_w = tf.get_variable('review_w', [2 * FLAGS.hidden_dim, FLAGS.hidden_dim])
            attribute_w = tf.get_variable('attribute_w', [FLAGS.emb_dim, FLAGS.hidden_dim])
            fused_review = tf.matmul(fused_review, review_w)  # [batch_size, hidden_dim]
            fused_attribute = tf.matmul(fused_attribute, attribute_w)  # [batch_size, hidden_dim]

            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,
                                              state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.truth_emb,
                                                                                dtype=tf.float32, swap_memory=True)
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # concatenate the forwards and backwards states
            truth_projection = tf.get_variable('truth_projection', [2 * FLAGS.hidden_dim, FLAGS.hidden_dim])
            truth_encode = tf.matmul(tf.reshape(encoder_outputs, [-1, 2 * FLAGS.hidden_dim]),
                                     truth_projection)  # [batch_size*max_enc_steps, hidden_dim]
            truth_encode = tf.reshape(truth_encode, [-1, FLAGS.max_dec_steps, FLAGS.hidden_dim])

            def classify(data, reuse=False, batch_norm=False):
                """

                :param data: [batch_size, max_len, hidden_dim]
                :param reuse: bool
                :param batch_norm: bool
                :return:
                """
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    conv = conv2d(tf.expand_dims(data, 1), FLAGS.hidden_dim, 1, kernel_size)
                    # [batch_size, 1, enc_len-1, hidden_dim]
                    if batch_norm:
                        conv = tf.layers.batch_normalization(conv)
                    pooled = tf.nn.max_pool(tf.nn.relu(conv),
                                            [1, 1, data.get_shape()[1].value - kernel_size + 1, 1],
                                            [1, 1, 1, 1], 'VALID')  # [batch, 1, 1, hidden_dim]
                    pooled = tf.squeeze(pooled)
                    fused = tf.nn.relu(pooled + fused_review + fused_attribute + bias)
                    # [batch_size, hidden_dim]
                    dense = tf.layers.dense(inputs=fused, units=int(FLAGS.hidden_dim / 2), activation=tf.nn.relu,
                                            name='dense1')
                    # [batch_size, hidden_dim/2]
                    if batch_norm:
                        dense = tf.layers.batch_normalization(dense)
                    # dropout = tf.layers.dropout(inputs=dense, rate=FLAGS.dropout, training=FLAGS.mode == 'train')
                    logits = tf.layers.dense(dense, 1, name='output')
                    # [batch_size, 1]
                    return logits

            fake_logits = classify(fake_decoder_outputs, batch_norm=FLAGS.wgan)
            decoder_logits = classify(decoder_outputs, reuse=True, batch_norm=FLAGS.wgan)
            truth_logits = classify(truth_encode, reuse=True, batch_norm=FLAGS.wgan)

            if not FLAGS.wgan:
                decoder_logits = tf.nn.sigmoid(decoder_logits)
                fake_logits = tf.nn.sigmoid(fake_logits)
                truth_logits = tf.nn.sigmoid(truth_logits)

                labels = tf.concat(
                    [tf.zeros([FLAGS.batch_size]), tf.zeros([FLAGS.batch_size]), tf.ones([FLAGS.batch_size])], 0)
                logits = tf.concat([tf.squeeze(decoder_logits), tf.squeeze(fake_logits), tf.squeeze(truth_logits)], 0)
                tf.summary.histogram('discrimiantor_logits', tf.sigmoid(logits))
                self.discriminator_pred = tf.greater(tf.sigmoid(logits), 0.5)
                self.discriminator_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(self.discriminator_pred, tf.cast(labels, tf.bool)), tf.float32))
                tf.summary.scalar('discriminator_accuracy', self.discriminator_accuracy)

                d_score = tf.reduce_mean(tf.log(1. - decoder_logits) + tf.log(1. - fake_logits) + tf.log(truth_logits))
                g_score = -tf.reduce_mean(tf.log(decoder_logits))
                # d_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                # d_score = tf.reduce_mean(d_score)
                # g_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([FLAGS.batch_size]),
                #                                                   logits=tf.squeeze(decoder_logits))
                # g_score = tf.reduce_mean(g_score)
                return d_score, g_score
            else:
                # disc_cost = tf.reduce_mean(fake_logits) + tf.reduce_mean(decoder_logits) - tf.reduce_mean(truth_logits)
                # gen_cost = -tf.reduce_mean(decoder_logits)
                disc_cost = tf.reduce_mean(truth_logits - decoder_logits - fake_logits)
                gen_cost = tf.reduce_mean(decoder_logits)
                if FLAGS.wgan_gp:
                    alpha = tf.random_uniform(shape=[FLAGS.batch_size, 1, 1], minval=0., maxval=1.)
                    interpolates = alpha * decoder_outputs + ((1 - alpha) * truth_encode)
                    disc_interpolates = classify(interpolates, True)
                    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
                    tf.summary.scalar("gp_loss", gradient_penalty)
                    disc_cost += FLAGS.gp_lambda * gradient_penalty
                return disc_cost, gen_cost

def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor
    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def conv2d(input_, output_dim, k_h, k_w, name="conv2d", reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

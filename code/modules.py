# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

def truncated_normal_var(name, shape, dtype):
    return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))

def zero_var(name, shape, dtype):
    return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class RNNModelEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob,model_name="RNNModelEncoder"):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
        self.model_name=model_name

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(self.model_name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

class BiDAFAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys,keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        # values=questions	#(batch_size,question_hidden,2*hidden_size)
        # keys=context		#(batch_size,context_hidden,2*hidden_size)
        with vs.variable_scope("BiDAFAttn"):
            values_len=values.get_shape().as_list()[1]
            keys_len=keys.get_shape().as_list()[1]
            keys_resized = tf.expand_dims(keys,2)  # (batch_size,1,context_hidden,2*hidden_size)
            #print("keys_resized shape %s" % (keys_resized.get_shape()))
            values_resized = tf.expand_dims(values,1)  # (batch_size,question_hidden,1,2*hidden_size)
            #print("values_resized shape %s" % (values_resized.get_shape()))
            value_keys_product = values_resized * keys_resized  # (batch_size,context_hidden,question_hidden,2*hidden_size)
            #print("value_keys_product shape %s" % (value_keys_product.get_shape()))

            #print("keys shape %s" % (keys.get_shape()))
            w_sim_1=truncated_normal_var(name='w_sim_1',shape=[self.key_vec_size],dtype=tf.float32)    #(hidden*2,1)
            #print("w_sim_1 shape %s" % (w_sim_1.get_shape()))
            sim_value1=keys*w_sim_1     #(batch_size,context_hidden,2*hidden_size)
            #print("sim_value1 shape %s" % (sim_value1.get_shape()))
            sim_value1=tf.reduce_sum(sim_value1,[2],keep_dims=True)    #(batch_size,context_hidden,1)
            #print("sim_value1 redim %s" % (sim_value1.get_shape()))
            sim_value1=tf.tile(sim_value1,[1,1,values_len])   #(batch_size,context_hidden,question_hidden)
            #print("sim_value1 tiled shape %s" % (sim_value1.get_shape()))

            w_sim_2=truncated_normal_var(name='w_sim_2',shape=[self.value_vec_size],dtype=tf.float32)
            #print("w_sim_2 shape %s" % (w_sim_2.get_shape()))
            sim_value2=values*w_sim_2     #(batch_size,question_hidden,2*hidden)
            #print("sim_value2 shape %s" % (sim_value2.get_shape()))
            sim_value2=tf.reduce_sum(sim_value2,[2],keep_dims=True)
            #print("sim_value2 redim %s" % (sim_value2.get_shape()))
            sim_value2=tf.transpose(sim_value2, perm=[0, 2,1])      #(batch_size,1,question_hidden)
            #print("sim_value2 transpose shape %s" % (sim_value2.get_shape()))
            sim_value2 = tf.tile(sim_value2, [1, keys_len, 1])  # (batch_size,context_hidden,question_hidden)
            #print("sim_value2 tiled shape %s" % (sim_value2.get_shape()))

            w_sim_3 = truncated_normal_var(name='w_sim_3', shape=[self.value_vec_size/2+self.key_vec_size/2], dtype=tf.float32) #(hidden_size*2)
            #print("w_sim_3 shape %s" % (w_sim_3.get_shape()))
            sim_value3=value_keys_product*w_sim_3   #(batch_size,context_hidden,question_hidden,2*hidden_size)
            #print("sim_value3 shape %s" % (sim_value3.get_shape()))
            sim_value3=tf.reduce_sum(sim_value3,[3])    #(batch_size,context_hidden,question_hidden)
            #print("sim_value3 reduced shape %s" % (sim_value3.get_shape()))

            sim_matrix=sim_value1+sim_value2+sim_value3     #(batch_size,context_hidden,question_hidden)
            #print("sim_matrix reduced shape %s" % (sim_matrix.get_shape()))

            values_mask_resized=tf.expand_dims(values_mask,1)   #(batch_size,1,question_hidden)
            #print("values_mask_resized shape %s" % (values_mask_resized.get_shape()))
            _, attn_dist = masked_softmax(sim_matrix,values_mask_resized,2) #(batch_size,context_hidden,question_hidden)
            #print("attn_dist shape %s" % (attn_dist.get_shape()))
            a_sub_i= tf.matmul(attn_dist, values)   #(batch_size,context_hidden,2*hiden_size)
            #print("a_sub_i shape %s" % (a_sub_i.get_shape()))
            a_sub_i_cross_c_i=a_sub_i*keys    #(batch_size,context_hidden,2*hiden_size)
            #print("a_sub_i_cross_c_i final shape %s" % (a_sub_i_cross_c_i.get_shape()))

            m_matrix = tf.reduce_max(sim_matrix, axis=2)    #(batch_size,context_hidden)
            #print("m_matrix shape %s" % (m_matrix.get_shape()))
            #print("keys_mask shape %s"%(keys_mask.get_shape()))
            _, Beta= masked_softmax(m_matrix, keys_mask, 1)  # shape (batch_size, context_hidden)
            #print("Beta shape %s" % (Beta.get_shape()))
            Beta=tf.expand_dims(Beta,2)     # shape (batch_size, context_hidden)
            #print("Beta redim shape %s" % (Beta.get_shape()))
            c_prime = keys * Beta           # shape (batch_size, context_hidden,2*hidden_size)
            #print("c_prime shape %s" % (c_prime.get_shape()))
            c_prime=tf.reduce_sum(c_prime,axis=1,keep_dims=True)    # shape (batch_size, context_hidden,2*hidden_size)
            #print("c_prime reduced shape %s" % (c_prime.get_shape()))
            c_prime=keys*c_prime
            #print("c_prime new shape %s" % (c_prime.get_shape()))
            #c_prime=tf.tile(c_prime,[1,keys_len,1])     # shape (batch_size, context_hidden,2*hidden_size)
            #print("c_prime tiled shape %s" % (c_prime.get_shape()))

            #return_value=tf.concat([keys,a_sub_i,a_sub_i_cross_c_i,c_prime],axis=2)     # shape (batch_size, context_hidden,8*hidden_size)
            return_value = tf.concat([a_sub_i, a_sub_i_cross_c_i, c_prime],axis=2)  # shape (batch_size, context_hidden,6*hidden_size)
            #print("return_value shape %s" % (return_value.get_shape()))
            return_value= tf.nn.dropout(return_value, self.keep_prob)
            return return_value
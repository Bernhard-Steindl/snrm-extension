"""
The SNRM model proposed in:
Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, Jaap Kamps.
"From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing", In CIKM '18.
Authors: Hamed Zamani (zamani@cs.umass.edu)

SNRM model definition using PyTorch
Authors: Hamed Zamani (zamani@cs.umass.edu)
"""
from app_logger import logger
logger = logger(__file__)

from config import config
import params

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder

from collections import OrderedDict
from typing import List

#import numpy as np
import util

class SNRM(nn.Module):
    """
        The implementation of the SNRM model proposed by Zamani et al. (CIKM '18). The model learns a sparse
        representation for query and documents in order to take advantage of inverted indexing at the inference time for
        efficient retrieval. This is the first learning to rank model that does 'ranking' instead of 're-ranking'. For
        more information, please refer to the following paper:

        Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, Jaap Kamps.
        "From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing", In CIKM '18.

    """
    def __init__(self, 
                word_embeddings: TextFieldEmbedder,
                batch_size: int, 
                max_q_len: int, 
                max_doc_len: int,
                emb_dim: int,
                layer_size: List[int], 
                dropout_parameter   , 
                regularization_term, 
                learning_rate):
        """
            The SNRM constructor.
            Args:
            dictionary (obj): an instance of the class Dictionary containing terms and term IDs.
            pre_trained_embedding_file_name (str): the path to the pre-trained word embeddings for initialization.
                 This is optional. If a term in the dictionary does not appear in the pre-trained vector file, its
                 embedding will be initialized by a random vector. If this argument is 'None', the embedding matrix will
                 be initialized randomly with a uniform distribution.
            batch_size (int): the batch size for training and validation.
            max_q_len (int): maximum length of a query.
            max_doc_len (int): maximum length of a document.
            emb_dim (int): embedding dimensionality.
            layer_size (list): a list of int containing the size of each layer.
            dropout_parameter (float): the keep probability of dropout. 1 means no dropout.
            regularization_term (float): the weight of the l1 regularization for sparsity.
            learning_rate (float): the learning rate for the adam optimizer.
        """
        super(SNRM, self).__init__()
        self.word_embeddings = word_embeddings

        self.batch_size = batch_size
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len
        self.emb_dim = emb_dim
        self.layer_size = layer_size
        self.dropout_parameter = dropout_parameter
        self.regularization_term = regularization_term
        self.learning_rate = learning_rate

        self.convolution = nn.Sequential()
        conv_layer_dict = OrderedDict()


        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        for i in range(len(self.layer_size)-1):
            conv_layer_dict['conv_' + str(i)] = nn.Conv2d(in_channels=self.layer_size[i], 
                                                           out_channels=self.layer_size[i+1],
                                                           kernel_size=(1, 5 if i==0 else 1), 
                                                           stride=(1,1),
                                                           padding=(0, 2 if i==0 else 0),
                                                           dilation=(1,1),
                                                           bias=False,
                                                           padding_mode='zeros')
            # Fills the input Tensor with values drawn from the normal distribution
            nn.init.normal_(tensor=conv_layer_dict['conv_' + str(i)].weight, mean=0.0, std=1.0)

            # conv_layer_dict['relu_' + str(i)] = nn.ReLU()
        conv_layer_dict['relu_end'] = nn.ReLU()

        # TODO drop out?
        
        self.convolution = nn.Sequential(conv_layer_dict)

        # TODO ? torch.backends.cudnn.deterministic = True
        # https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d



    def forward(self, query: torch.Tensor, doc_pos: torch.Tensor, doc_neg: torch.Tensor) -> torch.Tensor:

        # TODO why do we need this?
        # shape: (batch, query_max)
        #query_pad_oov_mask = (query["tokens"] > 0).float()
        # shape: (batch, doc_max)
        #document_pad_oov_mask = (document["tokens"] > 0).float()

        # getting the embedding vectors for the query and the documents.
        query_embeddings = self.word_embeddings({"tokens": query})
        doc_pos_embeddings = self.word_embeddings({"tokens": doc_pos})
        doc_neg_embeddings = self.word_embeddings({"tokens": doc_neg})

        # 2020-01-24 16:51:51,770 INFO       snrm.py    get_embedding_layer_output emb shape name=emb_layer_query: [32, 1, 10, 300]
        # 2020-01-24 16:51:51,773 INFO       snrm.py    get_embedding_layer_output emb shape name=emb_layer_doc1: [32, 1, 103, 300]
        # 2020-01-24 16:51:51,775 INFO       snrm.py    get_embedding_layer_output emb shape name=emb_layer_doc2: [32, 1, 103, 300]
        # 2020-01-24 16:51:51,798 INFO       snrm.py    get_embedding_layer_output emb shape name=emb_layer_doc: [None, 1, 103, 300]
        # 2020-01-24 16:51:51,802 INFO       snrm.py    get_embedding_layer_output emb shape name=emb_layer_test_query: [None, 1, 10, 300]
        
        # TODO why do we sometimes get query batch with 9 num_tokens/per query instead of 10 
        # or doc_neg num_tokens of 83 instead of max_doc_length? i.e.
        # shape query_embeddings torch.Size([32, 9, 300])
        # shape doc_neg_embeddings torch.Size([32, 83, 300])
        query_num_token = query_embeddings.shape[1] # is <= self.max_q_len
        doc_pos_num_tokens = doc_pos_embeddings.shape[1] # is <= self.max_doc_len
        doc_neg_num_tokens = doc_neg_embeddings.shape[1] # is <= self.max_doc_len

        #logger.info('shape query_embeddings {}'.format(query_embeddings.size())) # [32, 10, 300])
        #logger.info('shape doc_pos_embeddings {}'.format(doc_pos_embeddings.size())) # [32, 103, 300]
        #logger.info('shape doc_neg_embeddings {}'.format(doc_neg_embeddings.size())) # [32, 103, 300]
        
        # network input should be in shape [N,C_in​,H,W]
        # where N := batch size, C_in := number of input_channels, H := height of input, W := width of input
        query_embeddings_nchw = query_embeddings.view(-1, self.emb_dim, 1, query_num_token) # [32, 300, 1, 10]
        doc_pos_embeddings_nchw = doc_pos_embeddings.view(-1, self.emb_dim, 1, doc_pos_num_tokens) # [32, 300, 1, 103]
        doc_neg_embeddings_nchw = doc_neg_embeddings.view(-1, self.emb_dim, 1, doc_neg_num_tokens) # [32, 300, 1, 103]

        #logger.info('shape query_embeddings_nchw {}'.format(query_embeddings_nchw.size())) # [32, 300, 1, 10]
        #logger.info('shape doc_pos_embeddings_nchw {}'.format(doc_pos_embeddings_nchw.size())) # [32, 300, 1, 103]
        #logger.info('shape doc_neg_embeddings_nchw {}'.format(doc_neg_embeddings_nchw.size())) # [32, 300, 1, 103]

        self.q_repr = self.convolution(query_embeddings_nchw)
        logger.info('q_repr before reduce_mean shape: {}'.format(self.q_repr.size())) # torch.Size([32, 50, 1, 10])
        self.d1_repr = self.convolution(doc_pos_embeddings_nchw)
        logger.info('d1_repr before reduce_mean shape: {}'.format(self.d1_repr.size())) # torch.Size([32, 50, 1, 103])
        self.d2_repr = self.convolution(doc_neg_embeddings_nchw)
        logger.info('d2_repr before reduce_mean shape: {}'.format(self.d2_repr.size())) # torch.Size([32, 50, 1, 103])

        reduction_dim = [2,3] # not [1,2] because of different order of dimensions than in legacy snrm code
        self.q_repr = torch.mean(self.q_repr, reduction_dim)
        logger.info('q_repr after reduce_mean shape: {}'.format(self.q_repr.size())) # torch.Size([32, 50])
        self.d1_repr = torch.mean(self.d1_repr, reduction_dim)
        logger.info('d1_repr after reduce_mean shape: {}'.format(self.d1_repr.size())) # torch.Size([32, 50])
        self.d2_repr = torch.mean(self.d2_repr, reduction_dim)
        logger.info('d2_repr after reduce_mean shape: {}'.format(self.d2_repr.size())) # torch.Size([32, 50])

    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         # The placeholders for input data for each batch.
    #         self.query_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_q_len])
    #         self.doc1_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_doc_len])
    #         self.doc2_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_doc_len])
    #         self.labels_pl = tf.placeholder(tf.float32, shape=[self.batch_size, 2])

    #         self.dropout_keep_prob = tf.constant(self.dropout_parameter)

    #         # For inverted index construction
    #         self.doc_pl = tf.placeholder(tf.int32, shape=[None, self.max_doc_len])
    #         self.test_query_pl = tf.placeholder(tf.int32, shape=[None, self.max_q_len])

    #         # Look up embeddings for inputs. The last element is for padding.
    #         embeddings = tf.concat([
    #             self.get_embedding_params(self.dictionary, self.emb_dim, self.pre_trained_embedding_file_name),
    #             tf.constant(np.zeros([1, self.emb_dim]), dtype=tf.float32)], 0)

    #         logger.info('obtained embeddings')

    #         # getting the embedding vectors for the query and the documents.
    #         emb_layer_q = self.get_embedding_layer_output(embeddings, self.emb_dim, 'emb_layer_query',
    #                                                             self.query_pl, self.max_q_len)
    #         emb_layer_d1 = self.get_embedding_layer_output(embeddings, self.emb_dim, 'emb_layer_doc1',
    #                                                             self.doc1_pl, self.max_doc_len)
    #         emb_layer_d2 = self.get_embedding_layer_output(embeddings, self.emb_dim, 'emb_layer_doc2',
    #                                                             self.doc2_pl, self.max_doc_len)

    #         self.weights, self.weights_name, self.biases, self.biases_name = self.get_network_params(self.layer_size)

    #         self.q_repr = self.network(emb_layer_q, self.weights, self.weights_name, self.biases, self.biases_name)
    #         self.d1_repr = self.network(emb_layer_d1, self.weights, self.weights_name, self.biases, self.biases_name)
    #         self.d2_repr = self.network(emb_layer_d2, self.weights, self.weights_name, self.biases, self.biases_name)

    #         logits_d1 = tf.reduce_mean(tf.multiply(self.q_repr, self.d1_repr), axis=1, keep_dims=True)
    #         logits_d2 = tf.reduce_mean(tf.multiply(self.q_repr, self.d2_repr), axis=1, keep_dims=True)
    #         logits = tf.concat([logits_d1, logits_d2], axis=1)

    #         # For inverted index construction:
    #         embedding_layer_doc = self.get_embedding_layer_output(
    #             embeddings, self.emb_dim, 'emb_layer_doc', self.doc_pl, self.max_doc_len)
    #         self.doc_representation = self.network(
    #             embedding_layer_doc, self.weights, self.weights_name, self.biases, self.biases_name)

    #         # For retrieval:
    #         embedding_layer_test_query = self.get_embedding_layer_output(
    #             embeddings, self.emb_dim, 'emb_layer_test_query', self.test_query_pl, self.max_q_len)
    #         self.query_representation = self.network(
    #             embedding_layer_test_query, self.weights, self.weights_name, self.biases, self.biases_name)

    #         # the hinge loss function for training
    #         # hinge_loss(labels, logits, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
    #         self.loss = tf.reduce_mean(
    #             tf.losses.hinge_loss(logits=logits, labels=self.labels_pl, scope='hinge_loss'))

    #         # the l1 regularization for sparsity. Since we use ReLU as the activation function, all the outputs of the
    #         # network are non-negative and thus we do not need to get the absolute value for computing the l1 loss.
    #         self.l1_regularization = tf.reduce_mean(
    #             tf.reduce_sum(tf.concat([self.q_repr, self.d1_repr, self.d2_repr], axis=1), axis=1),
    #             name='l1_regularization')
    #         # the cost function including the hinge loss and the l1 regularization.
    #         self.cost = self.loss + (tf.constant(self.regularization_term, dtype=tf.float32) * self.l1_regularization)

    #         # computing the l0 losses for visualization purposes.
    #         l0_regularization_docs = tf.cast(tf.count_nonzero(tf.concat([self.d1_repr, self.d2_repr], axis=1)), tf.float32) \
    #                                  / tf.constant(2 * self.batch_size, dtype=tf.float32)

    #         l0_regularization_query = tf.cast(tf.count_nonzero(self.q_repr), tf.float32) \
    #                                   / tf.constant(self.batch_size, dtype=tf.float32)

    #         # the Adam optimizer for training.
    #         self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    #         # Some plots for visualization
    #         tf.summary.scalar('loss', self.loss)
    #         tf.summary.scalar('cost', self.cost)
    #         tf.summary.scalar('l1', self.l1_regularization)
    #         tf.summary.scalar('l0-docs', l0_regularization_docs)
    #         tf.summary.scalar('l0-query', l0_regularization_query)

    #         tf.summary.scalar('mean-repr_q', tf.reduce_mean(self.q_repr))
    #         tf.summary.scalar('mean-repr_d1', tf.reduce_mean(self.d1_repr))
    #         tf.summary.scalar('mean-repr_d2', tf.reduce_mean(self.d2_repr))

    #         tf.summary.scalar('mean-input_q', tf.reduce_mean(self.query_pl))
    #         tf.summary.scalar('mean-input_d1', tf.reduce_mean(self.doc1_pl))
    #         tf.summary.scalar('mean-input_d2', tf.reduce_mean(self.doc2_pl))

    #         tf.summary.scalar('mean-logits_d1', tf.reduce_mean(logits_d1))
    #         tf.summary.scalar('mean-logits_d2', tf.reduce_mean(logits_d2))

    #         tf.summary.scalar('ratio_num_zero_elements_q_repr_batch', 1 - (tf.cast(tf.count_nonzero(self.q_repr), tf.float32) / tf.cast(np.prod(self.q_repr.shape), tf.float32)))
    #         tf.summary.scalar('ratio_num_zero_elements_d1_repr_batch', 1 - (tf.cast(tf.count_nonzero(self.d1_repr), tf.float32) / tf.cast(np.prod(self.d1_repr.shape), tf.float32)))
    #         tf.summary.scalar('ratio_num_zero_elements_d2_repr_batch', 1 - (tf.cast(tf.count_nonzero(self.d2_repr), tf.float32) / tf.cast(np.prod(self.d2_repr.shape), tf.float32)))
            
    #         tf.summary.scalar('ratio_num_non_zero_elements_q_repr_batch', (tf.cast(tf.count_nonzero(self.q_repr), tf.float32) / tf.cast(np.prod(self.q_repr.shape), tf.float32)))
    #         tf.summary.scalar('ratio_num_non_zero_elements_d1_repr_batch', (tf.cast(tf.count_nonzero(self.d1_repr), tf.float32) / tf.cast(np.prod(self.d1_repr.shape), tf.float32)))
    #         tf.summary.scalar('ratio_num_non_zero_elements_d2_repr_batch', (tf.cast(tf.count_nonzero(self.d2_repr), tf.float32) / tf.cast(np.prod(self.d2_repr.shape), tf.float32)))
        
    #         self.summary_op = tf.summary.merge_all()

    #         # Add variable initializer.
    #         self.init = tf.global_variables_initializer()

    #         # For storing a trained model,
    #         self.saver = tf.train.Saver()

    # def get_network_params(self, layer_sizes):
    #     """
    #         Returning the parameters of the network.
    #         Args:
    #             layer_sizes (list): a list containing the output size of each layer.

    #         Returns:
    #             weights (dict): a mapping from layer name to TensorFlow Variable corresponding to the layer weights.
    #             weights_name (list): a list of str containing layer names for weight parameters.
    #             biases (dict): a mapping from layer name to TensorFlow Variable corresponding to the layer biases.
    #             biases_name (list): a list of str containing layer names for bias parameters.
    #     """
    #     weights = {}
    #     weights_name = ['w' + str(i) for i in range(1, len(layer_sizes) - 1)] + ['w_out']

    #     biases = {}
    #     biases_name = ['b' + str(i) for i in range(1, len(layer_sizes) - 1)] + ['b_out']

    #     for i in range(len(layer_sizes) - 1):
    #         with tf.name_scope(weights_name[i]):
    #             weights[weights_name[i]] = \
    #                 tf.Variable(tf.random_normal([1, 5 if i==0 else 1, layer_sizes[i], layer_sizes[i + 1]],
    #                                              name=weights_name[i]))
    #                 # Outputs random values from a normal distribution.
    #                 # tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)

    #     return weights, weights_name, biases, biases_name

    # def get_embedding_params(self, dictionary, dim, pre_trained_embedding_file_name=None):
    #     """
    #         Returning the parameters of the network.
    #         Args:
    #             dictionary (obj): an instance of the class Dictionary containing terms and term IDs.
    #             dim (int): embedding dimensionality.
    #             pre_trained_embedding_file_name (str): the path to the pre-trained word embeddings for initialization.
    #              This is optional. If a term in the dictionary does not appear in the pre-trained vector file, its
    #              embedding will be initialized by a random vector. If this argument is 'None', the embedding matrix will
    #              be initialized randomly with a uniform distribution.

    #         Returns:
    #             embedding_matrix (obj): a 2D TensorFlow Varibale containing the embedding vector for each term ID. For
    #              unknown terms, the term_id is zero.
    #      """
    #     if pre_trained_embedding_file_name is None:
    #         return tf.Variable(tf.random_uniform([dictionary.size(), dim], -1.0, 1.0))
    #     else:
    #         term_to_id, id_to_term, we_matrix = util.load_word_embeddings(pre_trained_embedding_file_name, dim, True, dictionary.term_to_id)
    #         init_matrix = np.random.random((dictionary.size(), dim))

    #         unknown_terms_in_embedding = 0

    #         for i in range(dictionary.size()):
    #             if dictionary.id_to_term[i] in term_to_id:
    #                 tid = term_to_id[dictionary.id_to_term[i]]
    #                 init_matrix[i] = we_matrix[tid]
    #             else:
    #                 unknown_terms_in_embedding += 1

    #         logger.debug('found {} unkonwn terms from dictionary that are missing in embedding \n'
    #             .format(str(unknown_terms_in_embedding)))
    #         # The tf.constant_initializer() function might not accept a tf.Tensor as an argument,
    #         # but tf.get_variable() does accept a tf.Tensor as its initializer argument. This means you can write:
    #         return tf.get_variable('embeddings', shape=[dictionary.size(), dim],
    #                                trainable=False,
    #                                initializer=tf.constant_initializer(init_matrix))

    # def get_embedding_layer_output(self, embeddings, dim, layer_name, input, n_terms):
    #     """
    #         Getting the output of embedding layer for a batch.
    #         Args:
    #             embeddings (obj): a TensorFlow Variable (or Tensor) containing the word embedding vectors.
    #             dim (int): Embedding dimensionality.
    #             layer_name (str): a scope name for the embedding layer.
    #             input (obj): a 2D Tensor (or Placeholder) containing the term ids with the size of batch_size * n_terms.
    #             n_terms (int): number of terms per instance (text).

    #         Returns: a 2D Tensor containing the output of the embedding layer for a batch for text.
    #     """
    #     with tf.name_scope('embedding_layer'):
    #         with tf.name_scope(layer_name):
    #             emb = tf.nn.embedding_lookup(embeddings, tf.reshape(input, [-1]))
    #             emb = tf.reshape(emb, [-1, 1, n_terms, dim])
    #     logger.info('emb shape name={}: {}'.format(layer_name, repr(emb.get_shape().as_list())))
    #     return emb

    # def network(self, input_layer, weights, weights_name, biases, biases_name):
    #     """
    #         Neural network architecture: a convolutional network with ReLU activations for hidden layers and dropout for
    #         regularization.

    #         Args:
    #             input_layer (obj): a Tensor representing the output of embedding layer which is the input of the neural
    #              ranking models.
    #             weights (dict): a mapping from layer name to TensorFlow Variable corresponding to the layer weights.
    #             weights_name (list): a list of str containing layer names for weight parameters.
    #             biases (dict): a mapping from layer name to TensorFlow Variable corresponding to layer biases.
    #             biases_name (list): a list of str containing layer names for bias parameters.

    #         Returns: a Tensor containing the logits for the inputs.
    #     """

    #     layers = [input_layer]
    #     for i in range(len(weights)):
    #         with tf.name_scope('layer_' + str(i + 1)):
    #             # we did not use the biases.

    #             # Given an input tensor of shape [batch, in_height, in_width, in_channels] and 
    #             # a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #             
    # # i.e. 
    # # input shape  = N,H,W,C_in     
    # # kernel shape = H,W,C_in,C_out
    #               
    # # default tf conv2d = NHWC 
    # data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC"
    # https://github.com/Bernhard-St/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/nn/conv2d.md
    #
    # conv2d(
    #   input,
    #   filter,
    #   strides,
    #   padding,
    #   use_cudnn_on_gpu=True,
    #   data_format='NHWC',
    #   name=None
    # )
    #
    # TODO which dilation are we using? assumption (1,1) although not available in tf.nn.conv2d signature
    #
    #             layers.append(tf.nn.conv2d(input=layers[i],
    #                                        filter=weights[weights_name[i]],
    #                                        strides=[1, 1, 1, 1],
    #                                        padding='SAME'))
    #             # 'SAME' padding specifies that the output size should be the same as the input size
    #             # In order to achieve this, there is a 1 pixel? width padding around the image, 
    #             # and the filter slides outside the image into this padding area

    #             if i == len(weights)-1:
    #                 layers[i + 1] = tf.nn.relu(layers[i + 1])

    #     return tf.reduce_mean(layers[len(layers) - 1], [1, 2])



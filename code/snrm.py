"""
The SNRM model proposed in:
Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, Jaap Kamps.
"From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing", In CIKM '18.
Authors: Hamed Zamani (zamani@cs.umass.edu)

SNRM model definition using PyTorch
Authors: Hamed Zamani (zamani@cs.umass.edu), Bernhard Steindl
"""
from app_logger import logger
logger = logger(__file__)

from config import config
import params

import torch
import torch.nn as nn

from allennlp.modules.text_field_embedders import TextFieldEmbedder

from collections import OrderedDict
from typing import List, Optional

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
                dropout_parameter: float, 
                regularization_term: float, 
                learning_rate: float):
        """
            The SNRM constructor.
            Args:
            dictionary (obj): an instance of the class Dictionary containing terms and term IDs.
            word_embeddings (TextFieldEmbedder): instance of word embedding
            batch_size (int): the batch size for training and validation.
            max_q_len (int): maximum length of a query.
            max_doc_len (int): maximum length of a document.
            emb_dim (int): embedding dimensionality.
            layer_size (list): a list of int containing the size of each layer.
            dropout_parameter (float): the probability of dropout. 0 means no dropout.
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

        for i in range(len(self.layer_size)-1):
            conv_layer_dict['conv_' + str(i)] = nn.Conv2d(in_channels=self.layer_size[i], 
                                                           out_channels=self.layer_size[i+1],
                                                           kernel_size=(1, 5 if i==0 else 1), 
                                                           stride=(1,1),
                                                           padding=(0, 2 if i==0 else 0),
                                                           dilation=(1,1),
                                                           bias=True,
                                                           padding_mode='zeros')
            # Fills the input Tensor with values drawn from the normal distribution
            # nn.init.normal_(tensor=conv_layer_dict['conv_' + str(i)].weight, mean=0.0, std=1.0)

            # Either use ReLu after every convolution layer or only on the last layer:
            # conv_layer_dict['relu_' + str(i)] = nn.ReLU()

        conv_layer_dict['relu_end'] = nn.ReLU()

        # Optionally add drop out
        # During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. 
        # Each channel will be zeroed out independently on every forward call.
        # torch.nn.Dropout(p=0.5, inplace=False) 
        # p = probability of an element to be zeroed
        # nn.Dropout(p=config.get('dropout_probability'), inplace=False) 
        
        self.convolution = nn.Sequential(conv_layer_dict)

    def forward(self, query: Optional[torch.Tensor], doc_pos: Optional[torch.Tensor], doc_neg: Optional[torch.Tensor]) -> (Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]):

        reduction_dim = [2,3] # not [1,2] because of different order of dimensions than in legacy snrm code

        (q_repr, d1_repr, d2_repr) = (None, None, None) # init - we may not return a Tensor for every input arg

        if query != None:
            mask_query_oov = (query > 1).float() # padding idx=0, oov idx=1; mask is 0 if padding or oov token, otherwise 1

            # getting the embedding vectors for the query and the documents.
            # shapes [32, 10, 300] * [32, 10, 1] (use unsqueeze to add a dimension on the specified index)
            query_embeddings = self.word_embeddings({"tokens": query}) * mask_query_oov.unsqueeze(2) # mask used for padding oov tokens with 0 value
            #logger.info('shape query_embeddings {}'.format(query_embeddings.size())) # [32, 10, 300])

            # ADD: AIR assignment hint: The batch tensors also have no fixed size, the max values in the readers are just to cap outliers
            # (the size will be based on the biggest sample in the batch (per tensor) and the others padded with 0 to the same length)
            query_num_token = query_embeddings.shape[1] # is <= self.max_q_len

            # network input should be in shape [N,C_in​,H,W]
            # where N := batch size, C_in := number of input_channels, H := height of input, W := width of input
            query_embeddings_nchw = query_embeddings.view(-1, self.emb_dim, 1, query_num_token) # [32, 300, 1, 10]
            #logger.info('shape query_embeddings_nchw {}'.format(query_embeddings_nchw.size())) # [32, 300, 1, 10]

            q_repr = self.convolution(query_embeddings_nchw)
            #logger.info('q_repr before reduce_mean shape: {}'.format(q_repr.size())) # torch.Size([32, 50, 1, 10])

            q_repr = torch.mean(q_repr, reduction_dim) # torch.Size([32, 50])
            #logger.info('q_repr after reduce_mean shape: {}'.format(q_repr.size())) # torch.Size([32, 50])

        if doc_pos != None:
            mask_doc_pos_oov = (doc_pos > 1).float() # padding idx=0, oov idx=1; mask is 0 if padding or oov token, otherwise 1
            # shapes [32, 103, 300] * [32, 103, 1] (use unsqueeze to add a dimension on the specified index)
            doc_pos_embeddings = self.word_embeddings({"tokens": doc_pos}) * mask_doc_pos_oov.unsqueeze(2) # mask used for padding oov tokens with 0 value
            #logger.info('shape doc_pos_embeddings {}'.format(doc_pos_embeddings.size())) # [32, 103, 300]

            # ADD: AIR assignment hint: The batch tensors also have no fixed size, the max values in the readers are just to cap outliers
            # (the size will be based on the biggest sample in the batch (per tensor) and the others padded with 0 to the same length)
            doc_pos_num_tokens = doc_pos_embeddings.shape[1] # is <= self.max_doc_len

            # network input should be in shape [N,C_in​,H,W]
            # where N := batch size, C_in := number of input_channels, H := height of input, W := width of input
            doc_pos_embeddings_nchw = doc_pos_embeddings.view(-1, self.emb_dim, 1, doc_pos_num_tokens) # [32, 300, 1, 103]
            #logger.info('shape doc_pos_embeddings_nchw {}'.format(doc_pos_embeddings_nchw.size())) # [32, 300, 1, 103]
            
            d1_repr = self.convolution(doc_pos_embeddings_nchw)
            #logger.info('d1_repr before reduce_mean shape: {}'.format(d1_repr.size())) # torch.Size([32, 50, 1, 103])

            d1_repr = torch.mean(d1_repr, reduction_dim) # torch.Size([32, 50])
            #logger.info('d1_repr after reduce_mean shape: {}'.format(d1_repr.size())) # torch.Size([32, 50])

        if doc_neg != None:
            mask_doc_neg_oov = (doc_neg > 1).float() # padding idx=0, oov idx=1; mask is 0 if padding or oov token, otherwise 1
            # shapes [32, 103, 300] * [32, 103, 1] (use unsqueeze to add a dimension on the specified index)
            doc_neg_embeddings = self.word_embeddings({"tokens": doc_neg}) * mask_doc_neg_oov.unsqueeze(2) # mask used for padding oov tokens with 0 value
            #logger.info('shape doc_neg_embeddings {}'.format(doc_neg_embeddings.size())) # [32, 103, 300]

            # ADD: AIR assignment hint: The batch tensors also have no fixed size, the max values in the readers are just to cap outliers
            # (the size will be based on the biggest sample in the batch (per tensor) and the others padded with 0 to the same length)
            doc_neg_num_tokens = doc_neg_embeddings.shape[1] # is <= self.max_doc_len

            # network input should be in shape [N,C_in​,H,W]
            # where N := batch size, C_in := number of input_channels, H := height of input, W := width of input
            doc_neg_embeddings_nchw = doc_neg_embeddings.view(-1, self.emb_dim, 1, doc_neg_num_tokens) # [32, 300, 1, 103]
            #logger.info('shape doc_neg_embeddings_nchw {}'.format(doc_neg_embeddings_nchw.size())) # [32, 300, 1, 103]
        
            d2_repr = self.convolution(doc_neg_embeddings_nchw)
            #logger.info('d2_repr before reduce_mean shape: {}'.format(d2_repr.size())) # torch.Size([32, 50, 1, 103])
        
            d2_repr = torch.mean(d2_repr, reduction_dim) # torch.Size([32, 50])
            #logger.info('d2_repr after reduce_mean shape: {}'.format(d2_repr.size())) # torch.Size([32, 50])
        
        return (q_repr, d1_repr, d2_repr)
"""
Training the SNRM model.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

from app_logger import logger
logger = logger(__file__)

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({})) # sets the seeds to be fixed

from config import config
import params

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import WordTokenizer

from data_loading import IrTripleDatasetReader

import os

from snrm import SNRM


vocabulary: 'Vocabulary' = Vocabulary.from_files(directory=config.get('vocab_dir_path'))

token_embedding: 'Embedding' = Embedding.from_params(vocab=vocabulary, params=Params({"pretrained_file": config.get('pre_trained_embedding_file_name'),
                                                                              "embedding_dim": config.get('emb_dim'),
                                                                              "trainable": config.get('embedding_trainable') == 'True',
                                                                              "max_norm": None, 
                                                                              "norm_type": 2.0,
                                                                              "padding_index": 0}))

word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

triple_loader = IrTripleDatasetReader(lazy=True, 
                                       max_doc_length=config.get('max_doc_len'),
                                       max_query_length=config.get('max_q_len'),
                                       tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())) 
                                       # already spacy tokenized, so that it is faster 

iterator = BucketIterator(batch_size=config.get('batch_size'),
                          sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

iterator.index_with(vocabulary)


# layer_size is a list containing the size of each layer. It can be set through the 'hidden_x' arguments.
layer_size = [config.get('emb_dim')] # input layer
for i in [config.get('hidden_1'), config.get('hidden_2'), config.get('hidden_3'), config.get('hidden_4'), config.get('hidden_5')]:
    if i <= 0:
        break
    layer_size.append(i)


# The SNRM model.
model = SNRM(word_embeddings= word_embedder,
            batch_size=config.get('batch_size'),
            max_q_len=config.get('max_q_len'),
            max_doc_len=config.get('max_doc_len'),
            emb_dim=config.get('emb_dim'),
            layer_size=layer_size,
            dropout_parameter=config.get('dropout_parameter'),
            regularization_term=config.get('regularization_term'),
            learning_rate=config.get('learning_rate'))

logger.info('Model "{}" parameters: {}'.format(config.get('run_name'), sum(p.numel() for p in model.parameters() if p.requires_grad)))
logger.info('Network: {}'.format(repr(model)))

for n, p in model.named_parameters():
    logger.info('model parameter "{}" shape {}'.format(n, p.shape))

# TODO loss, optimizer

# torch.nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')

#regularization_term = config.get('regularization_term')
# https://pytorch.org/docs/stable/nn.html#l1loss
#l1_regularization = 1
#l1_regularization = torch.mul(l1_regularization, regularization_term)
# https://pytorch.org/docs/stable/nn.html#hingeembeddingloss
#hinge_loss = F.hinge_embedding_loss(input=1, 
                                    #target=1,
                                    #reduction='mean')
# F.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor
# F.l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
#criterion = torch.sum(hinge_loss, l1_regularization)

# optimizer 
# TODO remove (create after model.cuda - according to its docs)
model_params = model.parameters()
optimizer = Adam(model_params, lr=config.get('learning_rate'))

# training
num_steps = config.get('num_train_steps')
average_loss = 0

# TODO how stopword removal?
for step in range(num_steps):
    if step % 100 == 0:
        logger.info('training step {} of {}'.format(step+1, num_steps))

    for batch in Tqdm.tqdm(iterator(triple_loader.read(config.get('training_data_triples_file')), num_epochs=1)):
        q_repr, doc_pos_repr, doc_neg_repr = model.forward(batch["query_tokens"]["tokens"], 
                                                           batch["doc_pos_tokens"]["tokens"], 
                                                           batch["doc_neg_tokens"]["tokens"])
        # q_repr and doc_pos_repr and doc_neg_repr have shape [batch_size, nn_output_dim]

        # torch.mean: If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
        logits_d1 = torch.mean(input=torch.mul(q_repr, doc_pos_repr), dim=1, keepdim=True) # shape: torch.Size([batch_size, 1])
        logits_d2 = torch.mean(input=torch.mul(q_repr, doc_neg_repr), dim=1, keepdim=True) # shape: torch.Size([batch_size, 1])
        logits = torch.cat(tensors=(logits_d1, logits_d2), dim=1) # shape: torch.Size([batch_size, 2])

        # Assumption doc1 is relevant/positive (1), but doc2 is non-relevant/negative (0) for every doc-pair in batch
        # instead of using 0, PyTorch wants us to use -1 instead of 0
        target_relevance_labels = torch.tensor([[1, -1]]).repeat(config.get('batch_size'), 1) # [1,-1]*batch_size, shape: torch.Size([batch_size, 2])
        
#       self.labels_pl = tf.placeholder(tf.float32, shape=[self.batch_size, 2])
#       labels = np.array(labels) # shape [batch_size]
#       labels = np.concatenate(
#                [labels.reshape(FLAGS.batch_size, 1), 1. - labels.reshape(FLAGS.batch_size, 1)], axis=1)
#         # the hinge loss function for training
#         # hinge_loss(labels, logits, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
#         self.loss = tf.reduce_mean(
#             tf.losses.hinge_loss(logits=logits, labels=self.labels_pl, scope='hinge_loss'))
        
        # F.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor
        # https://pytorch.org/docs/stable/nn.html#hingeembeddingloss
        hinge_loss = F.hinge_embedding_loss(input=logits, 
                                            target=target_relevance_labels, 
                                            margin=1.0, 
                                            reduction='mean') # torch.Size([]) i.e. scalar tensor
        # logger.info('hinge_loss shape {} data {}'.format(hinge_loss.size(), hinge_loss.data))

#         self.l1_regularization = tf.reduce_mean(
#             tf.reduce_sum(tf.concat([self.q_repr, self.d1_repr, self.d2_repr], axis=1), axis=1),
#             name='l1_regularization')
#         # the cost function including the hinge loss and the l1 regularization.
#         self.cost = self.loss + (tf.constant(self.regularization_term, dtype=tf.float32) * self.l1_regularization)

        
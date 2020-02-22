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
import json


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


# optimizer 
# TODO remove (create after model.cuda - according to its docs)
model_params = model.parameters()
optimizer = Adam(model_params, lr=config.get('learning_rate'))

# training
num_training_steps = config.get('num_train_steps')
average_loss = 0
regularization_term = config.get('regularization_term')

# TODO how stopword removal?
# TODO steps are done in inner loop - outer one could be epochs, but we do not have this in legacy code
curr_training_step = 0
    

for batch in Tqdm.tqdm(iterator(triple_loader.read(config.get('training_data_triples_file')), num_epochs=1)):
    curr_training_step += 1
    optimizer.zero_grad()  # zero the gradient buffers

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
    
    # TODO experiment by using torch.mean(tensor, dim=1)  instead of torch.sum(tensor, dim=1)
    l1_regularization = torch.mean(torch.sum(input=torch.cat(tensors=(q_repr, doc_pos_repr, doc_neg_repr), dim=1), dim=1)) # torch.Size([]) i.e. scalar tensor
    # logger.info('l1_regularization shape {} data {}'.format(l1_regularization.size(), l1_regularization.data))
    cost = torch.add(hinge_loss, torch.mul(l1_regularization, regularization_term)) # torch.Size([]) i.e. scalar tensor
    # logger.info('cost {}'.format(cost.data))

    if curr_training_step % 100 == 0:
        log_output_train = (curr_training_step, num_training_steps,cost.data, hinge_loss.data, l1_regularization.data)
        logger.info('Training step {:6d} of {:6d} \t Cost={:10.4f} (Hinge-Loss={:10.4f}, L1-Reg={:10.4f})'.format(*log_output_train))

    loss = cost
    loss.backward() # calculate gradients of weights for network
    optimizer.step()  # updates network with new weights




    # https://pytorch.org/docs/stable/notes/serialization.html#best-practices
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # save model to file (saves only the model parameters)
    if curr_training_step == num_training_steps:
        model_save_path = config.get('model_path') + 'model-state_' + config.get('run_name') + '.pt'
        logger.info('Reached {:6d} training steps, now saving model'.format(curr_training_step))
        torch.save(model.state_dict(), model_save_path)
        logger.info('Model has been saved to "{}"'.format(model_save_path))
        model_config_save_path = config.get('model_path') + 'model-config_' + config.get('run_name') + '.json'
        with open(model_config_save_path, 'w') as fp:
            json.dump(dict(config.items()), fp, indent=4)
        logger.info('Model Config has been saved to "{}"'.format(model_config_save_path))
        break
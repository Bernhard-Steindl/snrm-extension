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
        output_pos = model.forward(batch["query_tokens"]["tokens"], batch["doc_pos_tokens"]["tokens"], batch["doc_neg_tokens"]["tokens"])


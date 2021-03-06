"""
Training the SNRM model.

Authors: Hamed Zamani (zamani@cs.umass.edu), Bernhard Steindl
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
from torch.utils.tensorboard import SummaryWriter

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import WordTokenizer

from allennlp.nn.util import move_to_device

from data_loading import IrTripleDatasetReader
from allennlp.data.tokenizers.word_filter import StopwordFilter

import os

from snrm import SNRM
import json
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info('PyTorch uses device {}'.format(device))

def obtain_loss(q_repr: torch.Tensor, doc_pos_repr: torch.Tensor, doc_neg_repr: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    # q_repr and doc_pos_repr and doc_neg_repr have shape [batch_size, nn_output_dim]

     # torch.mean: If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
    logits_d1 = torch.mean(input=torch.mul(q_repr, doc_pos_repr), dim=1, keepdim=True) # shape: torch.Size([batch_size, 1])
    logits_d2 = torch.mean(input=torch.mul(q_repr, doc_neg_repr), dim=1, keepdim=True) # shape: torch.Size([batch_size, 1])
    logits = torch.cat(tensors=(logits_d1, logits_d2), dim=1) # shape: torch.Size([batch_size, 2])

    # AIR Assignment hint: The iterators do not guarantee a fixed batch size (the last one will probably be smaller)
    current_batch_size = batch['query_tokens']['tokens'].shape[0] # is <= config.get('batch_size')
    # Assumption doc1 is relevant/positive (1), but doc2 is non-relevant/negative (0) for every doc-pair in batch
    # instead of using 0, PyTorch wants us to use -1 instead of 0
    target_relevance_labels = torch.tensor([[1, -1]], device=device).repeat(current_batch_size, 1) # [1,-1]*batch_size, shape: torch.Size([batch_size, 2])
       
    # F.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor
    # https://pytorch.org/docs/stable/nn.html#hingeembeddingloss
    hinge_loss = F.hinge_embedding_loss(input=logits, 
                                        target=target_relevance_labels, 
                                        margin=1.0, 
                                        reduction='mean') # torch.Size([]) i.e. scalar tensor
    # logger.info('hinge_loss shape {} data {}'.format(hinge_loss.size(), hinge_loss.data))

    doc_reg_term = config.get('doc_reg_term')
    q_reg_term = config.get('q_reg_term')

    # l1-regularization
    l1_regularization_docs = (doc_pos_repr + doc_neg_repr) * doc_reg_term # shape [batch_size, nn_output_dim]
    l1_regularization_query = q_repr * q_reg_term # shape [batch_size, nn_output_dim]
    l1_regularization = torch.mean(torch.sum(input=(l1_regularization_docs + l1_regularization_query), dim=1)) # torch.Size([]) i.e. scalar tensor

    cost = torch.add(hinge_loss, l1_regularization) # torch.Size([]) i.e. scalar tensor
    # logger.info('cost {}'.format(cost.data))
    return cost, hinge_loss, l1_regularization

def count_zero(tensor: torch.Tensor) -> int:
    return (tensor == 0.0).sum().item()

def write_to_tensorboard(writer: SummaryWriter, step: int, cost: torch.Tensor, hinge_loss: torch.Tensor, l1_regularization: torch.Tensor, q_repr: torch.Tensor, doc_pos_repr: torch.Tensor, doc_neg_repr: torch.Tensor) -> None:
    # https://pytorch.org/docs/stable/tensorboard.html
    # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # using slash (/) in the tag name groups items
    writer.add_scalar('Loss/Cost', cost.item(), step)
    writer.add_scalar('Loss/Hinge_Loss', hinge_loss.item(), step)
    writer.add_scalar('Loss/L1_Regularization', l1_regularization.item(), step)

    writer.add_scalar('Mean_Representation_Batch/Query', torch.mean(q_repr), step)
    writer.add_scalar('Mean_Representation_Batch/Document_Positive', torch.mean(doc_pos_repr), step)
    writer.add_scalar('Mean_Representation_Batch/Document_Negative', torch.mean(doc_neg_repr), step)

    writer.add_scalar('Sparsity_Representation_Batch/Query', count_zero(q_repr) / q_repr.numel(), step)
    writer.add_scalar('Sparsity_Representation_Batch/Document_Positive', count_zero(doc_pos_repr) / doc_pos_repr.numel(), step)
    writer.add_scalar('Sparsity_Representation_Batch/Document_Negative', count_zero(doc_neg_repr) / doc_neg_repr.numel(), step)

def write_histograms_to_tensorboard(writer: SummaryWriter, step: int, model: nn.Module, q_repr: torch.Tensor, doc_pos_repr: torch.Tensor, doc_neg_repr: torch.Tensor) -> None:
    non_zero_indices_query = q_repr[0].nonzero().flatten() # indices of values > 0
    non_zero_indices_doc_pos = doc_pos_repr[0].nonzero().flatten()
    non_zero_indices_doc_neg = doc_neg_repr[0].nonzero().flatten()
    if non_zero_indices_query.nelement() > 0:
        writer.add_histogram('Sparsity_Representation_First_Batch_Item/Query', non_zero_indices_query, step)
    if non_zero_indices_doc_pos.nelement() > 0:
        writer.add_histogram('Sparsity_Representation_First_Batch_Item/Document_Positive', non_zero_indices_doc_pos, step)
    if non_zero_indices_doc_neg.nelement() > 0:
        writer.add_histogram('Sparsity_Representation_First_Batch_Item/Document_Negative', non_zero_indices_doc_neg, step)

    # tensorboard add histogram of value distribution of network output/representation (first item in batch)
    writer.add_histogram('Representation_First_Batch_Item/Query', q_repr[0], step)
    writer.add_histogram('Representation_First_Batch_Item/Document_Positive', doc_pos_repr[0], step)
    writer.add_histogram('Representation_First_Batch_Item/Document_Negative', doc_neg_repr[0], step)

    # tensorboard add histogram for model parameter weight/bias properties
    for n, p in model.named_parameters():
        if p.requires_grad == False:
            continue
        writer.add_histogram('Model_Param_Weight/{}'.format(n), p.data, step)

def validate_model(model: nn.Module, num_validation_steps: int, vocabulary: Vocabulary, writer_valid: SummaryWriter, last_training_step: int) -> float:
    validation_loss = 0.
    model.eval() # Sets the module in evaluation mode. This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm, etc.
    validation_triple_loader = IrTripleDatasetReader(lazy=True, 
                                                     max_doc_length=config.get('max_doc_len'),
                                                     max_query_length=config.get('max_q_len'),
                                                     tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(),
                                                                               word_filter=StopwordFilter()))
                                                     # already spacy tokenized, so that it is faster 
    iterator = BucketIterator(batch_size=config.get('batch_size'),
                            sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])
    iterator.index_with(vocabulary)
    curr_validation_step = 0
    for batch in Tqdm.tqdm(iterator(validation_triple_loader.read(config.get('validation_data_triples_file')), num_epochs=1)):
        curr_validation_step += 1
        batch = move_to_device(obj=batch, cuda_device=(0 if torch.cuda.is_available() else -1))
        q_repr, doc_pos_repr, doc_neg_repr = model.forward(batch["query_tokens"]["tokens"], 
                                                           batch["doc_pos_tokens"]["tokens"], 
                                                           batch["doc_neg_tokens"]["tokens"])
        # q_repr and doc_pos_repr and doc_neg_repr have shape [batch_size, nn_output_dim]
        cost, hinge_loss, l1_regularization = obtain_loss(q_repr, doc_pos_repr, doc_neg_repr)
        if curr_validation_step % 10 == 0:
            log_output_train = (curr_validation_step, num_validation_steps, cost.data, hinge_loss.data, l1_regularization.data)
            logger.info('Validation step {:6d} of {:6d} \t Cost={:10.4f} (Hinge-Loss={:10.4f}, L1-Reg={:8.5f})'.format(*log_output_train))
        if (last_training_step < 1000 and curr_validation_step % 5 == 0) or (last_training_step >= 1000 and curr_validation_step % 10 == 0):
            write_to_tensorboard(writer_valid, last_training_step + curr_validation_step, cost, hinge_loss, l1_regularization, q_repr, doc_pos_repr, doc_neg_repr)
        validation_loss += cost
        if curr_validation_step == num_validation_steps:
            break
    return validation_loss / num_validation_steps # average validation_loss

def save_model(model: nn.Module, step_num='final') -> None:
    # https://pytorch.org/docs/stable/notes/serialization.html#best-practices
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # save model to file (saves only the model parameters)
    model_save_path = '{0}model-state_{1}_{2}.pt'.format(config.get('model_path'), config.get('run_name'), step_num)
    logger.info('Reached {:6d} training steps, now saving model'.format(curr_training_step))
    torch.save(model.state_dict(), model_save_path)
    logger.info('Model has been saved to "{}"'.format(model_save_path))
    model_config_save_path = '{0}model-config_{1}_{2}.json'.format(config.get('model_path'), config.get('run_name'), step_num)
    with open(model_config_save_path, 'w') as fp:
        json.dump(dict(config.items()), fp, indent=4)
    logger.info('Model Config has been saved to "{}"'.format(model_config_save_path))

######################################################################
######################################################################


should_validate_every_n_steps = config.get('validate_every_n_steps') > -1

# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
tensorboard_log_path = config.get('log_path') + time.strftime("%Y-%m-%d-%H%M%S") + '_' + config.get('run_name')
writer_train = SummaryWriter(tensorboard_log_path + '/train')
if should_validate_every_n_steps:
    writer_valid = SummaryWriter(tensorboard_log_path + '/valid')

vocabulary: 'Vocabulary' = Vocabulary.from_files(directory=config.get('vocab_dir_path'))

token_embedding: 'Embedding' = Embedding.from_params(vocab=vocabulary, params=Params({"pretrained_file": config.get('pre_trained_embedding_file_name'),
                                                                              "embedding_dim": config.get('emb_dim'),
                                                                              "trainable": config.get('embedding_trainable') == 'True',
                                                                              "max_norm": None, 
                                                                              "norm_type": 2.0,
                                                                              "padding_index": 0}))
word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
train_triple_loader = IrTripleDatasetReader(lazy=True,
                                       max_doc_length=config.get('max_doc_len'),
                                       max_query_length=config.get('max_q_len'),
                                       tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(),
                                                                 word_filter=StopwordFilter())) 
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
             dropout_parameter=config.get('dropout_probability'),
             regularization_term=config.get('regularization_term'),
             learning_rate=config.get('learning_rate')
            ).to(device)

logger.info('Model "{}" parameters: {}'.format(config.get('run_name'), sum(p.numel() for p in model.parameters() if p.requires_grad)))
logger.info('Network: {}'.format(repr(model)))

for n, p in model.named_parameters():
    logger.info('model parameter "{}": shape={}, requires_grad={}'.format(n, p.shape, p.requires_grad))

model_params = model.parameters()
optimizer = Adam(model_params, lr=config.get('learning_rate'))

num_training_steps = config.get('num_train_steps')
average_loss = 0
regularization_term = config.get('regularization_term')

curr_training_step = 0
should_save_snapshot_every_n_steps = config.get('save_snapshot_every_n_steps') > -1 


for batch in Tqdm.tqdm(iterator(train_triple_loader.read(config.get('training_data_triples_file')), num_epochs=1)):
    curr_training_step += 1
    batch = move_to_device(obj=batch, cuda_device=(0 if torch.cuda.is_available() else -1))

    if curr_training_step == 1:
        writer_train.add_graph(model, (batch["query_tokens"]["tokens"], 
                                batch["doc_pos_tokens"]["tokens"], 
                                batch["doc_neg_tokens"]["tokens"])) # adds graph of model to tensorboard
        writer_train.add_text('config', json.dumps(dict(config.items())), 0) # add config key-value pairs to tensorboard

    optimizer.zero_grad()  # zero the gradient buffers

    q_repr, doc_pos_repr, doc_neg_repr = model.forward(batch["query_tokens"]["tokens"], 
                                                        batch["doc_pos_tokens"]["tokens"], 
                                                        batch["doc_neg_tokens"]["tokens"])
    # q_repr and doc_pos_repr and doc_neg_repr have shape [batch_size, nn_output_dim]

    cost, hinge_loss, l1_regularization = obtain_loss(q_repr, doc_pos_repr, doc_neg_repr)

    if (curr_training_step < 1000 and curr_training_step % 10 == 0) or (curr_training_step >= 1000 and curr_training_step % 100 == 0):
        write_to_tensorboard(writer_train, curr_training_step, cost, hinge_loss, l1_regularization, q_repr, doc_pos_repr, doc_neg_repr)
        write_histograms_to_tensorboard(writer_train, curr_training_step, model, q_repr, doc_pos_repr, doc_neg_repr)
        
    if curr_training_step % 100 == 0:
        log_output_train = (curr_training_step, num_training_steps, cost.data, hinge_loss.data, l1_regularization.data)
        logger.info('Training step {:6d} of {:6d} \t Cost={:10.4f} (Hinge-Loss={:10.4f}, L1-Reg={:8.5f})'.format(*log_output_train))

    loss = cost
    loss.backward() # calculate gradients of weights for network
    optimizer.step()  # updates network with new weights

    if should_validate_every_n_steps and curr_training_step % config.get('validate_every_n_steps') == 0:
        num_validation_steps = config.get('num_valid_steps')
        logger.info('Validating model at step {:6d} - with {} validation steps'.format(curr_training_step, num_validation_steps))
        try:
            mean_validation_loss = validate_model(model, num_validation_steps, vocabulary, writer_valid, curr_training_step)
            logger.info('Average loss on validation set at step {}: {:10.4f}'.format(curr_training_step, mean_validation_loss))
        except Exception as e:
            # we do not care if something bad happens during validation, we keep on training and log the exception
            logger.exception('Failed to validate model at training step {}'.format(curr_training_step))
        finally:
            model.train() # Sets the module in training mode.

    if should_save_snapshot_every_n_steps and curr_training_step > 0 and curr_training_step % config.get('save_snapshot_every_n_steps') == 0:
        save_model(model, curr_training_step)

    if curr_training_step == num_training_steps:
        save_model(model)
        break

# TODO save model if we did not reach num_training_steps, but no more training samples are left

writer_train.flush()
writer_train.close()

if should_validate_every_n_steps:
    writer_valid.flush()
    writer_valid.close()
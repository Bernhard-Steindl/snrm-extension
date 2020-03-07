"""
The inference (retrieval) sample file.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

from app_logger import logger
logger = logger(__file__)

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({})) # sets the seeds to be fixed

from config import config
import params

from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import WordTokenizer

from data_loading import IrTupleDatasetReader

import pickle as pkl
import time
import numpy as np
import torch

from snrm import SNRM
from inverted_index import MemMappedInvertedIndex


# TODO extract to utils file, already used in train2 file
def count_zero(tensor: torch.Tensor) -> int:
    return (tensor == 0.0).sum().item()

# layer_size is a list containing the size of each layer. It can be set through the 'hidden_x' arguments.
layer_size = [config.get('emb_dim')] # input layer
for i in [config.get('hidden_1'), config.get('hidden_2'), config.get('hidden_3'), config.get('hidden_4'), config.get('hidden_5')]:
    if i <= 0:
        break
    layer_size.append(i)

logger.info('Loading vocabulary')
vocabulary: Vocabulary = Vocabulary.from_files(directory=config.get('vocab_dir_path'))
logger.info('Loading embedding')
token_embedding: Embedding = Embedding.from_params(vocab=vocabulary, params=Params({"pretrained_file": config.get('pre_trained_embedding_file_name'),
                                                                              "embedding_dim": config.get('emb_dim'),
                                                                              "trainable": False, # TODO is this ok?
                                                                              "max_norm": None, 
                                                                              "norm_type": 2.0,
                                                                              "padding_index": 0}))
word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
# The SNRM model.
model = SNRM(word_embeddings= word_embedder,
            batch_size=config.get('batch_size'),
            max_q_len=config.get('max_q_len'),
            max_doc_len=config.get('max_doc_len'),
            emb_dim=config.get('emb_dim'),
            layer_size=layer_size,
            dropout_parameter=config.get('dropout_probability'),
            regularization_term=config.get('regularization_term'),
            learning_rate=config.get('learning_rate'))

model_load_path = '{0}model-state_{1}.pt'.format(config.get('model_path'), config.get('run_name'))
logger.info('Restoring model parameters from "{}"'.format(model_load_path))
# restore model parameter
model.load_state_dict(torch.load(model_load_path))
model.eval() # set model in evaluation mode


def get_retrieval_queries():
    """
    Returns 
    TODO doc
    """
    # TODO which qrel file and which query file should we use?
    # queries.validation.tsv
    # queries.dev.small.tsv
    queries = dict()
    with open(config.get('base_path') + config.get('evaluation_query_file')) as f:
        for line in f:
            line_components = line.rstrip('\n').split('\t')
            qid = line_components[0]
            query_text = line_components[1]
            queries[qid] = query_text
    return queries


inverted_index = MemMappedInvertedIndex(layer_size[-1])
inverted_index.load()

logger.info('Initializing document tuple loader and iterator')
query_tuple_loader = IrTupleDatasetReader(lazy=True, 
                                             max_text_length=config.get('max_q_len'),
                                             tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())) 
                                             # already spacy tokenized, so that it is faster 
iterator = BucketIterator(batch_size=config.get('batch_size'), # TODO should we only process one query at a time? would be faster if > 1
                          sorting_keys=[("text_tokens", "num_tokens")])
iterator.index_with(vocabulary)


num_queries_processed = 0
batch_num = 0

current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
candidate_file_name = config.get('base_path') + config.get('evaluation_result_candidate_file_prefix') + current_timestamp_str + '.txt'
max_retrieval_docs = config.get('num_retrieval_documents_per_query') # we are only interested in the top k document for a query

with open(candidate_file_name, 'w') as evaluationCandidateFile:
    logger.info('Created and writing evaluation candidate file "{}"'.format(candidate_file_name))
    
    for batch in Tqdm.tqdm(iterator(query_tuple_loader.read(config.get('evaluation_query_file')), num_epochs=1)):
        batch_num += 1
        
        query_ids = batch['id']
        query_repr, _, _ = model.forward(batch['text_tokens']['tokens'], None, None)

        if batch_num % 4 == 0:
            zero_elements = count_zero(query_repr)
            num_elements = query_repr.numel()
            ratio_zero = (zero_elements / num_elements)
            logger.info('query_repr batch #{} has zero elements={}, total size={}'.format(batch_num, zero_elements, num_elements))
            logger.info('query_repr batch #{} has ratio_zero_elements={:6.5f}'.format(batch_num, ratio_zero))
            logger.info('retrieving document scores for query qid={}'.format(repr(query_ids)))

        num_queries_in_batch = query_repr.shape[0]
        for q in range(num_queries_in_batch):
            query_repr_v = query_repr[q]
            qid = query_ids[q]
            retrieval_scores = dict() # maps doc_id to retrieval score for the query

            if q % 10 == 0:
                zero_elements = count_zero(query_repr_v)
                num_elements = query_repr_v.numel()
                ratio_zero = (zero_elements / num_elements)
                logger.debug('query_repr qid={} has zero elements={}, total size={}'.format(qid, zero_elements, num_elements))
                logger.debug('query_repr qid={} has ratio_zero_elements={:6.5f}'.format(qid, ratio_zero))

            sum_docs_processed = 0
            for i in range(len(query_repr_v)): # TODO can this be optimized / parallelized?
                if query_repr_v[i] > 0.:
                    if not i in inverted_index.index:
                        # logger.debug('A latent term dimension (dim={}) of a query (qid={}) has no assigned documents in index'.format(i, qid))
                        # TODO log or write something
                        continue # no document is in this latent term dimension
                    # logger.info('found {} docs in latent term dimension {}'.format(str(len(inverted_index.index[i])),str(i)))
                    for doc_id in inverted_index.index[i]: # for every doc in the current latent term dimension
                        sum_docs_processed += 1
                        if doc_id not in retrieval_scores:
                            retrieval_scores[doc_id] = 0.
                        doc_representation_v = inverted_index.get_doc_representation(doc_id)
                        weight = doc_representation_v[i]
                        retrieval_scores[doc_id] += query_repr_v[i] * weight
            mean_docs_processed = round(sum_docs_processed / len(query_repr_v), 3)
            #logger.debug('processed avg. {} non-distinct docs for query per dimension (whole: {})'.format(str(mean_docs_processed), str(sum_docs_processed)))
            logger.info('Obtained a score for {} distinct docs for query qid={}'.format(len(retrieval_scores), qid))

            retrieval_result_for_qid = sorted(retrieval_scores.items(), key=lambda x: x[1], reverse=True)
            retrieval_result_for_qid = retrieval_result_for_qid[:max_retrieval_docs]

            if len(retrieval_result_for_qid) == 0:
                logger.warning('Could not retrieve any relevant document for query qid={}'.format(qid))

            # writing retrieval result to candidate file
            for rank, (doc_id, retrieval_score) in enumerate(retrieval_result_for_qid):
                # logger.debug('qid={}\t\tdoc_id={}\tscore={}\trank={}'.format(qid,doc_id,retrieval_score, rank+1))
                evaluationCandidateFile.write('{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n'.format(qid, doc_id, rank+1, retrieval_score, config.get('run_name')))
                if rank < 10:
                    logger.debug('{0}\tQ0\t{1}\t{2}\t{3}\t{4}'.format(qid, doc_id, rank+1, retrieval_score, config.get('run_name')))

            num_queries_processed += 1
            if num_queries_processed == config.get('num_evaluation_queries'):
                logger.info('Ending retrieval after processing {} queries'.format(num_queries_processed))
                break
        logger.info('Processed {} queries for retrieval'.format(num_queries_processed))
        if num_queries_processed == config.get('num_evaluation_queries'):
            break




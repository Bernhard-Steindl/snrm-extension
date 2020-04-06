import logging
FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filemode='a', filename='results/index_stats.log')

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({})) # sets the seeds to be fixed

from config import config
import params

import numpy as np
from inverted_index import MemMappedInvertedIndex
import torch
from snrm import SNRM
import sys


from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import WordTokenizer

from data_loading import IrTupleDatasetReader
from allennlp.data.tokenizers.word_filter import StopwordFilter

from allennlp.nn.util import move_to_device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('PyTorch uses device {}'.format(device))


def get_layer_size():
    layer_size = [config.get('emb_dim')] # input layer
    for i in [config.get('hidden_1'), config.get('hidden_2'), config.get('hidden_3'), config.get('hidden_4'), config.get('hidden_5')]:
        if i <= 0:
            break
        layer_size.append(i)
    return layer_size

def evaluate_document_repr():
    layer_size = get_layer_size()
    inverted_index = MemMappedInvertedIndex(layer_size[-1])
    inverted_index.load()
    logging.info('Loaded inverted index')

    np.set_printoptions(precision=12)

    doc_id_to_memmap_idx = inverted_index._doc_id_to_memmap_idx
    doc_repr_memmap = inverted_index._doc_repr_memmap
    index = inverted_index.index

    num_distinct_docs = inverted_index.count_documents()
    logging.info('Inverted Index contains {} distinct docs'.format(num_distinct_docs))
    
    doc_id_to_occurence = dict() # maps doc_id to num occurence in latent term dimensions
    num_docs_per_latent_dim = []
    for latent_dim in index:
        num_docs = len(index[latent_dim])
        num_docs_per_latent_dim.append(num_docs)
        for doc_id in index[latent_dim]:
            if doc_id not in doc_id_to_occurence:
                doc_id_to_occurence[doc_id] = 1
            else:
                doc_id_to_occurence[doc_id] += 1
        # logging.info('docs in latent dim {}:\t{}'.format(latent_dim, repr(index[latent_dim])))
    
    logging.info('=============================')
    logging.info('now printing statistics of docs in index resp. latent term dimensions')
    logging.info('number of documents per latent dimension: min={}'.format(np.min(num_docs_per_latent_dim)))
    logging.info('number of documents per latent dimension: max={}'.format(np.max(num_docs_per_latent_dim)))
    logging.info('number of documents per latent dimension: mean={}'.format(np.mean(num_docs_per_latent_dim)))
    logging.info('number of documents per latent dimension: median={}'.format(np.median(num_docs_per_latent_dim)))
    logging.info('number of documents per latent dimension: 0.75-quantile={:.4f}'.format(np.quantile(num_docs_per_latent_dim, 0.75),4))
    logging.info('number of documents per latent dimension: 0.80-quantile={:.4f}'.format(np.quantile(num_docs_per_latent_dim, 0.8),4))
    logging.info('number of documents per latent dimension: 0.90-quantile={:.4f}'.format(np.quantile(num_docs_per_latent_dim, 0.9),4))

    logging.info('=============================')
    logging.info('now printing statistics per doc representation')
    sum_doc_repr = dict()
    mean_doc_repr = dict()
    median_doc_repr = dict()
    max_doc_repr = dict()
    min_doc_repr = dict()
    quantile_80_doc_repr = dict()
    quantile_95_doc_repr = dict()
    quantile_99_doc_repr = dict()

    for i, doc_id in enumerate(doc_id_to_occurence):
        doc_repr_idx = doc_id_to_memmap_idx[doc_id]
        doc_repr = doc_repr_memmap[doc_repr_idx]

        max_doc_repr[doc_id] = np.max(doc_repr)
        min_doc_repr[doc_id] = np.min(doc_repr)
        sum_doc_repr[doc_id] = np.sum(doc_repr)
        mean_doc_repr[doc_id] = np.mean(doc_repr)
        median_doc_repr[doc_id] = np.median(doc_repr)
        quantile_80_doc_repr[doc_id] = np.quantile(doc_repr, 0.8)
        quantile_95_doc_repr[doc_id] = np.quantile(doc_repr, 0.95)
        quantile_99_doc_repr[doc_id] = np.quantile(doc_repr, 0.99)
    
    for i, doc_id in enumerate(doc_id_to_occurence):
        logging.info('{}. doc_repr values doc_id={}: min={:.5f},\tmax={:.5f},\tsum={:.5f},\tmean={:.7f},\t0.80-quantil={:.7f},\t0.95-quantil={:.7f}\t0.99-quantil={:.10f}'
            .format(i+1, doc_id, min_doc_repr[doc_id], max_doc_repr[doc_id], sum_doc_repr[doc_id], mean_doc_repr[doc_id], median_doc_repr[doc_id], quantile_80_doc_repr[doc_id], quantile_95_doc_repr[doc_id], quantile_99_doc_repr[doc_id]))

    logging.info('=============================')
    logging.info('now printing number of occurences of docs in latent term dimensions, sorted ascending by occurence')
    
    for doc_id, occurence in sorted(doc_id_to_occurence.items(), key=lambda item: item[1]):
        logging.info('doc_id={:6},\toccurence={:4},\tsum_doc_repr_values={:8.7f},\tmean_doc_repr_values={:8.7f}'.format(doc_id, occurence, sum_doc_repr[doc_id], mean_doc_repr[doc_id]))

def evaluate_query_repr():
    layer_size = get_layer_size()
    logging.info('Loading vocabulary')
    vocabulary: Vocabulary = Vocabulary.from_files(directory=config.get('vocab_dir_path'))
    logging.info('Loading embedding')
    token_embedding: Embedding = Embedding.from_params(vocab=vocabulary, params=Params({"pretrained_file": config.get('pre_trained_embedding_file_name'),
                                                                                "embedding_dim": config.get('emb_dim'),
                                                                                "trainable": False, # TODO is this ok?
                                                                                "max_norm": None, 
                                                                                "norm_type": 2.0,
                                                                                "padding_index": 0}))
    word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
    logging.debug('Now Loading model')
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
    logging.info('Restoring model parameters from "{}"'.format(model_load_path))
    # restore model parameter https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # model.load_state_dict(torch.load(model_load_path))
    # if you saved model on GPU and now want to load it on a CPU:
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to(device)
    model.eval() # set model in evaluation mode
    
    sum_q_repr = dict()
    mean_q_repr = dict()
    median_q_repr = dict()
    max_q_repr = dict()
    min_q_repr = dict()
    quantile_80_q_repr = dict()
    quantile_95_q_repr = dict()
    quantile_99_q_repr = dict()
    ratio_non_zero_q_repr = dict()


    logging.info('Initializing document tuple loader and iterator')
    query_tuple_loader = IrTupleDatasetReader(lazy=True, 
                                                max_text_length=config.get('max_q_len'),
                                                tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(),
                                                                          word_filter=StopwordFilter())) 
                                                # already spacy tokenized, so that it is faster 
    iterator = BucketIterator(batch_size=config.get('batch_size'), # TODO should we only process one query at a time? would be faster if > 1
                            sorting_keys=[("text_tokens", "num_tokens")])
    iterator.index_with(vocabulary)

    batch_num = 0
    for batch in Tqdm.tqdm(iterator(query_tuple_loader.read(config.get('evaluation_query_file')), num_epochs=1)):
        batch_num += 1
        batch = move_to_device(obj=batch, cuda_device=(0 if torch.cuda.is_available() else -1))

        query_ids = batch['id']
        query_repr, _, _ = model.forward(batch['text_tokens']['tokens'], None, None)

        num_queries_in_batch = query_repr.shape[0]
        for q in range(num_queries_in_batch):
            query_repr_v = query_repr[q].detach().numpy()
            qid = query_ids[q]

            retrieval_scores = dict()

            non_zero_elements = np.count_nonzero(query_repr_v)
            num_elements = len(query_repr_v)
            ratio_non_zero = (non_zero_elements / num_elements)

            ratio_non_zero_q_repr[qid] = ratio_non_zero
            max_q_repr[qid] = np.max(query_repr_v)
            min_q_repr[qid] = np.min(query_repr_v)
            sum_q_repr[qid] = np.sum(query_repr_v)
            mean_q_repr[qid] = np.mean(query_repr_v)
            median_q_repr[qid] = np.median(query_repr_v)
            quantile_80_q_repr[qid] = np.quantile(query_repr_v, 0.8)
            quantile_95_q_repr[qid] = np.quantile(query_repr_v, 0.95)
            quantile_99_q_repr[qid] = np.quantile(query_repr_v, 0.99)

    for i, qid in enumerate(sum_q_repr):
        logging.info('{}. query_repr values qid={}: ratio_non_zero={:.5f},\tmin={:.5f},\tmax={:.5f},\tsum={:.5f},\tmean={:.7f},\t0.80-quantil={:.7f},\t0.95-quantil={:.7f}\t0.99-quantil={:.10f}'
            .format(i+1, qid, ratio_non_zero_q_repr[qid], min_q_repr[qid], max_q_repr[qid], sum_q_repr[qid], mean_q_repr[qid], median_q_repr[qid], quantile_80_q_repr[qid], quantile_95_q_repr[qid], quantile_99_q_repr[qid]))



def usage():
    print('Usage: index_stats.py {query|doc}')
    exit()


def main():
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'query':
            evaluate_query_repr()
            return
        if arg == 'doc':
            evaluate_document_repr()
            return
        else:
            usage()
            return
    else:
        evaluate_query_repr()
        evaluate_document_repr()


if __name__ == '__main__':
    """
    Generates statistics from already created inverted index (doc sparsity) and/or model's query sparsity

    
    """
    
    if len(sys.argv) > 2:
        usage()

    main()

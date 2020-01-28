import logging
FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, filemode='a', filename='results/index_stats.log')

import numpy as np
import tensorflow as tf
from dictionary import Dictionary
from inverted_index import MemMappedInvertedIndex
from params import FLAGS
from snrm import SNRM

def get_retrieval_queries():
    """
    Returns 
    TODO doc
    copy from retrieval.py
    """
    queries = dict()
    with open(FLAGS.base_path + FLAGS.evaluation_query_file) as f:
        for line in f:
            line_components = line.rstrip('\n').split('\t')
            qid = line_components[0]
            query_text = line_components[1]
            queries[qid] = query_text
    return queries

def main():
    layer_size = [FLAGS.emb_dim]
    for i in [FLAGS.hidden_1, FLAGS.hidden_2, FLAGS.hidden_3, FLAGS.hidden_4, FLAGS.hidden_5]:
        if i <= 0:
            break
        layer_size.append(i)
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
    quantile_80_doc_repr = dict()
    quantile_95_doc_repr = dict()
    quantile_99_doc_repr = dict()

    for i, doc_id in enumerate(doc_id_to_occurence):
        doc_repr_idx = doc_id_to_memmap_idx[doc_id]
        doc_repr = doc_repr_memmap[doc_repr_idx]

        max_doc_repr[doc_id] = np.max(doc_repr)
        sum_doc_repr[doc_id] = np.sum(doc_repr)
        mean_doc_repr[doc_id] = np.mean(doc_repr)
        median_doc_repr[doc_id] = np.median(doc_repr)
        quantile_80_doc_repr[doc_id] = np.quantile(doc_repr, 0.8)
        quantile_95_doc_repr[doc_id] = np.quantile(doc_repr, 0.95)
        quantile_99_doc_repr[doc_id] = np.quantile(doc_repr, 0.99)
    
    for i, doc_id in enumerate(doc_id_to_occurence):
        logging.info('{}. doc_repr values doc_id={}: max={:.5f},\tsum={:.5f},\tmean={:.7f},\t0.80-quantil={:.7f},\t0.95-quantil={:.7f}\t0.99-quantil={:.10f}'
            .format(i+1, doc_id, max_doc_repr[doc_id], sum_doc_repr[doc_id], mean_doc_repr[doc_id], median_doc_repr[doc_id], quantile_80_doc_repr[doc_id], quantile_95_doc_repr[doc_id], quantile_99_doc_repr[doc_id]))

    logging.info('=============================')
    logging.info('now printing number of occurences of docs in latent term dimensions, sorted ascending by occurence')
    
    for doc_id, occurence in sorted(doc_id_to_occurence.items(), key=lambda item: item[1]):
        logging.info('doc_id={:6},\toccurence={:4},\tsum_doc_repr_values={:8.7f},\tmean_doc_repr_values={:8.7f}'.format(doc_id, occurence, sum_doc_repr[doc_id], mean_doc_repr[doc_id]))


    dictionary = Dictionary()
    dictionary.load_from_galago_dump(FLAGS.base_path + FLAGS.dict_file_name)
    logging.info('Loaded vocabulary into dictionary')
    logging.debug('Now Loading model')
    # The SNRM model.
    snrm = SNRM(dictionary=dictionary,
                pre_trained_embedding_file_name=FLAGS.base_path + FLAGS.pre_trained_embedding_file_name,
                batch_size=FLAGS.batch_size,
                max_q_len=FLAGS.max_q_len,
                max_doc_len=FLAGS.max_doc_len,
                emb_dim=FLAGS.emb_dim,
                layer_size=layer_size,
                dropout_parameter=FLAGS.dropout_parameter,
                regularization_term=FLAGS.regularization_term,
                learning_rate=FLAGS.learning_rate)
    
    sum_q_repr = dict()
    mean_q_repr = dict()
    median_q_repr = dict()
    max_q_repr = dict()
    min_q_repr = dict()
    quantile_80_q_repr = dict()
    quantile_95_q_repr = dict()
    quantile_99_q_repr = dict()
    ratio_non_zero_q_repr = dict()

    with tf.Session(graph=snrm.graph) as session:
        session.run(snrm.init)
        logging.debug('Initialized tf session')

        snrm.saver.restore(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)  # restore all variables
        logging.debug('Load model from {:s}'.format(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name))
        queries = get_retrieval_queries()
        num_queries = len(queries.keys())
        
        for qid in queries:
            q_term_ids = dictionary.get_term_id_list(queries[qid])
            q_term_ids.extend([0] * (FLAGS.max_q_len - len(q_term_ids)))
            q_term_ids = q_term_ids[:FLAGS.max_q_len]

            # logging.debug('retrieving document scores for query qid={}'.format(qid))
            query_repr = session.run(snrm.query_representation, feed_dict={snrm.test_query_pl: [q_term_ids]})
            retrieval_scores = dict()
            query_repr_v = query_repr[0]

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




if __name__ == '__main__':
    """
    Generates statistics from already created inverted index
    """
    main()

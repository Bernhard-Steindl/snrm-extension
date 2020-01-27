import logging
FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, filemode='a', filename='results/index_stats.log')

import numpy as np
from dictionary import Dictionary
from inverted_index import MemMappedInvertedIndex
from params import FLAGS

def main():
    dictionary = Dictionary()
    dictionary.load_from_galago_dump(FLAGS.base_path + FLAGS.dict_file_name)
    logging.info('Loaded vocabulary into dictionary')
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
        logging.info('doc_id={} (occurence={})\t'.format(doc_id, occurence))










if __name__ == '__main__':
    """
    Generates statistics from already created inverted index
    """
    main()

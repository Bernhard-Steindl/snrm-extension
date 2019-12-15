"""
The inference (retrieval) sample file.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import logging
import tensorflow as tf
import pickle as pkl
import time

from dictionary import Dictionary
from inverted_index import InMemoryInvertedIndex
from params import FLAGS
from snrm import SNRM

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# layer_size is a list containing the size of each layer. It can be set through the 'hidden_x' arguments.
layer_size = [FLAGS.emb_dim]
for i in [FLAGS.hidden_1, FLAGS.hidden_2, FLAGS.hidden_3, FLAGS.hidden_4, FLAGS.hidden_5]:
    if i <= 0:
        break
    layer_size.append(i)

# Dictionary is a class containing terms and their IDs. The implemented class just load the terms from a Galago dump
# file. If you are not using Galago, you have to implement your own reader. See the 'dictionary.py' file.
dictionary = Dictionary()
dictionary.load_from_galago_dump(FLAGS.base_path + FLAGS.dict_file_name)

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

def get_retrieval_queries():
    """
    Returns 
    TODO doc
    """
    # TODO which qrel file and which query file should we use?
    # queries.validation.tsv
    # queries.dev.small.tsv
    queries = dict()
    with open(FLAGS.base_path + FLAGS.evaluation_query_file) as f:
        for line in f:
            line_components = line.rstrip('\n').split('\t')
            qid = line_components[0]
            query_text = line_components[1]
            queries[qid] = query_text
    return queries

def write_retrieval_result_in_candidate_file(retrieval_result): 
    """
    Returns 
    TODO doc
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file should contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank

    python msmarco_eval.py <path-to-qrels-file> <path-to-evaluation-candidate-file> 
    TODO adjust for TREC format
    ./evaluation-tools/trec_eval/trec_eval /Users/bernhardsteindl/Development/python_workspace/bachelorarbeit/snrm-extension/data/evaluation/qrels.dev.small.tsv /Users/bernhardsteindl/Downloads/evaluation_candidate_2019-12-08_221238.txt

    """
    current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
    candidate_file_name = FLAGS.base_path + FLAGS.evaluation_result_candidate_file_prefix + current_timestamp_str
    with open(candidate_file_name, 'w') as f:
        for qid in retrieval_result.keys():
            for rank, (doc_id, retrieval_score) in enumerate(retrieval_result[qid]):
                # logging.debug('qid={}\t\tdoc_id={}\tscore={}\trank={}'.format(qid,doc_id,retrieval_score, rank+1))
                f.write('{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n'.format(qid, doc_id, rank+1, retrieval_score, FLAGS.run_name))

inverted_index = InMemoryInvertedIndex(layer_size[-1])
inverted_index.load(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + '-inverted-index.pkl')

with tf.Session(graph=snrm.graph) as session:
    session.run(snrm.init)
    print('Initialized')

    snrm.saver.restore(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)  # restore all variables
    logging.info('Load model from {:s}'.format(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name))

    queries = get_retrieval_queries()

    result = dict()
    for qid in queries:
        # logging.info('processing query #' + qid + ': ' + queries[qid])
        q_term_ids = dictionary.get_term_id_list(queries[qid])
        q_term_ids.extend([0] * (FLAGS.max_q_len - len(q_term_ids)))
        q_term_ids = q_term_ids[:FLAGS.max_q_len]

        query_repr = session.run(snrm.query_representation, feed_dict={snrm.test_query_pl: [q_term_ids]})
        retrieval_scores = dict()

        for i in range(len(query_repr[0])):
            if query_repr[0][i] > 0.:
                for (did, weight) in inverted_index.index[i]:
                    if did not in retrieval_scores:
                        retrieval_scores[did] = 0.
                    retrieval_scores[did] += query_repr[0][i] * weight

        result[qid] = sorted(retrieval_scores.items(), key=lambda x: x[1], reverse=True)
    
    write_retrieval_result_in_candidate_file(result)
    pkl.dump(result, open(FLAGS.base_path + FLAGS.result_path + FLAGS.run_name + '-test-queries.pkl', 'wb'))





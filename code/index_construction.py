"""
Inverted index construction from the latent terms to document IDs from the representations learned by SNRM.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import logging
import tensorflow as tf

from dictionary import Dictionary
from inverted_index import MemMappedInvertedIndex
from params import FLAGS
from snrm import SNRM

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# layer_size is a list containing the size of each layer. It can be set through the 'hiddein_x' arguments.
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


def generate_batch(batch_size, last_file_position = -1):
    """
        Generating a batch of documents from the collection for making the inverted index. This function should iterate
        over all the documents (each once) to learn an inverted index.
        Args:
            batch_size (int): total number of training or validation data in each batch.

        Returns:
            batch_doc_id (list): a list of str containing document IDs.
            batch_doc (list): a 2D list of int containing document term IDs with size (batch_size * FLAGS.max_doc_len).

        # TODO documentation 
    """
    # raise Exception('the generate_batch method is not implemented.')
    batch_doc_id = []
    batch_doc = []

    num_lines_per_batch = batch_size

    with open(FLAGS.base_path + FLAGS.document_collection_file, 'r') as file:
        if last_file_position == -1:
            file.tell()
        else:
            file.seek(last_file_position)

        for relative_line_num in range(num_lines_per_batch):
            line = file.readline()
            # logging.debug('relative_line_num={}\n{}'.format(relative_line_num, line))
            if line == '':
                raise ValueError('Failed to generate batch, because file does not have enough lines left.')

            line_components = line.rstrip('\n').split('\t')
            # tsv: pid, passage
            passage_id = line_components[0]
            passage_text = line_components[1]
            # logging.debug('passage_id={}, \t passage={}'.format(passage_id, passage_text))

            passage_term_ids = dictionary.get_term_id_list(passage_text)
            passage_term_ids.extend([0] * (FLAGS.max_doc_len - len(passage_term_ids)))
            passage_term_ids = passage_term_ids[:FLAGS.max_doc_len]

            batch_doc_id.append(passage_id)
            batch_doc.append(passage_term_ids)
            # logging.debug('passage_id={}, \t passage_term_ids len={}'.format(passage_id, len(passage_term_ids)))
            # logging.debug('passage_id={}, \t passage_term_ids={}'.format(passage_id, repr(passage_term_ids)))
        last_file_position = file.tell()
    return batch_doc_id, batch_doc, last_file_position

inverted_index = MemMappedInvertedIndex(layer_size[-1])
inverted_index.create()

with tf.Session(graph=snrm.graph) as session:
    session.run(snrm.init)
    print('Initialized')

    snrm.saver.restore(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)  # restore all variables
    logging.info('Load model from {:s}'.format(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name))

    # TODO use for what?
    docs = []
    doc_names = []
    # col_file = open(FLAGS.base_path + FLAGS.model_path + 'learned_robust.txt', 'wb')
    # doc_len_file = open(FLAGS.base_path + FLAGS.model_path + 'learned_robust_doc_len.txt', 'wb')

    # TODO why is here this loop? infinite loop?
    # TODO should batch_size here be the document collection size or a subset?
    # TODO is the doc_repr always the same for the same batch or why should we need a loop?
    # while True:
    #     #     doc_ids, docs = generate_batch(FLAGS.batch_size)
    #     #     try:
    #     #         doc_repr = session.run(snrm.doc_representation, feed_dict={snrm.doc_pl: docs})
    #     #         inverted_index.add(doc_ids, doc_repr)
    #     #     except Exception as ex:
    #     #         break


    last_doc_collection_file_position = -1

    for batch_num in range(FLAGS.num_document_batches):
        if batch_num % 50 == 0:
            logging.debug('generating document representation batch_num={} \t num_document_batches={}'.format(batch_num+1, FLAGS.num_document_batches))
        try:
            doc_ids, docs, last_doc_collection_file_position = generate_batch(FLAGS.batch_size_documents, last_doc_collection_file_position)
            doc_repr = session.run(snrm.doc_representation, feed_dict={snrm.doc_pl: docs})
            inverted_index.add(doc_ids, doc_repr)
        except Exception as ex:
            print(ex)
            break

    # for i in range(len(doc_ids)):
    #  logging.debug('adds doc_repr to index\tdoc_id={},\tdoc_repr=\n{}'.format(str(doc_ids[i]), repr(doc_repr[i])))
try:
    inverted_index.store()
except Exception as ex:
    print(ex)
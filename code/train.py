"""
Training the SNRM model.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import logging
import numpy as np
import tensorflow as tf

from dictionary import Dictionary
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
            # TODO should pre_trained_embedding_file_name be None?
            # pre_trained_embedding_file_name=FLAGS.base_path + FLAGS.pre_trained_embedding_file_name,
            pre_trained_embedding_file_name=None,
            batch_size=FLAGS.batch_size,
            max_q_len=FLAGS.max_q_len,
            max_doc_len=FLAGS.max_doc_len,
            emb_dim=FLAGS.emb_dim,
            layer_size=layer_size,
            dropout_parameter=FLAGS.dropout_parameter,
            regularization_term=FLAGS.regularization_term,
            learning_rate=FLAGS.learning_rate)



def generate_batch(batch_size, mode='train', last_file_position = -1):
    """
        Generating pairwise training or validation data for each batch. This function should be implemented.
        Note: For unknown terms term ID should be set to zero. Please use the dictionary size for padding. In other
        words, padding value should be |V|+1, where |V| is the vocabulary size.
        Args:
            batch_size (int): total number of training or validation data in each batch.
            mode (str): should be either 'train' or 'valid'.

        Returns:
            batch_query (list): a 2D list of int containing query term IDs with size (batch_size * FLAGS.max_q_len).
            batch_doc1 (list): a 2D list of int containing doc 1 term IDs with size (batch_size * FLAGS.max_doc_len).
            batch_doc2 (list): a 2D list of int containing doc 2 term IDs with size (batch_size * FLAGS.max_doc_len).
            batch_label (list): a 2D list of float within the range of [0, 1] with size (batch_size * 1).
             Label shows the probability of doc1 being more relevant than doc2. This can simply be 0 or 1.
    """

    # logging.info('args: batch_size={}, mode={}'.format(batch_size, mode))

    # TODO  what is meant with 'Please use the dictionary size for padding. In other words, padding value should be |V|+1, where |V| is the vocabulary size.'

    # TODO handle mode = 'valid'
    batch_query = []
    batch_doc1 = []
    batch_doc2 = []
    batch_label = []

    num_lines_per_batch = batch_size

    with open(FLAGS.base_path + FLAGS.training_data_triples_file, 'r') as file:
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
            query = line_components[0]
            positive_passage = line_components[1]
            negative_passage = line_components[2]
            # logging.info('query={}, pos_passage={}, neg_passage={}'.format(query, positive_passage, negative_passage))

            query_term_ids = dictionary.get_term_id_list(query)
            pos_passage_term_ids = dictionary.get_term_id_list(positive_passage)
            neg_passage_term_ids = dictionary.get_term_id_list(negative_passage)

            query_term_ids.extend([0] * (FLAGS.max_q_len - len(query_term_ids)))
            query_term_ids = query_term_ids[:FLAGS.max_q_len]
            pos_passage_term_ids.extend([0] * (FLAGS.max_doc_len - len(pos_passage_term_ids)))
            pos_passage_term_ids = pos_passage_term_ids[:FLAGS.max_doc_len]
            neg_passage_term_ids.extend([0] * (FLAGS.max_doc_len - len(neg_passage_term_ids)))
            neg_passage_term_ids = neg_passage_term_ids[:FLAGS.max_doc_len]

            # logging.debug('query_term_ids={}'.format(repr(query_term_ids)))
            # logging.debug('pos_passage_term_ids={}'.format(repr(pos_passage_term_ids)))
            # logging.debug('neg_passage_term_ids={}'.format(repr(neg_passage_term_ids)))

            batch_query.append(query_term_ids)
            batch_doc1.append(pos_passage_term_ids)
            batch_doc2.append(neg_passage_term_ids)
            batch_label.append(1)  # doc1 is better match for query than doc2
        last_file_position = file.tell()

    return batch_query, batch_doc1, batch_doc2, batch_label, last_file_position


writer = tf.summary.FileWriter(FLAGS.base_path + FLAGS.log_path + FLAGS.run_name, graph=snrm.graph)

# Launch the graph
with tf.Session(graph=snrm.graph) as session:
    session.run(snrm.init)
    logging.info('Initialized')

    ckpt = tf.train.get_checkpoint_state(FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)

    if ckpt and ckpt.model_checkpoint_path:
        logging.info(ckpt.model_checkpoint_path)
        snrm.saver.restore(session, ckpt.model_checkpoint_path)  # restore all variables
        logging.info('Load model from model_checkpoint_path {:s}'.format(ckpt.model_checkpoint_path))

    # training
    if not FLAGS.experiment_mode:
        num_steps = FLAGS.num_train_steps
        average_loss = 0
        last_file_position_training = -1
        last_file_position_validation = -1

        for step in range(num_steps):
            logging.info('training step {} of {}'.format(step+1, num_steps))
            query, doc1, doc2, labels, last_file_position_training = generate_batch(FLAGS.batch_size, 'train', last_file_position_training)
            labels = np.array(labels)
            labels = np.concatenate(
                [labels.reshape(FLAGS.batch_size, 1), 1. - labels.reshape(FLAGS.batch_size, 1)], axis=1)
            feed_dict = {snrm.query_pl: query,
                         snrm.doc1_pl: doc1,
                         snrm.doc2_pl: doc2,
                         snrm.labels_pl: labels}
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val, summary = session.run([snrm.optimizer, snrm.loss, snrm.summary_op], feed_dict=feed_dict)

            writer.add_summary(summary, step)

            # TODO this block, remove condition 'and step > 0'
            if step % FLAGS.validate_every_n_steps == 0 and step > 0:
                valid_loss = 0.
                valid_id = 0
                for valid_step in range(FLAGS.num_valid_steps):
                    query, doc1, doc2, labels, last_file_position_validation = generate_batch(FLAGS.batch_size, 'valid', last_file_position_validation)
                    labels = np.array(labels)
                    labels = np.concatenate(
                        [labels.reshape(FLAGS.batch_size, 1), 1. - labels.reshape(FLAGS.batch_size, 1)], axis=1)
                    feed_dict = {snrm.query_pl: query,
                                 snrm.doc1_pl: doc1,
                                 snrm.doc2_pl: doc2,
                                 snrm.labels_pl: labels}
                    loss_val = session.run(snrm.loss, feed_dict=feed_dict)
                    valid_loss += loss_val
                valid_loss /= FLAGS.num_valid_steps
                print('Average loss on validation set at step ', step, ': ', valid_loss)

            if step > 0 and step % FLAGS.save_snapshot_every_n_steps == 0:
                save_path = snrm.saver.save(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name + str(step))
                print('Model saved in file: %s' % save_path)

        save_path = snrm.saver.save(session, FLAGS.base_path + FLAGS.model_path + FLAGS.run_name)
        print('Model saved in file: %s' % save_path)

    else:
        print('Experiment Mode is ON!')
        # TODO is this experimentation not the same as if we would call index_construction.py and retrieval.py sequentially?
        # TODO is here anything to do necessarily?
        #
        # inference should be done. You should implement it. It's easy. Please refer to the paper. You should just
        # construct the inverted index from the learned representations. Then the query should fed to the network and
        # the documents that contain the "query latent terms" should be scored and ranked. If you have any question,
        # please do not hesitate to contact me (zamani@cs.umass.edu).

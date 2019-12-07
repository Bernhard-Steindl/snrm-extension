"""
Parameter file.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import random
import tensorflow as tf

# TODO is an experiment_mode necessary? see train.py
tf.flags.DEFINE_boolean('experiment_mode', False, 'Experiment mode is equivalent to testing a pre-trained model.')

# Note: all the following relative addresses should be relative to the base_path.
tf.flags.DEFINE_string('dict_file_name', 'data/allen_vocab_lower_10/tokens.txt', 'Relative address to the collection stats file.')

# TODO where can we get an embedding file? do we need one?
tf.flags.DEFINE_string('pre_trained_embedding_file_name', 'data/vectors.6B.100d.txt',
                       'Relative address to the pre-trained embedding file. default dim: 100.')

# paths on file system
tf.flags.DEFINE_string('base_path', '', 'The base path for codes and data.')
tf.flags.DEFINE_string('log_path', 'tf-log/', 'TensorFlow logging directory.')
tf.flags.DEFINE_string('model_path', 'model/', 'TensorFlow model directory.')
tf.flags.DEFINE_string('result_path', 'results/', 'TensorFlow model directory.')
tf.flags.DEFINE_string('run_name', 'example-run', 'A name for the run.')

# TODO which parameter values should we set here?
tf.flags.DEFINE_integer('batch_size', 10, 'Batch size for training. default: 512.')
tf.flags.DEFINE_integer('num_train_steps', 10, 'Number of steps for training. default: 100000.')
tf.flags.DEFINE_integer('num_valid_steps', 1000, 'Number of steps for training. default: 1000.')
tf.flags.DEFINE_integer('emb_dim', 100, 'Embedding dimensionality for words. default: 100.')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for Adam Optimizer. default: 0.0001.')
tf.flags.DEFINE_float('dropout_parameter', 1.0, 'Dropout parameter. default: 1.0 (no dropout).')
tf.flags.DEFINE_float('regularization_term', 0.0001, 'Dropout parameter. default: 0.0001 (it is not a good value).')

# TODO how to define hidden layer size?
tf.flags.DEFINE_integer('hidden_1', 4, 'Size of the first hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_2', 4, 'Size of the second hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_3', 4, 'Size of the third hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_4', 4, 'Size of the third hidden layer. Should be positive. default: -1.')
tf.flags.DEFINE_integer('hidden_5', 4, 'Size of the third hidden layer. Should be positive. default: -1.')

# TODO determine values for these settings
tf.flags.DEFINE_integer('validate_every_n_steps', 10000,
                        'Print the average loss value on the validation set at every n steps. default: 10000.')
tf.flags.DEFINE_integer('save_snapshot_every_n_steps', 10000, 'Save the model every n steps. default: 10000.')

# TODO set reasonable values
tf.flags.DEFINE_integer('max_q_len', 10, 'Maximum query length. default: 10.')
tf.flags.DEFINE_integer('max_doc_len', 128, 'Maximum document length. default: 1000.')

# TODO min_freq of term occurence in document collection to be added to term dictionary; 
#      might not be needed because we use AllenNLP vocabulary (either allen_voab_lower_10 or allen_vocab_lower_5)? 
#      where _10 = min of 10 occurrences, _5 = min of 5 term occurrences
#      therefore already determined by vocabulary dataset
# TODO from which document collection source is this vocabulary? (origin air-exercise)
tf.flags.DEFINE_integer('dict_min_freq', 20, 'minimum collection frequency of terms for dictionary. default: 20')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.run_name == '':
    print('The run_name argument should be given!')
    exit(0)

random.seed(43)


"""
The inverted index class. This file only contain a simple in-memory inverted index.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import logging
import pickle as pkl
import numpy as np
from params import FLAGS

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

class MemMappedInvertedIndex(object):

    def __init__(self, n_latent_terms):
        super(MemMappedInvertedIndex, self).__init__()
        self._n_latent_terms = n_latent_terms
        self._memmap_num_rows = FLAGS.batch_size_documents * FLAGS.num_document_batches
        self._memmap_num_cols = self._n_latent_terms

        index_storage_path_prefix = FLAGS.base_path + FLAGS.index_path + FLAGS.run_name
        self._filename_memmap_index = index_storage_path_prefix + '-memmap_docrepr_index'
        self._filename_latent_term_index = index_storage_path_prefix + '-latent_term_index'
        self._filename_doc_keymapping_index = index_storage_path_prefix + '-doc_keymapping_index'

        self._sequence_value = 0

    def create(self):
        self.index = dict()
        self._doc_id_to_memmap_idx = dict()
        # w+ Create or overwrite existing file for reading and writing.
        self._doc_repr_memmap = np.memmap(self._filename_memmap_index, dtype='float32', mode='w+', shape=(self._memmap_num_rows, self._memmap_num_cols))

    def add(self, doc_ids, doc_repr):
        for i in range(len(doc_ids)): # for each document indexed via i
            for j in range(len(doc_repr[i])): # for each doc_repr dimension for doc i
                if doc_repr[i][j] > 0.:
                    current_doc_id = doc_ids[i]
                    if j not in self.index:
                        self.index[j] = []
                    
                    if not current_doc_id in self._doc_id_to_memmap_idx:
                        memmap_index_for_doc = self._next_sequence_val()
                        self._doc_id_to_memmap_idx[current_doc_id] = memmap_index_for_doc
                        self._doc_repr_memmap[memmap_index_for_doc] = doc_repr[i]

                    self.index[j].append(current_doc_id)

    def store(self):
        del self._doc_repr_memmap # Deletion of memory mapped index flushes memory changes to disk before removing the object 
        
        pkl.dump(self.index, open(self._filename_latent_term_index, 'wb'))
        del self.index
        pkl.dump(self._doc_id_to_memmap_idx, open(self._filename_doc_keymapping_index, 'wb'))
        logging.info('Stored inverted index with {} documents in total'.format(str(len(self._doc_id_to_memmap_idx))))
        del self._doc_id_to_memmap_idx

    def load(self):
        self.index = pkl.load(open(self._filename_latent_term_index, 'rb'))
        self._doc_id_to_memmap_idx = pkl.load(open(self._filename_doc_keymapping_index, 'rb'))
        self._doc_repr_memmap = np.memmap(self._filename_memmap_index, dtype='float32', mode='r', shape=(self._memmap_num_rows, self._memmap_num_cols))

    def get_doc_representation(self, doc_id):
        memmmap_doc_repr_index = self._doc_id_to_memmap_idx[doc_id]
        doc_representation_v = self._doc_repr_memmap[memmmap_doc_repr_index]
        return doc_representation_v

    def _next_sequence_val(self):
        next_value = self._sequence_value
        self._sequence_value += 1
        return next_value
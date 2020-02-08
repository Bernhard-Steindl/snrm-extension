"""
The inverted index that is created using a memory mapped file.

Authors: Bernhard Steindl
"""
from app_logger import logger
logger = logger(__file__)

import params

import pickle as pkl
import numpy as np

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
            should_add_doc_to_index = False
            current_doc_id = doc_ids[i]
            current_doc_repr = doc_repr[i]

            for j in range(len(current_doc_repr)): # for each doc_repr dimension for doc i
                if current_doc_repr[j] > 0.:
                    should_add_doc_to_index = True
                    if j not in self.index:
                        self.index[j] = []
                    self.index[j].append(current_doc_id)

            if (should_add_doc_to_index == True) and (current_doc_id not in self._doc_id_to_memmap_idx):
                memmap_index_for_doc = self._next_sequence_val()
                self._doc_id_to_memmap_idx[current_doc_id] = memmap_index_for_doc
                self._doc_repr_memmap[memmap_index_for_doc] = current_doc_repr

    def store(self):
        logging.debug('found {} documents in index'.format(str(self.count_documents())))
        del self._doc_repr_memmap # Deletion of memory mapped index flushes memory changes to disk before removing the object 
        pkl.dump(self._doc_id_to_memmap_idx, open(self._filename_doc_keymapping_index, 'wb'))
        self._doc_id_to_memmap_idx
        self._doc_id_to_memmap_idx.clear()

        pkl.dump(self.index, open(self._filename_latent_term_index, 'wb'))
        self.index.clear()
        del self.index
        logging.info('Stored inverted index')
        

    def load(self):
        self.index = pkl.load(open(self._filename_latent_term_index, 'rb'))
        self._doc_id_to_memmap_idx = pkl.load(open(self._filename_doc_keymapping_index, 'rb'))
        self._doc_repr_memmap = np.memmap(self._filename_memmap_index, dtype='float32', mode='r', shape=(self._memmap_num_rows, self._memmap_num_cols))

    def get_doc_representation(self, doc_id):
        memmmap_doc_repr_index = self._doc_id_to_memmap_idx[doc_id]
        doc_representation_v = self._doc_repr_memmap[memmmap_doc_repr_index]
        return doc_representation_v

    def count_documents(self):
        return len(self._doc_id_to_memmap_idx)

    def _next_sequence_val(self):
        next_value = self._sequence_value
        self._sequence_value += 1
        return next_value
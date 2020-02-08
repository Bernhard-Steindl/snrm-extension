"""
A dictionary class that contains vocabulary terms and their IDs.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

from app_logger import logger
logger = logger(__file__)

from config import config
import params
import numpy as np

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # Resource punkt not found. Please use the NLTK Downloader to obtain the resource

from nltk.corpus import stopwords
nltk.download('stopwords') # Resource stopwords not found. Please use the NLTK Downloader to obtain the resource
stop_words = set(stopwords.words('english'))
# TODO should we exclude punctuation symbols
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 

class Dictionary(object):
    """
    This class contains a list of vocabulary terms and their mappings to term ids. The ids are zero indexed. The index
    zero is corresponding to 'UNKNOWN' terms.

    Attributes:
        id_to_term (list): a list of string containing vocabulary terms.
        term_to_id (dict): a dict of terms (str) to their ids (int).
    """

    def __init__(self):
        self.id_to_term = ['UNKNOWN']
        self.term_to_id = {'UNKNOWN': 0}

    def load_from_galago_dump(self, file_name):
        """
        loading vocabulary terms from the output of Galago's 'dump_term_stats' function. For more information, visit
        https://sourceforge.net/p/lemur/wiki/Galago%20Functions/.

        Args:
            file_name: The file address of the Galago's dump output.
        """
        id = 1
        with open(file_name) as f:
            for line in f:
                term = line.rstrip('\n')
                self.id_to_term.append(term)
                self.term_to_id[term] = id
                id += 1
        logger.info(str(id) + ' terms have been loaded to the dictionary')

    def size(self):
        return len(self.id_to_term)

    def get_emb_list(self, str, delimiter=''):
        words = str.strip().split(delimiter)
        return [(self.term_to_id[w] if w in self.term_to_id else self.term_to_id['UNKNOWN']) for w in words]

    def get_term_id_list(self, text):
        """
        Returns a list of term ids from the tokenized text;

        NLTK will be used for word tokenization
        :param text: query or document text as string
        :return: list of term ids (int) for each token in text
        """
        text_tokens = word_tokenize(text.lower())
        
        term_ids = [(self.term_to_id[t] if t in self.term_to_id
                else self.term_to_id['UNKNOWN'])
            for t in text_tokens if t not in stop_words]

        # logger.debug('text_tokens={}, \t term_ids={}'.format(repr(text_tokens), repr(term_ids)))

        # unknown_terms = set([t for t in text_tokens if t not in self.term_to_id])
        # logger.debug('unknown terms: {}'.format(repr(unknown_terms)))

        # num_non_zero =  np.count_nonzero(term_ids)
        # num_zero = len(text_tokens) - num_non_zero
        # ratio_known_tokens = num_non_zero / len(text_tokens)
        # logger.debug('word_tokenization of "{}" / len(text_tokens)={} / len(term_ids)={} / len(term_ids_without_unknown)={} / ratio_known_tokens={}'
        #    .format(text, str(len(text_tokens)), str(len(term_ids)), str(num_non_zero), str(ratio_known_tokens)))
        # logger.debug('ratio_known_tokens={}'.format(str( round(ratio_known_tokens, 4))))
        return term_ids

# just for test
if __name__ == '__main__':
    dictionary = Dictionary()
    dictionary.load_from_galago_dump(config.get('base_path') + config.get('dict_file_name'))

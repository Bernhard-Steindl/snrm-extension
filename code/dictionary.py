"""
A dictionary class that contains vocabulary terms and their IDs.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

from app_logger import logger
logger_instance = logger(__file__)

from config import config
#import params
#import numpy as np

#from nltk.tokenize import word_tokenize
#import nltk
#nltk.download('punkt')  # Resource punkt not found. Please use the NLTK Downloader to obtain the resource

#from nltk.corpus import stopwords
#nltk.download('stopwords') # Resource stopwords not found. Please use the NLTK Downloader to obtain the resource
#stop_words = set(stopwords.words('english'))
# TODO should we exclude punctuation symbols
#stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding

# TODO check todos in this file and then delete it

class Dictionary(object):
    """
    This class contains a list of vocabulary terms and their mappings to term ids. The ids are zero indexed. The index
    zero is corresponding to 'UNKNOWN' terms.

    Attributes:
        id_to_term (list): a list of string containing vocabulary terms.
        term_to_id (dict): a dict of terms (str) to their ids (int).
    """

    def __init__(self) -> None:
        self.id_to_term = ['UNKNOWN']
        self.term_to_id = {'UNKNOWN': 0}

    def load_from_galago_dump(self, file_name) -> None:
        """
        loading vocabulary terms from the output of Galago's 'dump_term_stats' function. For more information, visit
        https://sourceforge.net/p/lemur/wiki/Galago%20Functions/.

        Args:
            file_name: The file address of the Galago's dump output.
        """
        #id = 1
        #with open(file_name) as f:
            #for line in f:
             #   term = line.rstrip('\n')
              #  self.id_to_term.append(term)
                #self.term_to_id[term] = id
                #id += 1
        #logger.info(str(id) + ' terms have been loaded to the dictionary')

        # TODO should change padding_token='@@PADDING@@', oov_token='@@UNKNOWN@@'?
        # what is 'non_padded_namespaces.txt' why is it needed?
        #allennlp.modules.token_embedders.embedding.logger.setLevel(logging.INFO)


        self.vocabulary: 'Vocabulary' = Vocabulary.from_files(directory=config.get('vocab_dir_path'))
        self.embedding_layer = Embedding.from_params(vocab=self.vocabulary, params=Params({"pretrained_file": config.get('pre_trained_embedding_file_name'),
                                                                                           "embedding_dim": config.get('emb_dim'),
                                                                                           "trainable": config.get('embedding_trainable') == 'True',
                                                                                           "max_norm": None, 
                                                                                           "norm_type": 2.0,
                                                                                           "padding_index": 0}))

        # TODO padding index of embedding? should we add padding on padding_index=self.vocabulary.get_vocab_size()
        
        #self.embedding: 'Embedding' = Embedding.from_vocab_or_file(vocab=self.vocabulary, embedding_dim=config.get('emb_dim'), pretrained_file=config.get('pre_trained_embedding_file_name'), padding_index=self.vocabulary.get_vocab_size(), trainable= config.get('embedding_trainable'), max_norm=None, norm_type=2.0)

        # TODO we might use embedding norm, but which one?
        # OLD: see util.py fn load_word_embeddings
        #    norm = 1
        #    if normalize is True:
        #        norm = math.sqrt(sum(float(i) * float(i) for i in line[1: dim + 1]))
        #    we_matrix.append([float(i) / norm for i in line[1: dim + 1]])
        # 
        # Can be 2-Norm, but why is there no absolute value taken? 
        # https://de.wikipedia.org/wiki/P-Norm#Euklidische_Norm
        #
        # --> assumption: we use p=2 norm
        #
        # TODO but we do not use a max_norm in old code
        # 
        # pytorch only normalizes if max_norm is set.
        # https://github.com/pytorch/pytorch/blob/e2f12885140c36c1d5bf82de6eb47797856fdacd/torch/nn/functional.py#L1481 
        #
        # allennlp & pytorch: 
        # * max_norm : `float`, (optional, default=None)
        #       If given, will renormalize the embeddings to always have a norm lesser than this
        # * norm_type : `float`, (optional, default=2)
        #       The p of the p-norm to compute for the max_norm option
        # allennlp instantiates from torch.nn.functional.embedding with max_norm and norm_type  
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/embedding.py#L147
        # https://github.com/pytorch/pytorch/blob/e2f12885140c36c1d5bf82de6eb47797856fdacd/torch/nn/functional.py#L1490
        # https://github.com/pytorch/pytorch/blob/e2f12885140c36c1d5bf82de6eb47797856fdacd/torch/nn/functional.py#L1407
        # https://github.com/pytorch/pytorch/blob/b00345a6f2660e14cf5eed7d115675314ddf4d0c/aten/src/ATen/native/Embedding.cpp#L133
        #   auto row = self[sorted_indices[i]];
        #   auto norm = row.norm(norm_type).item<double>();
        #   if (norm > max_norm) {
        #       auto scale = max_norm / (norm + 1e-7);
        #       row *= scale;
        #   }
        # 
        # -> TODO now using no normalization with max_norm=None, norm_type=2.0


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

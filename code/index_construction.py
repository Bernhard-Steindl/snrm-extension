"""
Inverted index construction from the latent terms to document IDs from the representations learned by SNRM.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

from app_logger import logger
logger = logger(__file__)

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({})) # sets the seeds to be fixed

from config import config
import params

from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.data.iterators import BucketIterator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.tokenizers import WordTokenizer

from data_loading import IrTupleDatasetReader
from allennlp.data.tokenizers.word_filter import StopwordFilter

import torch

from snrm import SNRM
from inverted_index import MemMappedInvertedIndex

from allennlp.nn.util import move_to_device

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info('PyTorch uses device {}'.format(device))

# TODO extract to utils file, already used in train2 file
def count_zero(tensor: torch.Tensor) -> int:
    return (tensor == 0.0).sum().item()

# layer_size is a list containing the size of each layer. It can be set through the 'hiddein_x' arguments.
layer_size = [config.get('emb_dim')] # input layer
for i in [config.get('hidden_1'), config.get('hidden_2'), config.get('hidden_3'), config.get('hidden_4'), config.get('hidden_5')]:
    if i <= 0:
        break
    layer_size.append(i)

logger.info('Loading vocabulary')
vocabulary: Vocabulary = Vocabulary.from_files(directory=config.get('vocab_dir_path'))
logger.info('Loading embedding')
token_embedding: Embedding = Embedding.from_params(vocab=vocabulary, params=Params({"pretrained_file": config.get('pre_trained_embedding_file_name'),
                                                                              "embedding_dim": config.get('emb_dim'),
                                                                              "trainable": False, # TODO is this ok?
                                                                              "max_norm": None, 
                                                                              "norm_type": 2.0,
                                                                              "padding_index": 0}))
word_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
# The SNRM model.
model = SNRM(word_embeddings= word_embedder,
            batch_size=config.get('batch_size'),
            max_q_len=config.get('max_q_len'),
            max_doc_len=config.get('max_doc_len'),
            emb_dim=config.get('emb_dim'),
            layer_size=layer_size,
            dropout_parameter=config.get('dropout_probability'),
            regularization_term=config.get('regularization_term'),
            learning_rate=config.get('learning_rate'))

model_load_path = '{0}model-state_{1}.pt'.format(config.get('model_path'), config.get('run_name'))
logger.info('Restoring model parameters from "{}"'.format(model_load_path))
# restore model parameter https://pytorch.org/tutorials/beginner/saving_loading_models.html
model.load_state_dict(torch.load(model_load_path))
# if you saved model on GPU and now want to load it on a CPU:
# model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to(device)
model.eval() # set model in evaluation mode


# TODO should we also use torch.no_grad() ?

logger.info('Creating new Memory Mapped Index')
inverted_index = MemMappedInvertedIndex(layer_size[-1])
inverted_index.create()

logger.info('Initializing document tuple loader and iterator')
document_tuple_loader = IrTupleDatasetReader(lazy=True, 
                                             max_text_length=config.get('max_doc_len'),
                                             tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(),
                                                                       word_filter=StopwordFilter()))
                                             # already spacy tokenized, so that it is faster 
iterator = BucketIterator(batch_size=config.get('batch_size'),
                          sorting_keys=[("text_tokens", "num_tokens")])
iterator.index_with(vocabulary)

batch_num = 0
num_document_batches = config.get('num_document_batches')

# TODO exception handling file not found when iterating etc?

logger.info('Iterating over document collection file')
# if we reach end-of-file (EOF) no exception will be thrown, batch will have reduced size
for batch in Tqdm.tqdm(iterator(document_tuple_loader.read(config.get('document_collection_file')), num_epochs=1)):
    batch_num += 1
    batch = move_to_device(obj=batch, cuda_device=(0 if torch.cuda.is_available() else -1))
    doc_ids = batch['id']
    _, doc_repr, _ = model.forward(None, batch['text_tokens']['tokens'], None)

    inverted_index.add(doc_ids, doc_repr)

    if batch_num % 100 == 0:
        logger.info('generated document representation for batch_num={} of num_document_batches={}'.format(batch_num, num_document_batches))
        logger.info('index now holds {} documents '.format(str(inverted_index.count_documents())))
        zero_elements = count_zero(doc_repr)
        num_elements = doc_repr.numel()
        ratio_zero = (zero_elements / num_elements)
        logger.info('zero elements in doc_repr batch={}, total batch size={}'.format(zero_elements, num_elements))
        logger.info('generated doc_repr with ratio_zero_elements={:6.5f}'.format(ratio_zero))
        logger.info('added doc_ids = {}'.format(repr(doc_ids)))
    if batch_num == num_document_batches:
        break
    
logger.info('Ended iterating document batches last_batch_num={} of num_document_batches={}'.format(batch_num, num_document_batches))

# for i in range(len(doc_ids)):
#  logger.debug('adds doc_repr to index\tdoc_id={},\tdoc_repr=\n{}'.format(str(doc_ids[i]), repr(doc_repr[i])))
try:
    inverted_index.store()
except Exception as ex:
    logger.exception('Failed to store inverted index')
    print(ex)
# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py
# and: https://github.com/sebastian-hofstaetter/teaching/blob/master/advanced-information-retrieval/neural-ir-exercise/src/data_loading.py

from app_logger import logger
logger = logger(__file__)

from typing import Dict

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

class IrTripleDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                yield self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence)

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        query_tokenized = self._tokenizer.tokenize(query_sequence)
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]

        query_field = TextField(query_tokenized, self._token_indexers)
        
        doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence)
        if self.max_doc_length > -1:
            doc_pos_tokenized = doc_pos_tokenized[:self.max_doc_length]

        doc_pos_field = TextField(doc_pos_tokenized, self._token_indexers)

        doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence)
        if self.max_doc_length > -1:
            doc_neg_tokenized = doc_neg_tokenized[:self.max_doc_length]

        doc_neg_field = TextField(doc_neg_tokenized, self._token_indexers)

        return Instance({
            "query_tokens":query_field,
            "doc_pos_tokens":doc_pos_field,
            "doc_neg_tokens": doc_neg_field})


class IrLabeledTupleDatasetReader(DatasetReader):
    """
    Read a tsv file containing labeled tuple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        "query_id",
        "doc_id",
        "query_tokens",
        "doc_tokens"
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 4:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_id, doc_id, query_sequence, doc_sequence = line_parts
                yield self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence)

    @overrides
    def text_to_instance(self, query_id:str, doc_id:str, query_sequence: str, doc_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        query_id_field = LabelField(int(query_id), skip_indexing=True)
        doc_id_field = LabelField(int(doc_id), skip_indexing=True)

        query_tokenized = self._tokenizer.tokenize(query_sequence)
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]

        query_field = TextField(query_tokenized, self._token_indexers)
        
        doc_tokenized = self._tokenizer.tokenize(doc_sequence)
        if self.max_doc_length > -1:
            doc_tokenized = doc_tokenized[:self.max_doc_length]

        doc_field = TextField(doc_tokenized, self._token_indexers)

        return Instance({
            "query_id":query_id_field,
            "doc_id":doc_id_field,
            "query_tokens":query_field,
            "doc_tokens":doc_field})

class IrTupleDatasetReader(DatasetReader):
    """
    Read a tsv file containing tuple of document id and document text, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <id_string>\t<text_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        id: ``TextField`` and
        tokens: ``TextField``
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 max_text_length:int = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_text_length = max_text_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                id, text_tokens = line_parts
                yield self.text_to_instance(id, text_tokens)

    @overrides
    def text_to_instance(self, id: str, text_tokens: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        text_id_field = LabelField(int(id), skip_indexing=True)

        text_tokenized = self._tokenizer.tokenize(text_tokens)
        if self.max_text_length > -1:
            text_tokenized = text_tokenized[:self.max_text_length]
        text_field = TextField(text_tokenized, self._token_indexers)
        
        return Instance({
            "id":text_id_field,
            "text_tokens":text_field})
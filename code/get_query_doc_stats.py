import sys
import logging
import numpy as np

from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')  # Resource punkt not found. Please use the NLTK Downloader to obtain the resource

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


def _get_stats(token_count, filename):
    median = np.median(token_count)
    mean = np.mean(token_count)
    min_val = np.amin(token_count)
    max_val = np.amax(token_count)
    quantile_075 = np.quantile(token_count, 0.75)
    quantile_09 = np.quantile(token_count, 0.9)
    quantile_08 = np.quantile(token_count, 0.8)

    logging.info('token_count_stats file={} median={}, mean={}, min_value={}, max_value={}, 0.75-quantil={}, 0.8-quantil={}, 0.9-quantil={} \n\n'
        .format(filename, str(median), str(mean), str(min_val), str(max_val), str(quantile_075), str(quantile_08), str(quantile_09)))
    return median, mean, min_val, max_val

def main():
    """
  
    """

    token_count = []

    # generate new tokens file, which contains tokens of any positive occurence
    try:
        docs_file = 'data/document_collection/collection.tsv'
        query_file = 'data/training_data/triples.train.tsv'
        

        logging.info('Obtaining token count of doc_texts from file "{}"'.format(docs_file))
        count_line = 0
        with open(docs_file, 'r') as doc_col_file:
            for line in doc_col_file:
                # e.g. 0       doc_text
                line_components = line.rstrip('\n').split('\t') 
                doc_text = line_components[1]
                
                doc_text_tokens = word_tokenize(doc_text)
                len_doc_text_tokens = len(doc_text_tokens)
                token_count.append(len_doc_text_tokens)
                
                count_line += 1
                if count_line % 100000 == 0:
                    logging.info('processed {} lines'.format(count_line))
        _get_stats(token_count, docs_file)


        logging.info('Obtaining token count of query_texts from file "{}"'.format(query_file))
        token_count = []
        count_line = 0
        with open(query_file, 'r') as query_f:
            for line in query_f:
                # <query>        <passage_1>   <passage_2>
                line_components = line.rstrip('\n').split('\t') 
                query_text = line_components[0]
                
                query_text_tokens = word_tokenize(query_text)
                len_query_text_tokens = len(query_text_tokens)
                token_count.append(len_query_text_tokens)
                
                count_line += 1
                if count_line % 100000 == 0:
                    logging.info('processed {} lines'.format(count_line))
        _get_stats(token_count, query_file)

    except Exception as e:
        logging.error('Failure occured ', exc_info=True)
        exit()

if __name__ == '__main__':
    main();
import sys
import time
import logging

from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')  # Resource punkt not found. Please use the NLTK Downloader to obtain the resource

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


def find_tokens(docs_file):
    tokens = dict() # term -> num_occurence
    lowered_case_tokens = dict() # term -> num_occurence

    count_line = 0
    with open(docs_file, 'r') as qrels_file:
        for line in qrels_file:
            # e.g. 0       doc_text
            line_components = line.rstrip('\n').split('\t') 
            query_id = int(line_components[0])
            doc_text = line_components[1]
            
            doc_text_tokens = word_tokenize(doc_text)
            
            for token in doc_text_tokens:
                if token not in tokens:
                    tokens[token] = 1
                else:
                    tokens[token] += 1

                lower_case_token = token.lower()
                if lower_case_token not in lowered_case_tokens:
                    lowered_case_tokens[lower_case_token] = 1
                else:
                    lowered_case_tokens[lower_case_token] += 1
            
            count_line += 1
            if count_line % 10000 == 0:
                logging.info('processed {} lines'.format(count_line))

    logging.info('Found {} tokens and {} lowered case tokens'.format(str(len(tokens)), str(len(lowered_case_tokens))))
    return tokens, lowered_case_tokens


def main():
    """
  
    """

    if len(sys.argv) == 1:
        # generate new tokens file, which contains tokens of any positive occurence
        try:
            docs_file = 'data/document_collection/collection.tsv'
            current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
            result_file = 'data/tokens/tokens_{}'.format(current_timestamp_str)
            result_lowered_file = 'data/tokens/tokens_lowered_{}'.format(current_timestamp_str)

            tokens = dict() # term -> num_occurence
            lowered_case_tokens = dict() # term -> num_occurence

            logging.info('Obtaining tokens and corrensponding num_token_occurence from document_collection.')
            tokens, lowered_case_tokens = find_tokens(docs_file)

            # write tokens to file
            with open(result_file, 'w') as f:
                for (token, num_occurence) in tokens.items():
                    f.write('{0}\t{1}\n'.format(token, num_occurence))
            logging.info('Wrote tokens in result file "{}"'.format(result_file))
            
            # write lower case tokens to file
            with open(result_lowered_file, 'w') as f_lowered:
                for (token, num_occurence) in lowered_case_tokens.items():
                    f_lowered.write('{0}\t{1}\n'.format(token, num_occurence))
            logging.info('Wrote lowered_case_tokens in result file "{}"'.format(result_lowered_file))

        except Exception as e:
            logging.error('Failure occured while trying to obtain new tokens file from doc_collection', exc_info=True)
            exit()

    elif len(sys.argv) == 3:
        # reduce existing `tokens_file`, where only tokens are preserved which have a minimum occurence value `min_value`

        try:
            tokens_file = sys.argv[1]
            min_token_occurence = int(sys.argv[2])

            new_tokens_file = tokens_file + '_min_' + str(min_token_occurence) + '.txt'

            logging.info('Using existing tokens_file "{}"'.format(tokens_file))
            logging.info('Writing new tokens file "{}" with min_token_occurence={}'.format(new_tokens_file, min_token_occurence))

            count_lines = 0
            with open(tokens_file, 'r') as f:
                with open(new_tokens_file, 'w') as new_file:
                    for line in f:
                        token, num_occurence = line.rstrip('\n').split('\t')
                        num_occurence = int(num_occurence)

                        if num_occurence >= min_token_occurence:
                            count_lines += 1
                            new_file.write('{}\n'.format(token))

            logging.info('Finished writing {} lines to new tokens result file.'.format(count_lines))

        except Exception as e:
            logging.error('Failure occured while trying to process existing tokens file ', exc_info=True)
            exit()
        
    else:
        # invalid usage
        logging.error('Usage: generate_tokens.py [<tokens_file> <min_value>]')
        exit()


if __name__ == '__main__':
    main();
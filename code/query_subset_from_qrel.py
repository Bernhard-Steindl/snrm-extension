import sys
import time
import logging

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

def main():
    """
    Use for generating a subset of a query file using an existing qrel subset file and an original query file.

    You may use `qrels_subset.py` beforehand, to generate a qrel subset file.
    From `qrel_subset.py`:
        Outputs a qrels file that contains a subset of relevance assessments of the input qrels file `qrels_file_name`.
        Generated qrels file is written to `result_file`.
        In the result qrels file all queries are contained, that have at least one doc_id assigned, 
        whose doc_id value is >= `min_doc_id_value`

    Assume, you have a qrel subset file -- e.g. qrel_subset_ge7841823_2019-12-27_195358.tsv -- that holds all queries,
    that have at least one doc_id that is greater than 7841823, and their corresponding relevance assessment.

    Using this script `query_subset_from_qrel.py`, you can generate a new query file from the original and existing one,
    that is a subset of the original, and only contains those queries, that are contained in the qrel subset file.

    To clarify the example, this script can generate a new query file -- e.g. query_subset_ge7841823_2019-12-27_195358.tsv --
    that contains all (query_id, query_text) tuples of queries that are in the qrel_subset_ge7841823_2019-12-27_195358.tsv
    using an original and existing query file -- e.g. queries.dev.small.tsv. 

    """

    if len(sys.argv) == 3:

        qrel_subset_file_name = sys.argv[1] # e.g. qrel_subset_ge7841823_2019-12-27_195358.tsv
        query_file_name = sys.argv[2] # e.g. queries.dev.small.tsv

        logging.info('Using qrel subset file "{}"'.format(qrel_subset_file_name))
        logging.info('Using query file "{}"\n'.format(query_file_name))

        result_query_subset_file_name = qrel_subset_file_name.replace('qrel', 'query')
        logging.info('Asking for name of result query subset file name, proposal="{}"  \n'.format(result_query_subset_file_name))
        
        entered_query_subset_file_name = input('? Is "{}" ok for the result file, \nthen hit enter, or enter a different file name:    '
            .format(result_query_subset_file_name))

        clean_entered_query_subset_file_name = entered_query_subset_file_name.strip()
        if  clean_entered_query_subset_file_name != '':
            result_query_subset_file_name = clean_entered_query_subset_file_name

        logging.info('Using result query subset file "{}"\n'.format(result_query_subset_file_name))

        query_ids = set()
        
        logging.info('Finding all query_id entries from qrel subset file')
        with open(qrel_subset_file_name, 'r') as qrel_subset_file:
            for line in qrel_subset_file:
                # e.g. 789292	0	7842289	1
                line_components = line.rstrip('\n').split('\t')
                query_id = line_components[0]
                if query_id not in query_ids:
                    query_ids.add(query_id)

        logging.info('Found {} query_id entries in qrel subset file, that are now beeing searched in query file'.format(str(len(query_ids))))
        count_found_queries = 0
        with open(query_file_name, 'r') as query_file:
            with open(result_query_subset_file_name, 'w') as result_file:
                for line in query_file:
                    # e.g. 1048585 what is paula deen's brother
                    line_components = line.rstrip('\n').split('\t')
                    query_id = line_components[0]
                    query_text = line_components[1]

                    if query_id in query_ids:
                        count_found_queries += 1
                        result_file.write('{0}\t{1}\n'.format(query_id, query_text))
                        if count_found_queries == len(query_ids):
                            break
        if count_found_queries == len(query_ids):
            logging.info('Wrote all query id entries in result query subset file')
        else:
            logging.warning('Could only write {} of {} query id entries in result query subset file'.format(str(count_found_queries), str(len(query_ids))))

    else:
        print('Usage: query_subset_from_qrel.py <qrel_subset_file_name> <query_file_name>')
        exit()


    

if __name__ == '__main__':
    main();
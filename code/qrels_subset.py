import sys
import time
import logging

FORMAT = '%(asctime)-15s %(levelname)-10s %(filename)-10s %(funcName)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

def main():
    """
    Outputs a qrels file that contains a subset of relevance assessments of the input qrels file `qrels_file_name`.
    Generated qrels file is written to `result_file`.
    
    In the result qrels file all queries are contained, that have at least one doc_id assigned, 
    hose doc_id value is >= `min_doc_id_value`
    """
    qrels_file_name = 'data/evaluation/qrels.dev.small.tsv'
    current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
    query_ids = []

    min_doc_id_value = 7841823 # queries which have a doc_id greater or equal than `min_doc_id_value`
    result_file = 'data/evaluation/qrel_subset_ge{}_{}.tsv'.format(str(min_doc_id_value), current_timestamp_str)

    logging.info('Collecting all queries with at least 1 relevant doc_id, whose doc_id value is >= {}'.format(min_doc_id_value))

    # find all query ids that should be collected in the next step
    with open(qrels_file_name, 'r') as qrels_file:
        for line in qrels_file:
            line_components = line.rstrip('\n').split('\t') # e.g. 898686	0	7897243	1
            query_id = int(line_components[0])
            doc_id = int(line_components[2])
            
            if (doc_id >= min_doc_id_value) and (query_id not in query_ids):
                query_ids.append(query_id)

    logging.info('Found {} queries that have assigned at least 1 doc_id >= than {}'.format(str(len(query_ids)), str(min_doc_id_value)))
    
    logging.info('Writing a new qrels subset file to "{}"'.format(result_file))
    # write all collected query ids to result file, with all the corresponding relevant doc ids
    # writes not only doc ids that are greater than `min_doc_id_value`, 
    # but all relevant doc ids for a query that has at least one relevant doc id that is greater than `min_doc_id_value`.

    count_num_queries = 0
    with open(result_file, 'w') as f:
        with open(qrels_file_name, 'r') as qrels_file:
            for line in qrels_file:
                line_components = line.rstrip('\n').split('\t') # e.g. 898686	0	7897243	1
                query_id = int(line_components[0])
                doc_id = int(line_components[2])

                if query_id in query_ids:
                    f.write('{0}	0	{1}	1\n'.format(query_id, doc_id))
                    count_num_queries += 1
    logging.info('Wrote qrels result file with {} lines'.format(count_num_queries))

if __name__ == '__main__':
    main();
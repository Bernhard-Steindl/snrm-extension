import sys
import time

def load_qrel_file(file_name):
    """
    # TODO doc
    """
    qrel_assessment = dict()
    with open(file_name, 'r') as file:
        for line in file:
            try:
                line_components = line.rstrip('\n').split('\t')
                query_id = int(line_components[0])
                doc_id = int(line_components[2])
                if query_id in qrel_assessment:
                    pass
                else:
                    qrel_assessment[query_id] = []
                if not doc_id in qrel_assessment[query_id]:
                    qrel_assessment[query_id].append(doc_id)
            except:
                raise IOError('\"%s\" does not have a valid format qid\t0\doc_id\t1'.format(line))
    return qrel_assessment


def main():
    """Command line:
    python analyze_qrel.py <path_to_qrel_file>

    Example usage:
    (base) user@host snrm-extension % pwd
        /Users/user/snrm-extension
    (base) user@host snrm-extension % python evaluation-tools/analyze_qrel.py data/evaluation/qrels.dev.small.tsv
        analyze_qrel.py
        #####################
        path_to_qrel_file: data/evaluation/qrels.dev.small.tsv
        generated output file "results/analyze_qrel_num_docs_for_qid_2019-12-15_202124.txt"
        generated output file "results/analyze_qrel_distribution_docs_for_qid_2019-12-15_202124.txt"
        #####################
    """
    print('analyze_qrel.py')

    if len(sys.argv) == 2:
        path_to_qrel_file = sys.argv[1]
        qrel_assessment = load_qrel_file(path_to_qrel_file)
        print('#####################')
        print('path_to_qrel_file: {}'.format(path_to_qrel_file))
        base_path_part = 'results/analyze_qrel'
        current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
        number_of_docs_for_query_file = '{}_num_docs_for_qid_{}.txt'.format(base_path_part, current_timestamp_str)
        distribution_of_docs_for_query_file = '{}_distribution_docs_for_qid_{}.txt'.format(base_path_part, current_timestamp_str)

        with open(number_of_docs_for_query_file, 'w') as f:
            f.write('created at {} with analyze_qrel.py\n'.format(current_timestamp_str))
            f.write('path_to_qrel_file: {}\n'.format(path_to_qrel_file))
            f.write('numer_of_queries: {}\n'.format(len(qrel_assessment)))
            f.write('QUERY_ID\tNUM_DOCS_FOR_QUERY_ID\n\n')
            for query_id in sorted(qrel_assessment):
                # print('{}: {}'.format(query_id, len(qrel_assessment[query_id])))
                f.write('{}: {}\n'.format(query_id, len(qrel_assessment[query_id])))
            print('generated output file "{}"'.format(number_of_docs_for_query_file))

        with open(distribution_of_docs_for_query_file, 'w') as f:
            f.write('created at {} with analyze_qrel.py\n'.format(current_timestamp_str))
            f.write('path_to_qrel_file: {}\n'.format(path_to_qrel_file))
            f.write('numer_of_queries: {}\n'.format(len(qrel_assessment)))
            f.write('NUM_DOCS_FOR_QUERY\tAMOUNT_OF_QUERIES\n\n')

            distribution = dict()

            for query_id in sorted(qrel_assessment):
                num_docs_for_query = len(qrel_assessment[query_id])
                # print('{}: {}'.format(query_id, len(qrel_assessment[query_id])))
                if not num_docs_for_query in distribution:
                    distribution[num_docs_for_query] = 1
                else:
                    distribution[num_docs_for_query] += 1
            
            for num_docs_for_query in distribution:
                f.write('{}: {}\n'.format(num_docs_for_query, distribution[num_docs_for_query]))
            
            print('generated output file "{}"'.format(distribution_of_docs_for_query_file))

        print('#####################')
    else:
        print('Usage: analyze_qrel.py <path_to_qrel_file>')
        exit()
    
if __name__ == '__main__':
    main()
import sys
import time



def main():
    """Command line:
    python analyze_candidate.py <path_to_candidate_file>

    """
    print('analyze_candidate.py')

    if len(sys.argv) == 2:
        path_to_candidate_file = sys.argv[1]

        doc_ids = []
        query_ids = []
        
        with open(path_to_candidate_file, 'r') as file:
            for line in file:
                try:
                    line_components = line.rstrip('\n').split('\t')
                    # 1048585	Q0	49628	1	131.4897553360861	snrm-extension-example-run-local-1
                    query_id = int(line_components[0])
                    doc_id = int(line_components[2])
                    rank = int(line_components[3])
                    score = float(line_components[4])
                    run_name = line_components[5]
                    if doc_id not in doc_ids:
                        doc_ids.append(doc_id)
                    if query_id not in query_ids:
                        query_ids.append(query_id)
                except:
                    raise IOError('\"%s\" does not have a valid format'.format(line))

        print('#####################')
        print('path_to_candidate_file: {}'.format(path_to_candidate_file))


        base_path_part = 'results/analyze_candidate'
        current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
        doc_ids_in_candidate = '{}_doc_ids_in_candidate_{}.txt'.format(base_path_part, current_timestamp_str)

        with open(doc_ids_in_candidate, 'w') as f:
            f.write('created at {} with analyze_candidate.py\n'.format(current_timestamp_str))
            f.write('path_to_candidate_file: {}\n\n'.format(path_to_candidate_file))

            doc_ids = sorted(doc_ids)
            f.write('number of distinct query_id items: {}\n'.format(len(query_ids)))
            f.write('number of distinct doc_id items: {}\n'.format(len(doc_ids)))
            f.write('min doc_id: {}\n'.format(doc_ids[0]))
            f.write('max doc_id: {}\n\n\n'.format(doc_ids[len(doc_ids)-1]))



            for doc_id in doc_ids:
                f.write('{}\n'.format(doc_id))

        print('generated output file "{}"'.format(doc_ids_in_candidate))
        print('#####################')
    else:
        print('Usage: analyze_candidate.py <path_to_candidate_file>')
        exit()
    
if __name__ == '__main__':
    main()
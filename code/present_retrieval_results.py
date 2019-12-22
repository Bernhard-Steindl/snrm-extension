import sys
import time

def main():
    """
    Outputs a file for presenting the retrieval result using an retrieval evaluation candidate file.
    Result will be written in file name as set by variable $result_file
    
    Example output of result file:

        Query: 	 what is paula deen's brother
        * doc_id=5299 	 score=41564597.432536274 
        <document_text_of_doc_id>

        * doc_id=4674 	 score=40289889.34893748 
        <document_text_of_doc_id>

        * doc_id=5303 	 score=39828032.35023293 
        <document_text_of_doc_id>

        * doc_id=5308 	 score=39046091.92665547 
        <document_text_of_doc_id>

        ====================================
        Query: 	  Androgen receptor define
        ...
    """
    
    if len(sys.argv) == 2:

        query_collection = 'data/evaluation/queries.dev.small.tsv'
        document_collection = 'data/document_collection/collection.tsv'
        current_timestamp_str = time.strftime("%Y-%m-%d_%H%M%S")
        result_file = 'results/retrieval_results_presentation_{}'.format(current_timestamp_str)
        evaluation_candidate_file = sys.argv[1]

        max_documents_per_query = 4 # max number of documents per query to be presented
        max_queries = 2 # max number of queries to be presented

        query_and_rankings = dict()
        doc_to_text = dict() # holds doc_id -> doc_text

        queries_processed = 0
        with open(evaluation_candidate_file, 'r') as candidate_file:
            for line in candidate_file:
                line_components = line.rstrip('\n').split('\t')
                query_id = line_components[0]
                doc_id = line_components[2]
                rank =  line_components[3]
                score = line_components[4]
                if not query_id in query_and_rankings.keys():
                    queries_processed += 1
                    if queries_processed > max_queries:
                        break
                    query_and_rankings[query_id] = dict()
                    query_and_rankings[query_id]['docs'] = []
                    query_and_rankings[query_id]['text'] = ''

                if len(query_and_rankings[query_id]['docs']) < max_documents_per_query:
                    query_and_rankings[query_id]['docs'].append((doc_id, score))

        # fetch query text for all queries to be presented
        with open(query_collection, 'r') as query_collection_file:
            for qid in query_and_rankings.keys():
                query_collection_file.seek(0) # jump back to beginning of file

                for line in query_collection_file:
                    line_components = line.rstrip('\n').split('\t')
                    query_id = line_components[0]
                    query_text = line_components[1]
                    if query_id == qid:
                        query_and_ranking = query_and_rankings[qid]
                        query_and_ranking['text'] = query_text
                        break

        # fetch document_text for all relevant documents to be presented              
        with open(document_collection, 'r') as document_collection_file:
            # all unique doc_ids of relevant documents to be looked up in doc collection
            list_of_doc_id_sublists = [dict_per_query['docs'] for dict_per_query in query_and_rankings.values()]
            doc_ids = set([ doc_id for sublist in list_of_doc_id_sublists for (doc_id, score) in sublist])
            
            # for doc_id in doc_ids: 
                # print('doc id {}'.format(doc_id))

            num_docs_found_in_collection = 0

            for line in document_collection_file:
                line_components = line.rstrip('\n').split('\t')
                document_id = line_components[0]
                document_text = line_components[1]
                if document_id in doc_ids:
                    if not document_id in doc_to_text.keys():
                        doc_to_text[document_id] = document_text
                        num_docs_found_in_collection += 1
                        if num_docs_found_in_collection == len(doc_ids):
                            break;
            
        # write file
        with open(result_file, 'w') as f:
            f.write('query_collection: {}\n'.format(query_collection))
            f.write('document_collection: {}\n'.format(document_collection))
            f.write('evaluation_candidate_file: {}\n'.format(evaluation_candidate_file))
            f.write('result_file: {}\n'.format(result_file))

            for qid in query_and_rankings.keys():
                query_and_ranking = query_and_rankings[qid]
                f.write('====================================\n')
                f.write('Query: \t {}\n'.format(query_and_ranking['text']))

                for (doc_id, score) in query_and_ranking['docs']:
                    doc_text = doc_to_text[doc_id] if doc_id in doc_to_text else '************ DOC NOT FOUND *************'
                    f.write('* doc_id={0} \t score={1} \n{2}\n\n'.format(doc_id, score, doc_text))


        
      
    else:
        print('Usage: present_retrieval_results.py <path_to_retrieval_candidate_file>')
        exit()



if __name__ == '__main__':
    main();
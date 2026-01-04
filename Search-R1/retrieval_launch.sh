

index_file=/root/autodl-tmp/proj/index/bm25_wiki18
corpus_file=/root/autodl-tmp/proj/data/wiki-18.jsonl
retriever_name=bm25
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \


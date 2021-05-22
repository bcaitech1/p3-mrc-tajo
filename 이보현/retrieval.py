from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np

from datasets import Dataset
from rank_bm25 import BM25Plus
import re
from elasticsearch import Elasticsearch
from pororo import Pororo

import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

def preprocess(context):
    context = re.sub(r'\n', " ", context)
    context = re.sub(r"\\n", " ", context)
    context = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", " ", context)
    return context    

class SparseRetrieval:
    def __init__(self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.contexts =list(map(preprocess,self.contexts))
        self.ids = list(range(len(self.contexts)))

        self.BM25 = None
        self.tokenizer = tokenize_fn
        
        #Elastic search를 사용하기 위해 기본 세팅이 필요합니다. readme 참고후 주석 해제하세요.
        #self.es = Elasticsearch('localhost:9200')
    
    # BM25
    def get_embedding_BM25(self):

        pickle_name = f"BM25_embedding.bin"        
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.BM25 = pickle.load(file)            
            print("BM25 Embedding pickle load.")
        else:
            print("Build passage BM25_embedding")
            tokenized_contexts= [self.tokenizer(i) for i in self.contexts]
            self.BM25 = BM25Plus(tokenized_contexts)           
            with open(emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25 Embedding pickle saved.")
    
    def retrieve_BM25(self, query_or_dataset, topk=1):                
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_BM25(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_BM25(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="BM25 retrieval: ")):
                topK_context =''
                for i in range(len(doc_indices[idx])):                      
                     topK_context+=self.contexts[doc_indices[idx][i]]
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id, top_0 정보만 담고 있음. 따로 쓰이는데 없어서 그냥 둠 
                    "context": topK_context # retrieved doument, top_k context 이어 붙인 형태로 넘김. 어차피 max_seq_len으로 잘려짐.
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)           

            cqas = pd.DataFrame(total)

        return cqas
                   
    def get_relevant_doc_BM25(self, query, k=1):
        tokenized_query = self.tokenizer(query) 
        
        doc_scores = self.BM25.get_scores(tokenized_query)
        doc_indices = doc_scores.argmax()
        print(doc_scores, doc_indices)
        return doc_scores, doc_indices

    def get_relevant_doc_bulk_BM25(self, queries, k=10):

        pickle_score_name = f"BM25_score.bin"     
        pickle_indice_name = f"BM25_indice.bin"    
        score_path = os.path.join(self.data_path, pickle_score_name)      
        indice_path = os.path.join(self.data_path, pickle_indice_name)
        if os.path.isfile(score_path) and os.path.isfile(indice_path):
            with open(score_path, "rb") as file:
                doc_scores = pickle.load(file)  
            with open(indice_path, "rb") as file:
                doc_indices= pickle.load(file)            
            print("BM25 pickle load.")
        else:
            print("Build BM25 pickle")
            tokenized_queries= [self.tokenizer(i) for i in queries]        
            doc_scores = []
            doc_indices = []
            for i in tqdm(tokenized_queries):
                scores = self.BM25.get_scores(i)

                sorted_score = np.sort(scores)[::-1]
                sorted_id = np.argsort(scores)[::-1]
                max_nintypercent = sorted_score>sorted_score[0]*0.85             
            
                if len(sorted_score[max_nintypercent])<=k:
                    doc_scores.append(sorted_score[max_nintypercent])
                    doc_indices.append(sorted_id[max_nintypercent])
                else:
                    # 85퍼센트 score 넘는 passage가 k개 넘으면 자른다. 
                    doc_scores.append(sorted_score[:k])
                    doc_indices.append(sorted_id[:k])
            with open(score_path, "wb") as file:
                pickle.dump(doc_scores, file)
            with open(indice_path, "wb") as file:
                pickle.dump(doc_indices, file)
            print("BM25 pickle saved.")        

        return doc_scores, doc_indices

    # elastic search
    def retrieve_ES(self, query_or_dataset, topk=1):
        total = []
        with timer("query exhaustive search"):
            doc = self.get_relevant_doc_bulk_ES(query_or_dataset['question'], k=topk)
        for idx, example in enumerate(tqdm(query_or_dataset, desc="ES retrieval: ")):
            topK_context = ''
            for i in range(topk):
                topK_context+=doc[idx][i]['_source']['text_origin']                
            tmp = {
                "question": example["question"],
                "id": example['id'],
                "context_id": 0,  # 쓰이지 않음. 
                "context": topK_context # retrieved doument, top_k context 이어 붙인 형태로 넘김. 어차피 max_seq_len으로 잘려짐.
            }
            total.append(tmp)                      

        cqas = pd.DataFrame(total)

        return cqas
                   
    def get_relevant_doc_bulk_ES(self, queries, k=10):        
        ner = Pororo(task="ner", lang="ko")
        doc = []        
        for question in queries:
            query = {
                    'query':{
                        'bool':{
                            'must':[
                                    {'match':{'text':question}}
                            ],
                            'should':[
                                    {'match':{'text':' '.join([i[0] for i in ner(question) if i[1]!='O'])}}
                            ]
                        }
                    }
                }
            documents = self.es.search(index='document',body=query,size=k)['hits']['hits']
            doc.append(documents)

        return doc

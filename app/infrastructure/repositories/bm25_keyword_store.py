import os
import math
import pickle
import logging
from collections import Counter, defaultdict

from app.infrastructure.repositories.keyword_store_interface import KeywordStoreInterface
from app.infrastructure.repositories.tokenizer_interface import TokenizerInterface
from app.domain.models import Document, SearchResult
from app.core.config import FILE_CACHE_DIR, BM25_B, BM25_K1

logger = logging.getLogger(__name__)

class BM25KeywordStore(KeywordStoreInterface):
    def __init__(self, tokenizer:TokenizerInterface) -> None:
        self.tokenizer = tokenizer
        self.index: dict[str,set] = defaultdict(set) # token -> doc_id set
        self.docmap: dict[str,Document] = dict() # doc_id -> Document
        self.term_frequency: dict[str, Counter[str]] = defaultdict(Counter) # {doc_id : {token : cnt}}
        self.doc_lengths: dict[str,int] = dict() # doc_id -> length

        self.index_path:str  = os.path.join(FILE_CACHE_DIR, 'inverted_index')
        self.docmap_path:str = os.path.join(FILE_CACHE_DIR, 'docmap.pkl')
        self.term_frequency_path:str = os.path.join(FILE_CACHE_DIR, 'term_frequencies.pkl')
        self.doc_lengths_path:str    = os.path.join(FILE_CACHE_DIR, 'doc_lengths.pkl')

    @property
    def total_document(self) -> int:
        return len(self.docmap)
    
    def build_index(self, documents:list[Document]):
        for document in documents: # .id .content .metadata
            self.__add_document(document.id, document.content)
            self.docmap[document.id] = document
        logger.info(f"BM25KeywordStore build complete: {self.total_document} documents indexed.")

    def search(self, query:str, limit:int) -> list[SearchResult]:
        q_tokens = self.tokenizer.tokenize(query)
        doc_scores:dict[str,float] = defaultdict(float)
        for q_token in set(q_tokens):
            doc_ids = self.get_documents(q_token)
            for doc_id in doc_ids:
                doc_scores[doc_id] += self.bm25(doc_id, q_token, BM25_K1, BM25_B)
        top_doc_scores = sorted([(doc_id, score) for doc_id, score in doc_scores.items()], key=lambda x: x[1], reverse=True)[:limit]
        result = []
        for i, (doc_id, score) in enumerate(top_doc_scores):
            doc = self.docmap[doc_id]
            rank = i + 1
            result.append(SearchResult(document=doc, score=score, rank=rank))
        return result

    def retrieve_by_id(self, id:str) -> Document:
        return self.docmap[id]

    def get_documents(self, term:str) -> list[str]: 
        """Get doc_id list associated with input term"""
        doc_ids = list(self.index.get(term.lower(), set()))
        doc_ids.sort()
        return doc_ids
    
    def __add_document(self, doc_id:str, text:str):
        tokens = self.tokenizer.tokenize(text)
        # update inverted index and term frequencies
        for token in tokens: 
            self.index[token].add(doc_id)
            self.term_frequency[doc_id][token] += 1
        # update doc_lengths
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if self.total_document == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.total_document


    def _process_single_token_input(self, term:str) -> str:
        q_tokens = self.tokenizer.tokenize(term)
        if len(q_tokens) != 1:
            raise Exception(f"Expect single token but got {len(q_tokens)}: {q_tokens}")
        q_token = q_tokens[0]
        return q_token
    
    # -- tf-idf --
    def get_tf(self, doc_id:str, term:str) -> int:
        term_token = self._process_single_token_input(term)
        return self.term_frequency[doc_id][term_token]
    
    def get_idf(self, term:str) -> float:
        term_token = self._process_single_token_input(term)
        doc_count = self.total_document
        # print("doc_count",doc_count )
        term_doc_count = len(self.index[term_token])
        # print("term_doc_count", len(self.index[term_token]))
        return math.log( (doc_count+1) / (term_doc_count+1) )
    
    def get_tf_idf(self, doc_id:str, term:str) -> float:
        term_token = self._process_single_token_input(term)
        tf = self.get_tf(doc_id, term_token) 
        idf = self.get_idf(term_token)
        # print(f'-- tf:{tf} / idf:{idf}')
        return tf * idf
    
    # -- bm25 --
    def get_bm25_tf(self, doc_id:str, term:str, k1:float = BM25_K1, b:float = BM25_B) -> float:
        term_token = self._process_single_token_input(term)
        tf = self.get_tf(doc_id, term_token)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length() )
        return (tf * (k1 + 1)) / (tf + k1 * length_norm) 

    def get_bm25_idf(self, term:str) -> float:
        term_token = self._process_single_token_input(term)
        doc_count = self.total_document
        term_doc_count = len(self.index[term_token])
        return math.log( (doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def bm25(self, doc_id:str, term:str, k1:float = BM25_K1, b:float = BM25_B):
        term_token = self._process_single_token_input(term)
        return self.get_bm25_tf(doc_id, term_token, k1, b) * self.get_bm25_idf(term_token)

    # -- file I/O
    def save_index(self):
        os.makedirs(FILE_CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequency_path, 'wb') as f:
            pickle.dump(self.term_frequency, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    
    def load_index(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequency_path, 'rb') as f:
            self.term_frequency = pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)




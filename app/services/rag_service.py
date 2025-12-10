from infrastructure.clients.llm_client_interface import LLMClientInterface
from infrastructure.retriever import RetrieverInterface
from domain.models import Document


class RagService:
    def __init__(self, retriever:RetrieverInterface, llm_client:LLMClientInterface):
        self.retriever = retriever
        self.llm_client = llm_client
        self._is_built = False
    
    def build(self, documents:list[Document]):
        self.retriever.build(documents)
        self._is_built = True


    def ping_llm(self) -> dict:
        """Send a ping message through llm_client as a health check"""
        return self.llm_client.ping()       


    def answer(self, query: str, top_k:int = 5, mode='semantic'): 
        # TODO: search with conversation history
        if not self._is_built:
            raise RuntimeError("Service not built yet.")
        match mode:
            case "semantic": 
                result = self.retriever.semantic_search(query, top_k=top_k)
            case _:
                raise ValueError(f"Not supported mode: {mode}")
        
        
        context = "\n".join([f"{res.document.content}" for res in result])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        return self.llm_client.generate(prompt)
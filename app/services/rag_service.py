from anyio import to_thread
from textwrap import dedent
from app.infrastructure.clients.llm_client_interface import LLMClientInterface, AsyncLLMClientInterface
from app.infrastructure.retriever import RetrieverInterface
from app.domain.models import Document, SearchResult


class RagService:
    def __init__(self, retriever:RetrieverInterface, llm_client:AsyncLLMClientInterface):
        self.retriever = retriever
        self.llm_client = llm_client
        self._is_built = False
    
    def build(self, documents:list[Document]):
        self.retriever.build(documents)
        self._is_built = True

    def _search(self, query: str, top_k:int, method:str) -> list[SearchResult]:
        # TODO: search with conversation history
        if not self._is_built:
            raise RuntimeError("Service not built yet.")
        match method:
            case "semantic": 
                return self.retriever.semantic_search(query, top_k=top_k)
            case "keyword": 
                return self.retriever.keyword_search(query, top_k=top_k)
            case "hybrid": 
                return self.retriever.hybrid_search(query, top_k=top_k)
            case _:
                raise ValueError(f"Unsupported search method: {method}")

    async def search(self, query: str, top_k:int, method:str) -> dict[str,list[SearchResult]]:
        """ Pure search without LLM answer """
        search_result = await to_thread.run_sync(self._search, query, top_k, method)
        return {"search_result":search_result}

    async def hello_llm(self) -> dict:
        """Send "hello" to llm """
        response: dict[str,str] = await self.llm_client.hello()       
        return response

    async def answer(self, query: str, top_k:int = 5, method='semantic') -> dict: 
        """ Get LLM's answer for input query with rag result
        """
        search_result = await to_thread.run_sync(self._search, query, top_k, method)

        context = []
        for i, res in enumerate(search_result):
            context.append(f"{i+1}. | link = {res.document.metadata.get('permalink','')} | content:{res.document.content}"""
            )      
        context = "\n".join(context)


        prompt = dedent(f"""
        Answer the question or provide information based on the provided documents. 

        This should be tailored to answer users questions or queries of the site content.
        If the Question is irrelevant to the documents of the site, politely state that you are only able to answer questions related to the site content.
        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {context}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Provide recommended reading only from the documents if needed. 
        - Provide link to the recommended reading at the end in format of [link text](url)
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct, succinct, welcoming and informative

        Answer:
        """).strip()

        response: dict[str,str] = await self.llm_client.generate(prompt)
        return {"search_result": search_result, "answer":response.get("answer","")}
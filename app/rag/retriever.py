import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.integrations.llm_client import get_agent_model
from app.rag.vector_store import VectorStoreService
from app.utils.prompt_loader import load_rag_prompts

logger = logging.getLogger("rag-agent.retriever")


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_template = PromptTemplate.from_template(
            load_rag_prompts()
        )
        self.model = get_agent_model()
        self.chain = (
            self.prompt_template | self.model | StrOutputParser()
        )

    def retriever_docs(self, query: str) -> list:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += (
                f"[{i}] {doc.page_content} | meta: {doc.metadata}\n"
            )

        return self.chain.invoke({"input": query, "context": context})

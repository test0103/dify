from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings


class HuggingfaceEndpointEmbeddings(BaseModel, Embeddings):
    client: Any
    model: str

    task_type: Optional[str] = None
    huggingfacehub_api_type: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_query(self, text: str) -> List[float]:
        pass

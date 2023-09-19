from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.embeddings import HuggingFaceEmbeddings


class HuggingfaceHubEmbeddings(BaseModel, Embeddings):
    client: Any
    model: str

    task_type: Optional[str] = None
    huggingfacehub_api_type: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values['huggingfacehub_api_token'] = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )

        values['client'] = HuggingFaceEmbeddings(
            model_name=values['model']
        )

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)

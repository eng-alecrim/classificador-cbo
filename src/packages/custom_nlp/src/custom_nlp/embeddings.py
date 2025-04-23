# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Type, Union

import joblib
from common.utils import get_project_root
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .tratamento_texto import normaliza_str

# =============================================================================
# CONSTANTES
# =============================================================================

project_root_dir = get_project_root("classificador-cbo")
embedding_models_dir = project_root_dir / "models/embedding"
embedding_models_dir.mkdir(exist_ok=True, parents=True)

# =============================================================================
# CLASSES
# =============================================================================


class EmbeddingStrategy(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> Iterable[float] | spmatrix:
        pass


class CountVect(EmbeddingStrategy):
    def __init__(self, model: CountVectorizer):
        super().__init__()
        self.model = model

    def create_embedding(self, text: Union[str, Iterable[str]]) -> spmatrix:
        return (
            self.model.transform([text])
            if isinstance(text, str)
            else self.model.transform(text)
        )


class TfidfVect(EmbeddingStrategy):
    def __init__(self, model: TfidfVectorizer):
        super().__init__()
        self.model = model

    def create_embedding(self, text: Union[str, Iterable[str]]) -> spmatrix:
        return (
            self.model.transform([text])
            if isinstance(text, str)
            else self.model.transform(text)
        )


class Word2Vect(EmbeddingStrategy):
    def __init__(self, model: CountVectorizer):
        super().__init__()
        self.model = model

    def create_embedding(self, text: str) -> Iterable[float]: ...


class Embedder:
    def __init__(self, strategy: Type[EmbeddingStrategy] = None, model=None) -> None:
        self.strategy = strategy(model) if (strategy and model) else None

    def set_strategy(self, strategy: Type[EmbeddingStrategy], model) -> None:
        self.strategy = strategy(model)

    def apply(self, text: str) -> str:
        return self.strategy.create_embedding(text)


# =============================================================================
# FUNÇÕES
# =============================================================================


def save_embedding_model(model, model_name: Optional[str] = None) -> None:
    model_name = model_name if model_name else normaliza_str(model.__str__())
    with open(embedding_models_dir / f"{model_name}.joblib", "wb") as joblib_f:
        joblib.dump(value=model, filename=joblib_f)
    return None


def load_embedding_model(model_name: str):
    with open(embedding_models_dir / f"{model_name}.joblib", "rb") as joblib_f:
        return joblib.load(filename=joblib_f)

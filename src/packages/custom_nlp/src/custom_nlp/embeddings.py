# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from abc import ABC, abstractmethod
from typing import Iterable, Union, Type
from scipy.sparse import spmatrix

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
        self.strategy = strategy(model)

    def set_strategy(self, strategy: Type[EmbeddingStrategy], model) -> None:
        self.strategy = strategy(model)

    def apply(self, text: str) -> str:
        return self.strategy.create_embedding(text)

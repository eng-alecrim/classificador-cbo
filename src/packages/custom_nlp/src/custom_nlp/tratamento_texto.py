# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import nltk
import pandas as pd

# =============================================================================
# CLASSES
# =============================================================================


class TextProcessingStrategy(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass


class NormalizationStrategy(TextProcessingStrategy):
    def __init__(self, lower: bool = True) -> None:
        self.lower = lower

    def process(self, text: str) -> str:
        text = str(text)
        nfkd_form = unicodedata.normalize("NFC", text)
        output_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        regex_tags = r"</?.>"
        output_str = re.sub(regex_tags, "", output_str)
        regex = re.compile(r"[^a-zA-Z_À-ÿ\s]+")
        tokens = regex.sub(" ", output_str).split()
        if self.lower:
            tokens = list(map(lambda x: x.lower(), tokens))

        return " ".join(map(lambda x: x.strip(), tokens))


class StopwordsRemovalStrategy(TextProcessingStrategy):
    def __init__(self, language: str = "portuguese") -> None:
        super().__init__()
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        self.language = language
        self.stopwords = nltk.corpus.stopwords.words(language)

    def set_language(self, language: str) -> None:
        self.language = language
        self.stopwords = nltk.corpus.stopwords.words(language)
        return None

    def process(self, text: str) -> str:
        tokens = text.split()
        tokens_wo_stopwords = [token for token in tokens if token not in self.stopwords]
        return " ".join(tokens_wo_stopwords)


class Preprocessor:
    def __init__(self, strategy: Type[TextProcessingStrategy] = None, **kwargs) -> None:
        self.strategy = strategy(**kwargs) if strategy else None
        self.pipeline = []

    def set_strategy(self, strategy: Type[TextProcessingStrategy], **kwargs) -> None:
        self.strategy = strategy(**kwargs)

    def apply(self, text: str) -> str:
        return self.strategy.process(text)

    def set_pipeline(
        self, strategy_pipeline: List[Tuple[TextProcessingStrategy, Dict[str, Any]]]
    ):
        self.pipeline = []
        for strategy, strategy_kwargs in strategy_pipeline:
            (
                self.pipeline.append(strategy(**strategy_kwargs))
                if strategy_kwargs
                else self.pipeline.append(strategy())
            )

    def apply_strategy_pipeline(self, text: str) -> str:
        treated_text = text

        for strategy in self.pipeline:
            treated_text = strategy.process(treated_text)

        return treated_text


# =============================================================================
# FUNÇÕES
# =============================================================================


def normaliza_str(input_str: str, minuscula: bool = True, formato: str = "NFC") -> str:
    """Função que remove todos os caracteres especiais e acentos das letras, retornando uma str com apenas letras.
    :param input_str:
    :param minuscula:
    :param formato:
    """
    input_str = str(input_str)
    # Normalizando o texto conforme a forma NFC
    normalized_form = unicodedata.normalize(formato, input_str)
    output_str = "".join([c for c in normalized_form if not unicodedata.combining(c)])
    # Removendo as possíveis tags de HTML
    regex_tags = r"</?.>"
    output_str = re.sub(regex_tags, "", output_str)
    # Substituindo os caracteres especiais e números (tudo o que NÃO estiver de A-z)
    regex = re.compile(r"[^a-zA-Z_À-ÿ\s]+")
    tokens = regex.sub(" ", output_str).split()
    # Deixando em minúscula
    if minuscula:
        tokens = list(map(lambda x: x.lower(), tokens))
    # Removendo possíveis espaços em branco no início e/ou fim da str

    output_str = " ".join(map(lambda x: x.strip(), tokens))
    # Retorna algo apenas se o resultado NÃO for uma str vazia
    if output_str != "":
        return output_str
    raise ValueError("Texto de entrada inválido!")


def renomeia_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_map = dict(
        zip(df.columns, [normaliza_str(col, formato="NFKD") for col in df.columns])
    )
    return df.rename(columns=cols_map)

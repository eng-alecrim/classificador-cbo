# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import os
from collections.abc import Iterable
from pathlib import Path
from time import time
from typing import Any, Dict

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from common.logging import configure_logging
from common.utils import get_project_root
from core.mlops.mlflow_aux import set_local_mlflow_tracking_uri
from custom_nlp.tratamento_texto import (
    NormalizationStrategy,
    Preprocessor,
    StopwordsRemovalStrategy,
)
from dotenv import find_dotenv, load_dotenv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from loguru import logger
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# CONSTANTES
# =============================================================================

load_dotenv(find_dotenv())

# Logging
project_name = os.getenv("PROJECT_NAME", "classificador-cbo")
configure_logging(project_name=project_name, log_to_file=True, log_level="DEBUG")

# Diretórios
project_root_dir = get_project_root("classificador-cbo")
treated_data_dir = project_root_dir / "data/silver/cbo_sintese_perfil"
path_csv = treated_data_dir / "familia.csv"

# MLflow
mlflow_tracking_uri = set_local_mlflow_tracking_uri(project_name=project_name)


# =============================================================================
# CLASSES
# =============================================================================


class Doc2VecWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model_kwargs: Dict[str, Any]):
        self.vectorizer_ = Doc2Vec
        self.kwargs_ = model_kwargs

    def fit(self, X: Iterable[str], y: Iterable[str | int]):
        tokenized_text_and_tag = zip(
            [word.strip() for text in X for word in text.split()], y
        )
        tagged_data = [
            TaggedDocument(words=tokens, tags=[tag])
            for tokens, tag in tokenized_text_and_tag
        ]
        self.vectorizer_ = self.vectorizer_(documents=tagged_data, **self.kwargs_)
        return None

    def transform(self, X: Iterable[str]) -> np.ndarray:
        embeddings = np.array([
            self.vectorizer_.infer_vector([word.strip() for word in text.split()])
            for text in X
        ])
        return embeddings

    def fit_transform(self, X: Iterable[str], y: Iterable[str | int]) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: Iterable[str]) -> np.ndarray:
        return self.transform(X)


# =============================================================================
# FUNÇÕES
# =============================================================================


def save_custom_artifact(object: Any, object_name: str) -> None:
    object_path = Path(f"./{object_name}.joblib")
    with open(object_path, "wb") as joblib_f:
        joblib.dump(value=object, filename=joblib_f)
    mlflow.log_artifact(
        local_path=str(object_path),
    )
    object_path.unlink()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    logger.info("INÍCIO: 02_03_creating_doc2vec_model.py")
    logger.debug(f"MLflow tracking URI: {mlflow_tracking_uri}")

    # -----------------------------------------------------------------------------
    # Carregando os dados com os textos
    # -----------------------------------------------------------------------------

    cols_interesse = ["codigo", "titulo", "sintese"]
    df = pd.read_csv(
        path_csv, sep="\t", dtype=str, usecols=cols_interesse, index_col="codigo"
    ).dropna()
    logger.debug(f"Dados '{path_csv}' carregados.")

    # -----------------------------------------------------------------------------
    # Criando experimento MLflow
    # -----------------------------------------------------------------------------

    experiment_name = "CBO_Familias_Embedding_Doc2Vec"

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        logger.debug(f"Started MLflow run with ID: {run.info.run_id}")
        logger.debug(f"Run artifact URI: {mlflow.get_artifact_uri()}")

        # -----------------------------------------------------------------------------
        # Fazendo o pré-processamento textual
        # -----------------------------------------------------------------------------

        # Configurando o pré-processamento
        t0 = time()
        preprocessador = Preprocessor()

        pipeline_preprocessamento = [
            (NormalizationStrategy, None),
            (StopwordsRemovalStrategy, {"language": "portuguese"}),
        ]
        preprocessador.set_pipeline(strategy_pipeline=pipeline_preprocessamento)

        # Salvando o pré-processador para futuro uso
        save_custom_artifact(
            object=preprocessador,
            object_name="preprocessador_texto",
        )
        logger.info("Artefato 'preprocessador_texto' salvo.")

        # Aplicando o pré-processamento
        df["sintese_tratado"] = df.loc[:, "sintese"].apply(
            func=preprocessador.apply_strategy_pipeline
        )

        textos_normalizados_s_stopwords = df.loc[:, "sintese_tratado"].values

        mlflow.log_metric(
            key="preprocessing_time", value=time() - t0, run_id=run.info.run_id
        )

        # -----------------------------------------------------------------------------
        # Doc2Vec
        # -----------------------------------------------------------------------------

        t0 = time()
        doc2vec_kwargs = {
            "vector_size": 500,
            "min_count": 5,
            "window": 10,
            "workers": 4,
            "seed": 42,
            "epochs": 100,
        }
        doc2vec = Doc2VecWrapper(model_kwargs=doc2vec_kwargs)

        # Salvando os embeddings na coluna "embedding" do DataFrame
        df["doc2vec_embedding"] = list(
            doc2vec.fit_transform(X=textos_normalizados_s_stopwords, y=df.index.values)
        )
        mlflow.log_metric(
            key="doc2vec_fit_time", value=time() - t0, run_id=run.info.run_id
        )

        # Salvando o modelo
        mlflow.sklearn.log_model(
            doc2vec,
            artifact_path="embedding_model",
        )
        logger.info("Modelo 'doc2vec' salvo como 'embedding_model'.")

        # Registrando o modelo
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="doc2vec",
            tags={"type": "embedding"},
        )
        logger.info("Modelo 'doc2vec_vect' registrado no Model Registry.")

        # -----------------------------------------------------------------------------
        # Salvando o DataFrame
        # -----------------------------------------------------------------------------

        dataset: PandasDataset = mlflow.data.from_pandas(df)
        mlflow.log_input(dataset=dataset, context="training")
        save_custom_artifact(
            object=df,
            object_name="fitting_data",
        )
        logger.info("Artefato 'fitting_data' salvo.")

        logger.debug(f"End of MLflow run with ID: {run.info.run_id}")

    logger.info("FIM: 02_03_creating_doc2vec_model.py")

    return None


if __name__ == "__main__":
    main()

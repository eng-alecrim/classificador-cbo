# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import joblib
import pandas as pd
from common.utils import get_project_root
from custom_nlp.tratamento_texto import (
    NormalizationStrategy,
    Preprocessor,
    StopwordsRemovalStrategy,
)
from dotenv import find_dotenv, load_dotenv
import os
from common.logging import configure_logging
from loguru import logger

load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

# Logging
project_name = os.getenv("PROJECT_NAME", "classificador-cbo")
configure_logging(project_name=project_name, log_to_file=True, log_level="DEBUG")

# Diretórios
project_root_dir = get_project_root("classificador-cbo")
silver_layer = project_root_dir / "data/silver"

processed_data_dir = silver_layer / "processed"
processed_data_dir.mkdir(parents=True, exist_ok=True)

treated_data_dir = silver_layer / "treated"
treated_data_dir.mkdir(parents=True, exist_ok=True)

resources_dir = project_root_dir / "src/resources"
resources_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    # -----------------------------------------------------------------------------
    # Fazendo o pré-processamento textual
    # -----------------------------------------------------------------------------

    # Configurando o pré-processamento
    preprocessador = Preprocessor()

    pipeline_preprocessamento = [
        (NormalizationStrategy, None),
        (StopwordsRemovalStrategy, {"language": "portuguese"}),
    ]
    preprocessador.set_pipeline(strategy_pipeline=pipeline_preprocessamento)

    with open(resources_dir / "text_preprocessor.joblib", "wb") as joblib_f:
        joblib.dump(value=preprocessador, filename=joblib_f)
        logger.info("Artefato 'text_preprocessor' salvo.")

    # Carregando os dados
    treated_csv_files = processed_data_dir.rglob("*.csv")
    for csv_file in treated_csv_files:
        treated_data = pd.read_csv(csv_file, dtype=str, encoding="utf-8", sep="\t")

        # Aplicando o pré-processamento
        treated_data["sintese"] = treated_data.loc[:, "sintese"].apply(
            func=preprocessador.apply_strategy_pipeline
        )

        treated_data["perfilocupacional"] = treated_data.loc[
            :, "perfilocupacional"
        ].apply(func=preprocessador.apply_strategy_pipeline)

        treated_data_destine_path = treated_data_dir / f"{csv_file.name}"

        treated_data.to_csv(
            treated_data_destine_path, index=False, sep="\t", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import pandas as pd
from common.utils import get_project_root
from typing import Iterable
from custom_nlp.src.custom_nlp.tratamento_texto import renomeia_cols

# =============================================================================
# CONSTANTES
# =============================================================================

project_root_dir = get_project_root("classificador-cbo")
treated_data_dir = project_root_dir / f"data/silver/cbo_sintese_perfil"
treated_data_dir.mkdir(parents=True, exist_ok=True)

texto_para_remover = "VERSÃO PRELIMINAR (Esta versão será substituída após conclusão da revisão de perfis, conhecimentos, habilidades, atitudes e níveis)"

# =============================================================================
# FUNÇÕES
# =============================================================================


def agg_f(textos: Iterable[str]) -> str:
    textos_sem_versao_preliminar = set(
        map(lambda texto: texto.replace(texto_para_remover, "").strip(), textos)
    )
    return ". ".join(textos_sem_versao_preliminar)


def agg_textos(
    df: pd.DataFrame, agg_col: str, cols_with_text: list[str]
) -> pd.DataFrame:
    return (
        df.dropna(subset=cols_with_text)
        .groupby(by=agg_col, as_index=False)[cols_with_text]
        .agg(agg_f)
        .rename(columns={agg_col: "codigo"})
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    # Carregando os dados
    qbq_path = project_root_dir / "data/bronze/OcupacoesCBO.xlsx"
    cols_interesse = ["CodCBO", "Ocupação", "Síntese", "PerfilOcupacional"]
    df = pd.read_excel(
        qbq_path, sheet_name="Ocupação", usecols=cols_interesse, dtype=str
    )

    # Pegando os códigos
    df["grande_grupo"] = df.loc[:, "CodCBO"].apply(lambda codigo: codigo[:1])
    df["subgrupo_principal"] = df.loc[:, "CodCBO"].apply(lambda codigo: codigo[:2])
    df["subgrupo"] = df.loc[:, "CodCBO"].apply(lambda codigo: codigo[:3])
    df["familia"] = df.loc[:, "CodCBO"].apply(lambda codigo: codigo[:4])

    #
    classificacoes = [
        ("grande_grupo", "Grande Grupo"),
        ("subgrupo_principal", "SubGrupo Principal"),
        ("subgrupo", "SubGrupo"),
        ("familia", "Familia"),
    ]

    for classificacao in classificacoes:
        # Obtendo os textos
        textos = renomeia_cols(
            agg_textos(df, classificacao[0], ["Síntese", "PerfilOcupacional"])
        )
        path_csv_titulos = (
            project_root_dir / f"data/bronze/CBO2002 - {classificacao[1]}.csv"
        )

        titulos = renomeia_cols(
            pd.read_csv(path_csv_titulos, dtype=str, encoding="latin1", sep=";")
        )

        textos_e_titulos = pd.merge(left=textos, right=titulos, on="codigo", how="left")

        path_csv_destino = treated_data_dir / f"{classificacao[0]}.csv"
        textos_e_titulos.to_csv(
            path_csv_destino, index=False, sep="\t", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

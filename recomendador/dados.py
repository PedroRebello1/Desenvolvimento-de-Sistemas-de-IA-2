"""Carga, limpeza e preparação das bases de alunos, materiais e interações.

As interações (10 milhões de linhas, ~466 MB em CSV) são lidas em blocos com tipos
compactos e guardadas em Parquet na primeira execução; as seguintes leem o cache.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# Os perfis usam abreviações de forma inconsistente ("BD" e "Banco de Dados" aparecem
# para a mesma disciplina). Tudo é convertido para um nome canônico antes do uso.
CANONICO = {
    "bd": "Banco de Dados",
    "banco de dados": "Banco de Dados",
    "so": "Sistemas Operacionais",
    "ia": "Inteligência Artificial",
    "ml": "Machine Learning",
    "pm": "Gestão de Projetos",
    "qa": "Qualidade de Software",
    "arquitetura": "Arquitetura de Computadores",
    "si": "Sistemas de Informação",
}

COLUNAS_OBRA = ["titulo", "tipo", "area", "nivel", "descricao", "autor"]
TIPOS_INTERACAO = ["visualizacao", "leitura"]
_EPOCA = np.datetime64("1970-01-01", "D")


def normalizar_termo(termo: str) -> str:
    termo = " ".join(str(termo).split())
    return CANONICO.get(termo.lower(), termo)


def separar_lista(texto: str) -> list[str]:
    """'BD, IA, Web' -> ['Banco de Dados', 'Inteligência Artificial', 'Web']."""
    termos = [normalizar_termo(t) for t in str(texto).split(",") if t.strip()]
    return list(dict.fromkeys(termos))


def dia_de(data: str) -> int:
    """Converte 'AAAA-MM-DD' no número de dias desde 1970-01-01 (formato da coluna `dia`)."""
    return int((np.datetime64(data, "D") - _EPOCA).astype(int))


def data_de(dia) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(dia, unit="D")


def carregar_alunos(caminho=config.ARQ_ALUNOS) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8")
    df["curso"] = df["curso"].map(normalizar_termo)
    df["disciplinas"] = df["disciplinas_cursadas"].map(separar_lista)
    df["interesses"] = df["areas_interesse"].map(separar_lista)
    # Alunos com o mesmo curso, período, disciplinas e interesses compartilham um perfil.
    chave = df[["curso", "periodo", "disciplinas_cursadas", "areas_interesse"]].astype(str).agg("|".join, axis=1)
    df["id_perfil"] = pd.factorize(chave)[0].astype("int16")
    return df.set_index("id_aluno", drop=False).sort_index()


def carregar_materiais(caminho=config.ARQ_MATERIAIS) -> pd.DataFrame:
    df = pd.read_csv(caminho, encoding="utf-8")
    # Materiais com metadados idênticos são cópias da mesma obra.
    chave = df[COLUNAS_OBRA].astype(str).agg("|".join, axis=1)
    df["id_obra"] = pd.factorize(chave)[0].astype("int16")
    return df.set_index("id_material", drop=False).sort_index()


def _ler_csv_interacoes(caminho, tamanho_bloco: int = 2_000_000) -> pd.DataFrame:
    tipos = {"id_aluno": "int32", "id_material": "int32", "tipo_interacao": "category",
             "avaliacao": "int8", "duracao_minutos": "int16", "data": "str"}
    blocos = []
    for bloco in pd.read_csv(caminho, usecols=list(tipos), dtype=tipos, chunksize=tamanho_bloco):
        datas = pd.to_datetime(bloco.pop("data"), format="%Y-%m-%d").to_numpy().astype("datetime64[D]")
        bloco["dia"] = (datas - _EPOCA).astype("int32")
        bloco["tipo_interacao"] = pd.Categorical(bloco["tipo_interacao"].astype(str), categories=TIPOS_INTERACAO)
        blocos.append(bloco)
    return pd.concat(blocos, ignore_index=True)


def carregar_interacoes(caminho=config.ARQ_INTERACOES, cache=config.ARQ_INTERACOES_CACHE,
                        usar_cache: bool = True) -> pd.DataFrame:
    """Colunas: id_aluno, id_material, tipo_interacao, avaliacao, duracao_minutos, dia.

    `id_interacao` é descartado: no arquivo ele é apenas a sequência 1..N.
    """
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)
    df = _ler_csv_interacoes(caminho)
    if usar_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


@dataclass
class Base:
    """As três bases carregadas, com os índices internos usados pelos modelos.

    Os ids são inteiros contíguos a partir de 1, então o índice interno é `id - 1`.
    """
    alunos: pd.DataFrame
    materiais: pd.DataFrame
    interacoes: pd.DataFrame

    @property
    def n_usuarios(self) -> int:
        return int(max(self.alunos.index.max(), self.interacoes["id_aluno"].max() if len(self.interacoes) else 0))

    @property
    def n_itens(self) -> int:
        return int(max(self.materiais.index.max(), self.interacoes["id_material"].max() if len(self.interacoes) else 0))

    @property
    def ids_alunos_cadastrados(self) -> np.ndarray:
        return self.alunos.index.to_numpy()

    @property
    def ids_materiais_cadastrados(self) -> np.ndarray:
        return self.materiais.index.to_numpy()

    def mascara_cadastradas(self, interacoes: pd.DataFrame | None = None) -> np.ndarray:
        """True nas interações cujo aluno E material existem nos cadastros."""
        df = self.interacoes if interacoes is None else interacoes
        return (df["id_aluno"].isin(self.alunos.index) & df["id_material"].isin(self.materiais.index)).to_numpy()


def carregar_base(usar_cache: bool = True) -> Base:
    return Base(carregar_alunos(), carregar_materiais(), carregar_interacoes(usar_cache=usar_cache))


def dividir_por_data(interacoes: pd.DataFrame, data_corte: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divisão temporal: antes da data de corte = treino; a partir dela = teste."""
    antes = interacoes["dia"].to_numpy() < dia_de(data_corte)
    return interacoes[antes], interacoes[~antes]

"""Métricas online: registro de eventos e cálculo de CTR, engajamento e satisfação.

A demonstração (app.py) grava cada evento em `artefatos/eventos_online.csv`:

    momento, sessao, id_aluno, modelo, evento, id_material, posicao, valor

    impressao   — um material exibido na posição `posicao` da lista
    clique      — o aluno abriu o material
    tempo       — minutos de estudo informados ao concluir o material (`valor`)
    satisfacao  — nota da pesquisa de satisfação da sessão, de 1 a 5 (`valor`)

Cada sessão recebe um modelo (escolhido ou sorteado no modo teste A/B), então as
métricas abaixo comparam os modelos com usuários reais da demonstração.
"""
from __future__ import annotations

import csv
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COLUNAS = ["momento", "sessao", "id_aluno", "modelo", "evento", "id_material", "posicao", "valor"]
EVENTOS = ("impressao", "clique", "tempo", "satisfacao")


def nova_sessao() -> str:
    return uuid.uuid4().hex[:12]


class RegistroEventos:
    def __init__(self, caminho: Path = config.ARQ_EVENTOS_ONLINE):
        self.caminho = Path(caminho)

    def registrar(self, sessao: str, id_aluno: int, modelo: str, evento: str, id_material: int | None = None,
                  posicao: int | None = None, valor: float | None = None) -> None:
        if evento not in EVENTOS:
            raise ValueError(f"evento desconhecido: {evento}")
        novo = not self.caminho.exists()
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with self.caminho.open("a", newline="", encoding="utf-8") as arquivo:
            escritor = csv.writer(arquivo)
            if novo:
                escritor.writerow(COLUNAS)
            escritor.writerow([datetime.now().isoformat(timespec="seconds"), sessao, id_aluno, modelo, evento,
                               "" if id_material is None else id_material, "" if posicao is None else posicao,
                               "" if valor is None else valor])

    def registrar_impressoes(self, sessao: str, id_aluno: int, modelo: str, ids_materiais) -> None:
        for posicao, id_material in enumerate(ids_materiais, start=1):
            self.registrar(sessao, id_aluno, modelo, "impressao", int(id_material), posicao)

    def carregar(self) -> pd.DataFrame:
        if not self.caminho.exists():
            return pd.DataFrame(columns=COLUNAS)
        return pd.read_csv(self.caminho, encoding="utf-8")


def metricas_online(eventos: pd.DataFrame) -> pd.DataFrame:
    """Por modelo: sessões, impressões, cliques, CTR, minutos por sessão e satisfação média.

    CTR = cliques / impressões. Minutos por sessão consideram todas as sessões com
    impressões (sessão sem clique conta 0). Satisfação = média das respostas à pesquisa.
    """
    if eventos.empty:
        return pd.DataFrame(columns=["modelo", "sessoes", "impressoes", "cliques", "CTR", "minutos por sessão",
                                     "satisfação (1–5)", "respostas"])
    linhas = []
    for modelo, grupo in eventos.groupby("modelo"):
        impressoes = int((grupo["evento"] == "impressao").sum())
        cliques = int((grupo["evento"] == "clique").sum())
        sessoes = grupo.loc[grupo["evento"] == "impressao", "sessao"].nunique()
        minutos = grupo.loc[grupo["evento"] == "tempo", "valor"].astype(float).sum()
        satisfacao = grupo.loc[grupo["evento"] == "satisfacao", "valor"].astype(float)
        linhas.append({"modelo": modelo, "sessoes": sessoes, "impressoes": impressoes, "cliques": cliques,
                       "CTR": cliques / impressoes if impressoes else np.nan,
                       "minutos por sessão": minutos / sessoes if sessoes else np.nan,
                       "satisfação (1–5)": satisfacao.mean() if len(satisfacao) else np.nan,
                       "respostas": len(satisfacao)})
    return pd.DataFrame(linhas)

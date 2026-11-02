"""Modelos de referência: qualquer recomendador útil precisa superá-los."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import Recomendador, valores_por_id


class Aleatorio(Recomendador):
    nome = "Aleatório"

    def __init__(self, semente: int = config.SEMENTE):
        self.semente = semente

    def pontuar(self, ids_alunos, ids_materiais):
        rng = np.random.default_rng([self.semente, int(np.asarray(ids_alunos)[0])])
        return rng.random((len(ids_alunos), len(ids_materiais)), dtype=np.float32)


class Popularidade(Recomendador):
    """Ordena pelo número de avaliações positivas (nota >= limiar) no treino."""
    nome = "Popularidade"

    def __init__(self, n_itens: int, limiar: int = config.LIMIAR_RELEVANCIA):
        self.n_itens, self.limiar = n_itens, limiar
        self.contagem = np.zeros(n_itens + 1, dtype=np.float32)

    def treinar(self, treino: pd.DataFrame) -> "Popularidade":
        positivos = treino.loc[treino["avaliacao"].to_numpy() >= self.limiar, "id_material"].to_numpy()
        self.contagem = np.bincount(positivos, minlength=self.n_itens + 1).astype(np.float32)
        return self

    def pontuar(self, ids_alunos, ids_materiais):
        linha = valores_por_id(self.contagem, ids_materiais)
        return np.broadcast_to(linha, (len(ids_alunos), len(linha)))


class MediaGlobal(Recomendador):
    """Prevê sempre a média das notas de treino (referência para o RMSE)."""
    nome = "Média global"

    def __init__(self):
        self.media = 3.0

    def treinar(self, treino: pd.DataFrame) -> "MediaGlobal":
        self.media = float(treino["avaliacao"].mean())
        return self

    def pontuar(self, ids_alunos, ids_materiais):
        return np.zeros((len(ids_alunos), len(ids_materiais)), dtype=np.float32)

    def prever_avaliacao(self, ids_alunos, ids_materiais):
        return np.full(len(ids_alunos), self.media, dtype=np.float32)


class BaselineVieses(Recomendador):
    """r̂(u, i) = μ + b_u + b_i, com vieses regularizados (Koren, 2008).

    Estimado por mínimos quadrados alternados: o viés de cada material é a média
    dos resíduos encolhida por `reg_item`; o do aluno, idem com `reg_usuario`.
    Também serve de âncora para os resíduos usados pelos demais modelos.
    """
    nome = "Vieses (μ + b_u + b_i)"

    def __init__(self, n_usuarios: int, n_itens: int, reg_usuario: float = 10.0, reg_item: float = 10.0,
                 iteracoes: int = 3):
        self.reg_usuario, self.reg_item, self.iteracoes = reg_usuario, reg_item, iteracoes
        self.media = 3.0
        self.vies_usuario = np.zeros(n_usuarios + 1, dtype=np.float32)
        self.vies_item = np.zeros(n_itens + 1, dtype=np.float32)

    def treinar(self, treino: pd.DataFrame) -> "BaselineVieses":
        u = treino["id_aluno"].to_numpy()
        i = treino["id_material"].to_numpy()
        r = treino["avaliacao"].to_numpy().astype(np.float64)
        self.media = float(r.mean())
        n_u, n_i = len(self.vies_usuario), len(self.vies_item)
        cont_u = np.bincount(u, minlength=n_u)
        cont_i = np.bincount(i, minlength=n_i)
        bu = np.zeros(n_u)
        for _ in range(self.iteracoes):
            bi = np.bincount(i, weights=r - self.media - bu[u], minlength=n_i) / (cont_i + self.reg_item)
            bu = np.bincount(u, weights=r - self.media - bi[i], minlength=n_u) / (cont_u + self.reg_usuario)
        self.vies_usuario, self.vies_item = bu.astype(np.float32), bi.astype(np.float32)
        return self

    def pontuar(self, ids_alunos, ids_materiais):
        return (self.media + valores_por_id(self.vies_usuario, ids_alunos)[:, None]
                + valores_por_id(self.vies_item, ids_materiais)[None, :]).astype(np.float32)

    def prever_avaliacao(self, ids_alunos, ids_materiais):
        return (self.media + valores_por_id(self.vies_usuario, ids_alunos)
                + valores_por_id(self.vies_item, ids_materiais)).astype(np.float32)

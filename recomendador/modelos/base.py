"""Interface comum a todos os recomendadores."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def top_k(pontuacoes: np.ndarray, k: int) -> np.ndarray:
    """Índices das k maiores pontuações de cada linha, em ordem decrescente."""
    k = min(k, pontuacoes.shape[1])
    parte = np.argpartition(-pontuacoes, k - 1, axis=1)[:, :k]
    ordem = np.argsort(-np.take_along_axis(pontuacoes, parte, axis=1), axis=1, kind="stable")
    return np.take_along_axis(parte, ordem, axis=1)


def normalizar_linhas(matriz: np.ndarray) -> np.ndarray:
    """Min-max por aluno: leva as pontuações de cada linha para [0, 1] (linha constante -> 0)."""
    minimo = matriz.min(axis=1, keepdims=True)
    amplitude = matriz.max(axis=1, keepdims=True) - minimo
    return (matriz - minimo) / np.where(amplitude > 0, amplitude, 1)


def valores_por_id(vetor: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """`vetor[ids]`, devolvendo 0 para ids fora do intervalo conhecido pelo modelo."""
    ids = np.asarray(ids)
    dentro = ids < len(vetor)
    saida = np.zeros((len(ids),) + vetor.shape[1:], dtype=vetor.dtype)
    saida[dentro] = vetor[ids[dentro]]
    return saida


class Recomendador(ABC):
    nome = "Recomendador"

    def treinar(self, treino: pd.DataFrame) -> "Recomendador":
        return self

    @abstractmethod
    def pontuar(self, ids_alunos: np.ndarray, ids_materiais: np.ndarray) -> np.ndarray:
        """Matriz (alunos x materiais) de pontuações; maior = mais recomendado."""

    def prever_avaliacao(self, ids_alunos: np.ndarray, ids_materiais: np.ndarray) -> np.ndarray:
        """Nota prevista (1–5) para pares alinhados aluno/material."""
        raise NotImplementedError(f"{self.nome} não prevê avaliações")

    @staticmethod
    def _desempatar(pontuacoes: np.ndarray, vistos: np.ndarray | None, semente: int) -> np.ndarray:
        """Cópia das pontuações com os já vistos em −inf e os empates desfeitos.

        Empates são comuns (as 1.000 cópias de uma obra têm o mesmo conteúdo) e são
        desfeitos por um ruído relativo de 1e-5, sorteado com semente fixa.
        """
        pontuacoes = np.array(pontuacoes, dtype=np.float32)
        ruido = np.random.default_rng(semente).random(pontuacoes.shape, dtype=np.float32)
        pontuacoes += ruido * 1e-5 * (np.abs(pontuacoes) + 1e-3)
        if vistos is not None:
            pontuacoes[vistos] = -np.inf
        return pontuacoes

    def selecionar(self, pontuacoes: np.ndarray, ids_candidatos: np.ndarray, k: int,
                   vistos: np.ndarray | None = None, semente: int = 0) -> np.ndarray:
        """Ids dos k melhores candidatos de cada linha de `pontuacoes`."""
        return ids_candidatos[top_k(self._desempatar(pontuacoes, vistos, semente), k)]

    def recomendar(self, ids_alunos: np.ndarray, ids_candidatos: np.ndarray, k: int,
                   vistos: np.ndarray | None = None) -> np.ndarray:
        """Ids dos k materiais recomendados a cada aluno, sem repetir os já vistos."""
        ids_alunos = np.asarray(ids_alunos)
        pontuacoes = self.pontuar(ids_alunos, ids_candidatos)
        return self.selecionar(pontuacoes, ids_candidatos, k, vistos, semente=int(ids_alunos[0]))

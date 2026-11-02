"""Recomendador híbrido: conteúdo + colaborativo + conhecimento.

Combina as três fontes em quatro etapas:

1. Ponderação  — cada componente é normalizado para [0, 1] por aluno (min-max) e
                 somado com pesos (ajustados na validação).
2. Troca       — os pesos mudam conforme o que se sabe do aluno:
                 * histórico curto (< HISTORICO_MINIMO_FC): o peso colaborativo cai na proporção;
                 * sem perfil cadastrado: as regras não se aplicam (peso de conhecimento = 0);
                 * sem perfil e sem histórico: recorre à popularidade.
3. Restrição   — material cujos pré-requisitos o aluno não cumpre perde PENALIDADE_INELEGIVEL.
4. Diversidade — re-ranqueamento MMR (Carbonell & Goldstein, 1998): escolhe a cada passo o item que maximiza
                 λ·relevância − (1 − λ)·similaridade máxima com os já escolhidos.
                 Roda sobre todos os candidatos: com 1.000 cópias por obra, os 100 primeiros
                 costumam ser a mesma obra e um corte prévio impediria qualquer diversificação.

Para prever notas (RMSE) usa empilhamento linear: nota = baseline + β·(desvios previstos
por cada componente), com β estimado por mínimos quadrados na validação.
"""
from __future__ import annotations

import numpy as np

from .. import config
from .base import Recomendador, normalizar_linhas, top_k, valores_por_id
from .baselines import Popularidade
from .colaborativo import FiltragemColaborativa
from .conhecimento import ModeloConhecimento
from .conteudo import FiltragemConteudo

COMPONENTES = ("conteudo", "colaborativo", "conhecimento")


def mmr(relevancia: np.ndarray, vetores: np.ndarray, k: int, lambda_: float,
        disponivel: np.ndarray | None = None) -> list[int]:
    """Posições escolhidas pelo Maximal Marginal Relevance (relevância em [0, 1])."""
    disponivel = np.ones(len(relevancia), dtype=bool) if disponivel is None else disponivel.copy()
    similaridade_max = np.zeros(len(relevancia))
    escolhidos: list[int] = []
    for _ in range(min(k, int(disponivel.sum()))):
        valor = lambda_ * relevancia - (1 - lambda_) * similaridade_max
        valor[~disponivel] = -np.inf
        j = int(np.argmax(valor))
        escolhidos.append(j)
        disponivel[j] = False
        similaridade_max = np.maximum(similaridade_max, vetores @ vetores[j])
    return escolhidos


class Hibrido(Recomendador):
    nome = "Híbrido"

    def __init__(self, conteudo: FiltragemConteudo, colaborativo: FiltragemColaborativa,
                 conhecimento: ModeloConhecimento, popularidade: Popularidade | None = None,
                 pesos: dict[str, float] | None = None, lambda_mmr: float = config.LAMBDA_MMR,
                 penalidade_inelegivel: float = config.PENALIDADE_INELEGIVEL,
                 historico_minimo: int = config.HISTORICO_MINIMO_FC, candidatos_mmr: int | None = None):
        self.conteudo, self.colaborativo, self.conhecimento = conteudo, colaborativo, conhecimento
        self.popularidade = popularidade
        self.pesos = dict(pesos or config.PESOS_PADRAO)
        self.lambda_mmr, self.penalidade_inelegivel = lambda_mmr, penalidade_inelegivel
        self.historico_minimo, self.candidatos_mmr = historico_minimo, candidatos_mmr
        self.coef_notas: np.ndarray | None = None

    # ----- ranqueamento ---------------------------------------------------------
    def pesos_efetivos(self, ids_alunos) -> np.ndarray:
        """Pesos (alunos x 3) depois das regras de troca, somando 1 (ou 0 sem nenhum sinal)."""
        ids_alunos = np.asarray(ids_alunos)
        pesos = np.tile([self.pesos[c] for c in COMPONENTES], (len(ids_alunos), 1)).astype(np.float64)
        historico = valores_por_id(self.colaborativo.contagem_usuario, ids_alunos)
        pesos[:, 1] *= np.minimum(1.0, historico / self.historico_minimo)
        tem_perfil = self.conhecimento.possui_perfil(ids_alunos)
        pesos[~tem_perfil, 2] = 0
        sem_conteudo = ~tem_perfil & (valores_por_id(self.conteudo.n_historico, ids_alunos) == 0)
        pesos[sem_conteudo, 0] = 0
        soma = pesos.sum(axis=1, keepdims=True)
        return np.divide(pesos, soma, out=np.zeros_like(pesos), where=soma > 0)

    def componentes(self, ids_alunos, ids_materiais) -> dict[str, np.ndarray]:
        """Pontuação de cada componente, normalizada para [0, 1] em cada linha."""
        return {"conteudo": normalizar_linhas(self.conteudo.pontuar(ids_alunos, ids_materiais)),
                "colaborativo": normalizar_linhas(self.colaborativo.pontuar(ids_alunos, ids_materiais)),
                "conhecimento": normalizar_linhas(self.conhecimento.pontuar(ids_alunos, ids_materiais))}

    def combinar(self, componentes: dict[str, np.ndarray], pesos: np.ndarray, elegivel: np.ndarray,
                 ids_alunos, ids_materiais) -> np.ndarray:
        total = sum(pesos[:, [j]] * componentes[c] for j, c in enumerate(COMPONENTES))
        total = total - self.penalidade_inelegivel * (~elegivel)
        sem_sinal = pesos.sum(axis=1) == 0
        if self.popularidade is not None and sem_sinal.any():
            total[sem_sinal] = normalizar_linhas(self.popularidade.pontuar(ids_alunos[sem_sinal], ids_materiais))
        return total.astype(np.float32)

    def pontuar(self, ids_alunos, ids_materiais):
        ids_alunos = np.asarray(ids_alunos)
        return self.combinar(self.componentes(ids_alunos, ids_materiais), self.pesos_efetivos(ids_alunos),
                             self.conhecimento.elegibilidade(ids_alunos, ids_materiais), ids_alunos, ids_materiais)

    def selecionar(self, pontuacoes, ids_candidatos, k, vistos=None, semente=0):
        if self.lambda_mmr >= 1:
            return super().selecionar(pontuacoes, ids_candidatos, k, vistos, semente)
        pontuacoes = self._desempatar(pontuacoes, vistos, semente)
        if self.candidatos_mmr is not None and self.candidatos_mmr < pontuacoes.shape[1]:
            fora = np.ones(pontuacoes.shape, dtype=bool)
            np.put_along_axis(fora, top_k(pontuacoes, self.candidatos_mmr), False, axis=1)
            pontuacoes[fora] = -np.inf
        vetores = self.conteudo.vetores_itens[ids_candidatos]
        saida = np.empty((len(pontuacoes), k), dtype=ids_candidatos.dtype)
        for linha, relevancia in enumerate(pontuacoes.astype(np.float64)):
            disponivel = np.isfinite(relevancia)
            minimo, maximo = relevancia[disponivel].min(), relevancia[disponivel].max()
            relevancia = np.where(disponivel, (relevancia - minimo) / max(maximo - minimo, 1e-12), 0)
            escolhidos = mmr(relevancia, vetores, k, self.lambda_mmr, disponivel)
            if len(escolhidos) < k:   # menos candidatos disponíveis que k: completa com o restante
                escolhidos += [j for j in np.argsort(-relevancia) if j not in escolhidos][:k - len(escolhidos)]
            saida[linha] = ids_candidatos[escolhidos]
        return saida

    # ----- previsão de notas ----------------------------------------------------
    def _atributos_nota(self, ids_alunos, ids_materiais) -> tuple[np.ndarray, np.ndarray]:
        base = self.conteudo.baseline.prever_avaliacao(ids_alunos, ids_materiais).astype(np.float64)
        atributos = np.column_stack([
            self.colaborativo.prever_avaliacao(ids_alunos, ids_materiais) - base,
            self.conteudo.prever_avaliacao(ids_alunos, ids_materiais) - base,
            self.conhecimento.pontuar_pares(ids_alunos, ids_materiais),
            np.ones(len(base)),
        ])
        return atributos, base

    def ajustar_notas(self, ids_alunos, ids_materiais, avaliacoes) -> np.ndarray:
        """Estima os coeficientes do empilhamento (colaborativo, conteúdo, conhecimento, intercepto)."""
        atributos, base = self._atributos_nota(ids_alunos, ids_materiais)
        self.coef_notas = np.linalg.lstsq(atributos, np.asarray(avaliacoes, dtype=np.float64) - base, rcond=None)[0]
        return self.coef_notas

    def prever_avaliacao(self, ids_alunos, ids_materiais):
        if self.coef_notas is None:
            return self.colaborativo.prever_avaliacao(ids_alunos, ids_materiais)
        atributos, base = self._atributos_nota(ids_alunos, ids_materiais)
        return np.clip(base + atributos @ self.coef_notas, 1, 5).astype(np.float32)

    # ----- explicação (usada na demonstração) ------------------------------------
    def explicar(self, id_aluno: int, ids_materiais: np.ndarray) -> dict:
        """Peso efetivo e pontuação normalizada de cada componente para os materiais dados."""
        ids_alunos = np.array([id_aluno])
        pesos = self.pesos_efetivos(ids_alunos)[0]
        componentes = self.componentes(ids_alunos, ids_materiais)
        return {"pesos": dict(zip(COMPONENTES, pesos)),
                "componentes": {c: componentes[c][0] for c in COMPONENTES},
                "elegivel": self.conhecimento.elegibilidade(ids_alunos, ids_materiais)[0]}

"""Protocolo e métricas de avaliação offline.

Ranqueamento (por aluno, média entre alunos com ao menos um item relevante no teste):
    Precisão@K = acertos / K
    Recall@K   = acertos / itens relevantes do aluno
    F1@K       = 2·P·R / (P + R), calculado sobre as médias de P@K e R@K
    NDCG@K     = ganho acumulado descontado pela posição, normalizado pelo ideal
Também no nível de obra (as 1.000 cópias de uma obra contam como a mesma), com K = 3.

Catálogo:
    Cobertura   = fração dos materiais candidatos que aparecem em alguma lista
    Diversidade = média, por lista, de (1 − similaridade de conteúdo) entre pares de itens

Notas: RMSE e MAE sobre os pares (aluno, material) do teste.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .modelos import Recomendador


@dataclass
class Protocolo:
    ids_alunos: np.ndarray            # alunos avaliados (têm >= 1 material relevante no teste)
    ids_candidatos: np.ndarray        # materiais que podem ser recomendados
    vistos: sp.csr_matrix             # (alunos x candidatos): já interagiu no treino
    relevantes: list[np.ndarray]      # materiais relevantes no teste, por aluno
    obras_relevantes: list[np.ndarray]


def montar_protocolo(treino: pd.DataFrame, teste: pd.DataFrame, ids_alunos: np.ndarray,
                     ids_candidatos: np.ndarray, obra_do_material: np.ndarray,
                     limiar: int = config.LIMIAR_RELEVANCIA) -> Protocolo:
    ids_candidatos = np.sort(np.asarray(ids_candidatos))
    relevante = (teste["avaliacao"].to_numpy() >= limiar) & teste["id_material"].isin(ids_candidatos).to_numpy() \
        & teste["id_aluno"].isin(ids_alunos).to_numpy()
    grupos = teste.loc[relevante].groupby("id_aluno")["id_material"].unique()
    alunos = grupos.index.to_numpy()
    relevantes = [np.asarray(v) for v in grupos.to_numpy()]
    obras = [np.unique(obra_do_material[v]) for v in relevantes]

    historico = treino[treino["id_aluno"].isin(alunos).to_numpy() & treino["id_material"].isin(ids_candidatos).to_numpy()]
    linhas = np.searchsorted(alunos, historico["id_aluno"].to_numpy())
    colunas = np.searchsorted(ids_candidatos, historico["id_material"].to_numpy())
    vistos = sp.csr_matrix((np.ones(len(linhas), dtype=np.float32), (linhas, colunas)),
                           shape=(len(alunos), len(ids_candidatos))).astype(bool)
    return Protocolo(alunos, ids_candidatos, vistos, relevantes, obras)


def gerar_listas(modelo: Recomendador, protocolo: Protocolo, k: int, obra_do_material: np.ndarray,
                 lote: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Top-k de materiais (sem os já vistos) e o ranking completo de obras de cada aluno.

    A pontuação de uma obra é a maior entre as de suas cópias; obras já vistas não
    são excluídas, porque no nível de obra rever o mesmo assunto é o comportamento comum.
    """
    obra_candidato = obra_do_material[protocolo.ids_candidatos]
    n_obras = int(obra_candidato.max()) + 1
    listas = np.empty((len(protocolo.ids_alunos), k), dtype=np.int64)
    obras = np.empty((len(protocolo.ids_alunos), n_obras), dtype=np.int64)
    rng = np.random.default_rng(config.SEMENTE)
    for inicio in range(0, len(protocolo.ids_alunos), lote):
        ids = protocolo.ids_alunos[inicio:inicio + lote]
        pontuacoes = modelo.pontuar(ids, protocolo.ids_candidatos)
        vistos = protocolo.vistos[inicio:inicio + lote].toarray()
        listas[inicio:inicio + lote] = modelo.selecionar(pontuacoes, protocolo.ids_candidatos, k, vistos,
                                                         semente=int(ids[0]))
        por_obra = np.column_stack([pontuacoes[:, obra_candidato == w].max(axis=1) for w in range(n_obras)])
        por_obra = por_obra + rng.random(por_obra.shape) * 1e-6 * (np.abs(por_obra) + 1e-3)
        obras[inicio:inicio + lote] = np.argsort(-por_obra, axis=1)
    return listas, obras


def acertos(listas: np.ndarray, relevantes: list[np.ndarray]) -> np.ndarray:
    """Matriz booleana (alunos x posições): o item recomendado está entre os relevantes?"""
    return np.array([np.isin(lista, rel) for lista, rel in zip(listas, relevantes)], dtype=bool)


def precisao_recall_ndcg(matriz_acertos: np.ndarray, n_relevantes: np.ndarray, k: int):
    """Vetores por aluno de Precisão@k, Recall@k e NDCG@k."""
    hits = matriz_acertos[:, :k]
    n_acertos = hits.sum(axis=1)
    desconto = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * desconto).sum(axis=1)
    ideal = np.cumsum(desconto)[np.minimum(n_relevantes, k) - 1]
    return n_acertos / k, n_acertos / n_relevantes, dcg / ideal


def f1(precisao: float, recall: float) -> float:
    return 0.0 if precisao + recall == 0 else 2 * precisao * recall / (precisao + recall)


def cobertura(listas: np.ndarray, n_candidatos: int) -> float:
    return len(np.unique(listas)) / n_candidatos


def diversidade_intra_lista(listas: np.ndarray, vetores_itens: np.ndarray) -> float:
    """Média de (1 − cosseno de conteúdo) entre os pares de itens de cada lista."""
    k = listas.shape[1]
    if k < 2:
        return 0.0
    vetores = vetores_itens[listas]                              # (alunos, k, d)
    similaridade = np.einsum("akd,ajd->akj", vetores, vetores)
    fora_diagonal = ~np.eye(k, dtype=bool)
    return float((1 - similaridade[:, fora_diagonal]).mean())


def intervalo_confianca(valores: np.ndarray, reamostras: int = 1000, semente: int = 0) -> tuple[float, float]:
    """Intervalo de 95% da média por bootstrap sobre alunos."""
    rng = np.random.default_rng(semente)
    medias = np.array([valores[rng.integers(0, len(valores), len(valores))].mean() for _ in range(reamostras)])
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def avaliar_ranqueamento(modelo: Recomendador, protocolo: Protocolo, vetores_itens: np.ndarray,
                         obra_do_material: np.ndarray, lista_k=config.LISTA_K, k_obras: int = 3) -> dict:
    """Todas as métricas de ranqueamento de um modelo; inclui os vetores por aluno para testes."""
    k_max = max(lista_k)
    listas, obras = gerar_listas(modelo, protocolo, k_max, obra_do_material)
    n_relevantes = np.array([len(r) for r in protocolo.relevantes])
    matriz = acertos(listas, protocolo.relevantes)
    resultado = {"modelo": modelo.nome, "n_alunos": len(protocolo.ids_alunos)}
    por_aluno = {}
    for k in lista_k:
        p, r, ndcg = precisao_recall_ndcg(matriz, n_relevantes, k)
        resultado |= {f"P@{k}": p.mean(), f"R@{k}": r.mean(), f"F1@{k}": f1(p.mean(), r.mean()),
                      f"NDCG@{k}": ndcg.mean(), f"Cobertura@{k}": cobertura(listas[:, :k], len(protocolo.ids_candidatos)),
                      f"Diversidade@{k}": diversidade_intra_lista(listas[:, :k], vetores_itens)}
        por_aluno |= {f"P@{k}": p, f"R@{k}": r, f"NDCG@{k}": ndcg}
    n_obras_rel = np.array([len(o) for o in protocolo.obras_relevantes])
    p, r, ndcg = precisao_recall_ndcg(acertos(obras, protocolo.obras_relevantes), n_obras_rel, k_obras)
    resultado |= {f"Obra P@{k_obras}": p.mean(), f"Obra R@{k_obras}": r.mean(), f"Obra NDCG@{k_obras}": ndcg.mean()}
    por_aluno |= {f"Obra P@{k_obras}": p, f"Obra NDCG@{k_obras}": ndcg}
    k = config.K_PRINCIPAL
    for chave in (f"P@{k}", f"NDCG@{k}", f"Obra P@{k_obras}"):
        resultado[f"{chave} IC95"] = intervalo_confianca(por_aluno[chave])
    resultado["_por_aluno"] = por_aluno
    resultado["_listas"] = listas
    return resultado


def avaliar_notas(modelo: Recomendador, pares: pd.DataFrame) -> dict:
    previstas = modelo.prever_avaliacao(pares["id_aluno"].to_numpy(), pares["id_material"].to_numpy())
    erro = pares["avaliacao"].to_numpy().astype(np.float64) - previstas
    return {"modelo": modelo.nome, "n_pares": len(pares), "RMSE": float(np.sqrt(np.mean(erro ** 2))),
            "MAE": float(np.mean(np.abs(erro)))}

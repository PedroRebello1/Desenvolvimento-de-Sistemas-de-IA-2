import numpy as np
import pandas as pd
import pytest

from recomendador.avaliacao import (acertos, cobertura, diversidade_intra_lista, f1, intervalo_confianca,
                                    precisao_recall_ndcg)
from recomendador.online import metricas_online


def test_precisao_recall_ndcg_caso_conhecido():
    listas = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    relevantes = [np.array([2, 9]), np.array([5])]
    p, r, ndcg = precisao_recall_ndcg(acertos(listas, relevantes), np.array([2, 1]), k=4)
    assert p.tolist() == [0.25, 0.25]
    assert r.tolist() == [0.5, 1.0]
    # aluno 1: acerto na posição 2 -> DCG = 1/log2(3); ideal com 2 relevantes = 1 + 1/log2(3)
    assert ndcg[0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert ndcg[1] == pytest.approx(1.0)


def test_f1_e_casos_degenerados():
    assert f1(0.5, 0.5) == pytest.approx(0.5)
    assert f1(0.0, 0.0) == 0.0


def test_cobertura_conta_itens_distintos():
    assert cobertura(np.array([[1, 2], [2, 3]]), n_candidatos=10) == pytest.approx(0.3)


def test_diversidade_zero_para_copias_e_um_para_ortogonais():
    vetores = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert diversidade_intra_lista(np.array([[1, 2]]), vetores) == pytest.approx(0.0)
    assert diversidade_intra_lista(np.array([[1, 3]]), vetores) == pytest.approx(1.0)


def test_intervalo_confianca_contem_a_media():
    valores = np.random.default_rng(0).normal(2.0, 1.0, 2000)
    inferior, superior = intervalo_confianca(valores)
    assert inferior < valores.mean() < superior
    assert superior - inferior < 0.2


def test_metricas_online():
    eventos = pd.DataFrame([
        {"sessao": "s1", "id_aluno": 1, "modelo": "Híbrido", "evento": "impressao", "id_material": 10, "posicao": 1, "valor": None},
        {"sessao": "s1", "id_aluno": 1, "modelo": "Híbrido", "evento": "impressao", "id_material": 11, "posicao": 2, "valor": None},
        {"sessao": "s1", "id_aluno": 1, "modelo": "Híbrido", "evento": "clique", "id_material": 10, "posicao": 1, "valor": None},
        {"sessao": "s1", "id_aluno": 1, "modelo": "Híbrido", "evento": "tempo", "id_material": 10, "posicao": 1, "valor": 30},
        {"sessao": "s1", "id_aluno": 1, "modelo": "Híbrido", "evento": "satisfacao", "id_material": None, "posicao": None, "valor": 4},
    ])
    linha = metricas_online(eventos).iloc[0]
    assert linha["CTR"] == pytest.approx(0.5)
    assert linha["minutos por sessão"] == pytest.approx(30)
    assert linha["satisfação (1–5)"] == pytest.approx(4)

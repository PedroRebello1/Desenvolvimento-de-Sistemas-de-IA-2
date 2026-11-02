import numpy as np
import pandas as pd

from recomendador.modelos import BaselineVieses, FiltragemColaborativa, mmr, normalizar_linhas, top_k


def test_top_k_ordena_decrescente():
    pontuacoes = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(pontuacoes, 3).tolist() == [[1, 3, 2]]


def test_normalizar_linhas_vai_para_zero_um_e_trata_linha_constante():
    saida = normalizar_linhas(np.array([[2.0, 4.0, 3.0], [1.0, 1.0, 1.0]]))
    assert saida[0].tolist() == [0.0, 1.0, 0.5]
    assert saida[1].tolist() == [0.0, 0.0, 0.0]


def test_mmr_com_lambda_um_segue_a_relevancia():
    relevancia = np.array([0.2, 1.0, 0.6, 0.9])
    vetores = np.eye(4)
    assert mmr(relevancia, vetores, 3, lambda_=1.0) == [1, 3, 2]


def test_mmr_evita_copias_quando_lambda_menor_que_um():
    # itens 0 e 1 são cópias (mesmo vetor); o item 2 é diferente e um pouco menos relevante
    relevancia = np.array([1.0, 0.95, 0.7])
    vetores = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert mmr(relevancia, vetores, 2, lambda_=0.7) == [0, 2]


def _avaliacoes_dois_grupos(n_alunos=40, n_itens=20, semente=0):
    """Alunos pares gostam dos itens 1–10; ímpares, dos itens 11–20."""
    rng = np.random.default_rng(semente)
    linhas = []
    for aluno in range(1, n_alunos + 1):
        for item in rng.choice(np.arange(1, n_itens + 1), 12, replace=False):
            gosta = (aluno % 2 == 0) == (item <= n_itens // 2)
            linhas.append((aluno, item, 5 if gosta else 1))
    return pd.DataFrame(linhas, columns=["id_aluno", "id_material", "avaliacao"])


def test_baseline_vieses_captura_material_bem_avaliado():
    treino = pd.DataFrame({"id_aluno": [1, 2, 3, 1, 2, 3], "id_material": [1, 1, 1, 2, 2, 2],
                           "avaliacao": [5, 5, 5, 1, 1, 1]})
    modelo = BaselineVieses(n_usuarios=3, n_itens=2, reg_item=0.1, reg_usuario=0.1).treinar(treino)
    assert modelo.vies_item[1] > 0 > modelo.vies_item[2]


def test_filtragem_colaborativa_aprende_grupos_de_gosto():
    treino = _avaliacoes_dois_grupos()
    modelo = FiltragemColaborativa(n_usuarios=40, n_itens=20, fatores=4, iteracoes=20, reg=0.1, reg_vies=0.1).treinar(treino)
    pontuacoes = modelo.pontuar(np.array([2, 3]), np.arange(1, 21))
    # aluno par prefere a primeira metade; ímpar, a segunda
    assert pontuacoes[0, :10].mean() > pontuacoes[0, 10:].mean()
    assert pontuacoes[1, 10:].mean() > pontuacoes[1, :10].mean()
    previsto = modelo.prever_avaliacao(treino["id_aluno"].to_numpy(), treino["id_material"].to_numpy())
    rmse = np.sqrt(np.mean((treino["avaliacao"].to_numpy() - previsto) ** 2))
    assert rmse < 1.0

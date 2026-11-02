"""Filtragem colaborativa: fatoração de matrizes com vieses (Koren, Bell & Volinsky, 2009).

    r̂(u, i) = μ + b_u + b_i + p_u · q_i

Cada aluno u e cada material i ganham um vetor latente de `fatores` dimensões,
aprendido só a partir das avaliações. Alunos com padrões de avaliação parecidos
acabam com vetores p_u próximos — é assim que o modelo usa "usuários similares".

Treino por mínimos quadrados alternados (ALS; Zhou et al., 2008): com os materiais
fixos, o vetor [p_u, b_u] de cada aluno é a solução exata de uma regressão ridge
sobre as notas dele; depois o mesmo para os materiais, e assim por diante.
A implementação é vetorizada: as matrizes de Gram Σ x_i x_iᵀ de todos os alunos
saem de um único produto esparso (indicadora aluno x material) @ (produtos externos
dos vetores dos materiais), e os sistemas (k+1)x(k+1) são resolvidos em lote.

A validação escolhe a iteração final. A curva de validação do ALS pode piorar nas
primeiras iterações antes de melhorar (os fatores saem de perto de zero), por isso
a paciência é generosa e o modelo guarda a melhor iteração vista.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import config
from .base import Recomendador, valores_por_id


def _rmse(reais: np.ndarray, previstas: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reais.astype(np.float64) - previstas) ** 2)))


class _EstruturaCSR:
    """Estrutura CSR (linhas x colunas) montada uma vez; a cada iteração só os valores mudam.

    Evita reordenar milhões de coordenadas por iteração. Avaliações repetidas do mesmo
    par ficam como entradas separadas, o que equivale a tratá-las como observações distintas.
    """

    def __init__(self, linhas: np.ndarray, colunas: np.ndarray, n_linhas: int, n_colunas: int):
        self.ordem = np.lexsort((colunas, linhas))
        self.indices = colunas[self.ordem].astype(np.int32)
        self.indptr = np.r_[0, np.cumsum(np.bincount(linhas, minlength=n_linhas))].astype(np.int32)
        self.forma = (n_linhas, n_colunas)

    def matriz(self, valores: np.ndarray) -> sp.csr_matrix:
        return sp.csr_matrix((valores[self.ordem], self.indices, self.indptr), shape=self.forma)


def _resolver_ridge(indicadora: sp.csr_matrix, alvos: sp.csr_matrix, X: np.ndarray, regularizacao: np.ndarray,
                    bloco: int = 20_000) -> np.ndarray:
    """Para cada linha u: argmin_x Σ_i (alvo_ui − x·X_i)² + xᵀ·diag(regularizacao)·x, com i nas colunas de u."""
    d = X.shape[1]
    externos = (X[:, :, None] * X[:, None, :]).reshape(len(X), d * d)
    saida = np.zeros((indicadora.shape[0], d))
    for inicio in range(0, indicadora.shape[0], bloco):
        fim = min(inicio + bloco, indicadora.shape[0])
        gram = np.asarray(indicadora[inicio:fim] @ externos).reshape(-1, d, d) + np.diag(regularizacao)
        termo = np.asarray(alvos[inicio:fim] @ X)
        saida[inicio:fim] = np.linalg.solve(gram, termo[..., None])[..., 0]
    return saida


class FiltragemColaborativa(Recomendador):
    nome = "Colaborativo"

    def __init__(self, n_usuarios: int, n_itens: int, fatores: int = config.FC_FATORES,
                 iteracoes: int = config.FC_ITERACOES, reg: float = config.FC_REGULARIZACAO,
                 reg_vies: float = config.FC_REGULARIZACAO_VIES, paciencia: int = config.FC_PACIENCIA,
                 escala_inicial: float = config.FC_ESCALA_INICIAL, semente: int = config.SEMENTE):
        self.fatores, self.iteracoes, self.reg, self.reg_vies = fatores, iteracoes, reg, reg_vies
        self.paciencia, self.escala_inicial, self.semente = paciencia, escala_inicial, semente
        self.media = 3.0
        self.vies_usuario = np.zeros(n_usuarios + 1, dtype=np.float32)
        self.vies_item = np.zeros(n_itens + 1, dtype=np.float32)
        self.P = np.zeros((n_usuarios + 1, fatores), dtype=np.float32)
        self.Q = np.zeros((n_itens + 1, fatores), dtype=np.float32)
        self.contagem_usuario = np.zeros(n_usuarios + 1, dtype=np.int64)
        self.historico: list[dict] = []
        self.melhor_iteracao: int | None = None

    def treinar(self, treino: pd.DataFrame, validacao: pd.DataFrame | None = None,
                iteracoes: int | None = None) -> "FiltragemColaborativa":
        """Com `validacao`, guarda os parâmetros da iteração de menor RMSE e para após
        `paciencia` iterações sem melhora. Sem ela, roda exatamente `iteracoes` iterações."""
        u = treino["id_aluno"].to_numpy()
        i = treino["id_material"].to_numpy()
        r = treino["avaliacao"].to_numpy().astype(np.float64)
        n_u, n_i, k = len(self.vies_usuario), len(self.vies_item), self.fatores
        rng = np.random.default_rng(self.semente)

        self.media = float(r.mean())
        P = rng.normal(0, self.escala_inicial, (n_u, k))
        Q = rng.normal(0, self.escala_inicial, (n_i, k))
        bu, bi = np.zeros(n_u), np.zeros(n_i)
        uns = np.ones(len(r))
        estrutura_u = _EstruturaCSR(u, i, n_u, n_i)
        estrutura_i = _EstruturaCSR(i, u, n_i, n_u)
        por_usuario, por_item = estrutura_u.matriz(uns), estrutura_i.matriz(uns)
        self.contagem_usuario = np.bincount(u, minlength=n_u)
        regularizacao = np.r_[np.full(k, self.reg), self.reg_vies]

        total = iteracoes or self.iteracoes
        self.historico, melhor_rmse, melhor, sem_melhora = [], np.inf, None, 0
        for iteracao in range(1, total + 1):
            x = _resolver_ridge(por_usuario, estrutura_u.matriz(r - self.media - bi[i]), np.c_[Q, np.ones(n_i)],
                                regularizacao)
            P, bu = x[:, :k], x[:, k]
            x = _resolver_ridge(por_item, estrutura_i.matriz(r - self.media - bu[u]), np.c_[P, np.ones(n_u)],
                                regularizacao)
            Q, bi = x[:, :k], x[:, k]
            self.P, self.Q = P.astype(np.float32), Q.astype(np.float32)
            self.vies_usuario, self.vies_item = bu.astype(np.float32), bi.astype(np.float32)
            if validacao is None:
                continue
            rmse = _rmse(validacao["avaliacao"].to_numpy(),
                         self.prever_avaliacao(validacao["id_aluno"].to_numpy(), validacao["id_material"].to_numpy()))
            self.historico.append({"iteracao": iteracao, "rmse_validacao": rmse})
            if rmse < melhor_rmse - 1e-5:
                melhor_rmse, sem_melhora, self.melhor_iteracao = rmse, 0, iteracao
                melhor = (self.P.copy(), self.Q.copy(), self.vies_usuario.copy(), self.vies_item.copy())
            else:
                sem_melhora += 1
                if sem_melhora >= self.paciencia:
                    break
        if melhor is not None:
            self.P, self.Q, self.vies_usuario, self.vies_item = melhor
        else:
            self.melhor_iteracao = total
        return self

    def pontuar(self, ids_alunos, ids_materiais):
        p = valores_por_id(self.P, ids_alunos)
        q = valores_por_id(self.Q, ids_materiais)
        return (self.media + valores_por_id(self.vies_usuario, ids_alunos)[:, None]
                + valores_por_id(self.vies_item, ids_materiais)[None, :] + p @ q.T).astype(np.float32)

    def prever_avaliacao(self, ids_alunos, ids_materiais, lote: int = 500_000):
        ids_alunos, ids_materiais = np.asarray(ids_alunos), np.asarray(ids_materiais)
        saida = np.empty(len(ids_alunos), dtype=np.float32)
        for inicio in range(0, len(ids_alunos), lote):
            a, m = ids_alunos[inicio:inicio + lote], ids_materiais[inicio:inicio + lote]
            saida[inicio:inicio + lote] = (self.media + valores_por_id(self.vies_usuario, a)
                                           + valores_por_id(self.vies_item, m)
                                           + np.einsum("ij,ij->i", valores_por_id(self.P, a),
                                                       valores_por_id(self.Q, m)))
        return np.clip(saida, 1, 5)

    def vizinhos(self, id_aluno: int, n: int = 5) -> pd.DataFrame:
        """Alunos com vetor latente mais parecido (cosseno), entre os que têm histórico."""
        normas = np.linalg.norm(self.P, axis=1)
        if id_aluno >= len(normas) or normas[id_aluno] == 0:
            return pd.DataFrame(columns=["id_aluno", "similaridade"])
        similaridade = (self.P @ self.P[id_aluno]) / np.maximum(normas * normas[id_aluno], 1e-12)
        similaridade[(self.contagem_usuario == 0) | (np.arange(len(normas)) == id_aluno)] = -np.inf
        melhores = np.argsort(-similaridade)[:n]
        return pd.DataFrame({"id_aluno": melhores, "similaridade": similaridade[melhores]})

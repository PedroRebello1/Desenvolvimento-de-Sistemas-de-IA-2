"""Filtragem baseada em conteúdo: similaridade entre o perfil do aluno e os metadados.

Cada material vira um vetor com três blocos, normalizados separadamente:
- área (one-hot nas 8 áreas do catálogo)          peso 0,5
- formato (tipo + nível de dificuldade, one-hot)   peso 0,2
- texto (TF-IDF de título + descrição)             peso 0,3
Assim, o produto escalar entre dois vetores é a média ponderada do cosseno de cada bloco.

O perfil do aluno vive no mesmo espaço e soma duas partes:
- declarada: interesses e disciplinas projetados nas áreas pela ontologia + TF-IDF desses termos;
- histórica (Rocchio): média dos materiais avaliados, com peso (nota − 3) / 2, ou seja,
  positivo para notas 4–5 e negativo para 1–2.
A parte histórica ganha peso com o tamanho do histórico: n / (n + n0).

Para prever notas (RMSE) usa um kNN de conteúdo: a nota de um material é o
baseline μ + b_u + b_i corrigido pelos resíduos que o aluno deu a materiais parecidos.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import ontologia
from .base import Recomendador, valores_por_id
from .baselines import BaselineVieses

STOPWORDS = frozenset("a o e de da do das dos em para por com sem sobre no na nos nas um uma ao aos as os".split())
PESOS_BLOCOS = {"area": 0.5, "formato": 0.2, "texto": 0.3}
PESO_DISCIPLINAS = 0.6   # interesses pesam 1; disciplinas já cursadas, um pouco menos


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sem acentos, sem stopwords e com um stemming mínimo (remove o plural em -s)."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    palavras = [p for p in re.findall(r"[a-z0-9]+", texto) if p not in STOPWORDS]
    return " ".join(p[:-1] if len(p) > 3 and p.endswith("s") else p for p in palavras)


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz / np.where(normas > 0, normas, 1)


class FiltragemConteudo(Recomendador):
    nome = "Conteúdo"

    def __init__(self, alunos: pd.DataFrame, materiais: pd.DataFrame, n_usuarios: int, n_itens: int,
                 baseline: BaselineVieses | None = None, n0_historico: float = 5.0,
                 expoente_knn: float = 2.0, encolhimento_knn: float = 1.0):
        self.alunos, self.materiais = alunos, materiais
        self.n_usuarios, self.n_itens = n_usuarios, n_itens
        self.baseline = baseline
        self.n0_historico, self.expoente_knn, self.encolhimento_knn = n0_historico, expoente_knn, encolhimento_knn

        self.vetorizador = TfidfVectorizer(preprocessor=normalizar_texto, token_pattern=r"(?u)\b\w\w+\b")
        texto = self.vetorizador.fit_transform(materiais["titulo"] + " " + materiais["descricao"]).toarray()
        area = np.array([[a == m for a in ontologia.AREAS] for m in materiais["area"]], dtype=np.float32)
        formato = np.array([[t == m for t in ontologia.TIPOS] + [n == v for n in ontologia.NIVEIS]
                            for m, v in zip(materiais["tipo"], materiais["nivel"])], dtype=np.float32)
        # Blocos crus (cada linha com norma 1) para itens cadastrados; demais itens ficam com zeros.
        self.blocos_itens = {}
        for nome, bloco in (("area", area), ("formato", formato), ("texto", texto)):
            matriz = np.zeros((n_itens + 1, bloco.shape[1]), dtype=np.float32)
            matriz[materiais.index.to_numpy()] = _normalizar(bloco)
            self.blocos_itens[nome] = matriz
        self.vetores_itens = self._juntar(self.blocos_itens)
        self.blocos_declarados = self._perfis_declarados()
        self.perfis = self._juntar(self.blocos_declarados)
        self.n_historico = np.zeros(n_usuarios + 1, dtype=np.int64)
        self.residuos = sp.csr_matrix((n_usuarios + 1, n_itens + 1), dtype=np.float32)

    @staticmethod
    def _juntar(blocos: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([np.sqrt(PESOS_BLOCOS[nome]) * _normalizar(blocos[nome]) for nome in PESOS_BLOCOS])

    def _perfis_declarados(self) -> dict[str, np.ndarray]:
        area = np.zeros((self.n_usuarios + 1, len(ontologia.AREAS)), dtype=np.float32)
        textos = []
        for id_aluno, disciplinas, interesses in zip(self.alunos.index, self.alunos["disciplinas"],
                                                     self.alunos["interesses"]):
            por_interesse = ontologia.afinidade_areas(interesses)
            por_disciplina = ontologia.afinidade_areas(disciplinas)
            area[id_aluno] = [max(por_interesse[a], PESO_DISCIPLINAS * por_disciplina[a]) for a in ontologia.AREAS]
            textos.append(" ".join(disciplinas + interesses))
        texto = np.zeros((self.n_usuarios + 1, len(self.vetorizador.vocabulary_)), dtype=np.float32)
        texto[self.alunos.index.to_numpy()] = self.vetorizador.transform(textos).toarray()
        formato = np.zeros((self.n_usuarios + 1, self.blocos_itens["formato"].shape[1]), dtype=np.float32)
        return {"area": area, "formato": formato, "texto": texto}

    def treinar(self, treino: pd.DataFrame) -> "FiltragemConteudo":
        com_conteudo = treino["id_material"].isin(self.materiais.index).to_numpy()
        sub = treino[com_conteudo]
        u = sub["id_aluno"].to_numpy()
        i = sub["id_material"].to_numpy()
        r = sub["avaliacao"].to_numpy().astype(np.float32)
        forma = (self.n_usuarios + 1, self.n_itens + 1)

        pesos = sp.csr_matrix(((r - 3) / 2, (u, i)), shape=forma)
        self.n_historico = np.bincount(u, minlength=self.n_usuarios + 1)
        confianca = (self.n_historico / (self.n_historico + self.n0_historico))[:, None].astype(np.float32)
        blocos = {}
        for nome, declarado in self.blocos_declarados.items():
            historico = _normalizar(np.asarray(pesos @ self.blocos_itens[nome], dtype=np.float32))
            blocos[nome] = _normalizar(declarado) + confianca * historico
        self.perfis = self._juntar(blocos).astype(np.float32)

        if self.baseline is not None:
            residuo = r - self.baseline.prever_avaliacao(u, i)
            self.residuos = sp.csr_matrix((residuo, (u, i)), shape=forma)
        return self

    def pontuar(self, ids_alunos, ids_materiais):
        return valores_por_id(self.perfis, ids_alunos) @ valores_por_id(self.vetores_itens, ids_materiais).T

    def prever_avaliacao(self, ids_alunos, ids_materiais):
        ids_alunos, ids_materiais = np.asarray(ids_alunos), np.asarray(ids_materiais)
        previsto = (self.baseline.prever_avaliacao(ids_alunos, ids_materiais).astype(np.float64)
                    if self.baseline is not None else np.full(len(ids_alunos), 3.0))
        ordem = np.argsort(ids_alunos, kind="stable")
        alunos_ordenados = ids_alunos[ordem]
        inicios = np.flatnonzero(np.r_[True, alunos_ordenados[1:] != alunos_ordenados[:-1]])
        for inicio, fim in zip(inicios, np.r_[inicios[1:], len(ordem)]):
            aluno = alunos_ordenados[inicio]
            if aluno >= self.residuos.shape[0]:
                continue
            linha = self.residuos[aluno]
            if linha.nnz == 0:
                continue
            posicoes = ordem[inicio:fim]
            alvo = valores_por_id(self.vetores_itens, ids_materiais[posicoes])
            similaridade = np.clip(alvo @ self.vetores_itens[linha.indices].T, 0, None) ** self.expoente_knn
            previsto[posicoes] += (similaridade @ linha.data) / (similaridade.sum(axis=1) + self.encolhimento_knn)
        return np.clip(previsto, 1, 5).astype(np.float32)

    def contribuicoes(self, id_aluno: int, id_material: int) -> dict[str, float]:
        """Quanto cada bloco (área, formato, texto) contribui para a similaridade."""
        saida, inicio = {}, 0
        for nome in PESOS_BLOCOS:
            largura = self.blocos_itens[nome].shape[1]
            fatia = slice(inicio, inicio + largura)
            saida[nome] = float(self.perfis[id_aluno, fatia] @ self.vetores_itens[id_material, fatia])
            inicio += largura
        return saida

    def termos_em_comum(self, id_aluno: int, id_material: int) -> list[str]:
        vocabulario = self.vetorizador.get_feature_names_out()
        comuns = (self.blocos_declarados["texto"][id_aluno] > 0) & (self.blocos_itens["texto"][id_material] > 0)
        return [str(t) for t in vocabulario[comuns]]

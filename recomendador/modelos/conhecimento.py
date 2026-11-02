"""Modelo baseado em conhecimento: regras explícitas sobre requisitos acadêmicos.

Para cada par (aluno, obra) aplica cinco regras da base de conhecimento (ontologia.py):

1. Pré-requisitos  — o aluno tem as competências exigidas? (restrição: inelegível -> 0)
2. Curso           — quanto a área da obra pesa no currículo do curso do aluno
3. Nível           — dificuldade da obra versus o nível esperado para o período
4. Novidade        — obra introdutória de algo que o aluno já cursou é redundante
5. Desbloqueio     — a obra ensina um pré-requisito que falta para outra obra de
                     uma área de interesse do aluno (trilha de aprendizagem)

    pontuação = elegível x (0,35·curso + 0,35·nível + 0,15·novidade + 0,15·desbloqueio)

As competências vêm das disciplinas cursadas e das obras concluídas no histórico
(avaliação >= 3). O modelo não aprende nada dos dados além dessas conclusões, por
isso funciona para alunos sem histórico (cold start) e sempre justifica a nota.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ontologia
from .base import Recomendador

PESOS_REGRAS = {"curso": 0.35, "nivel": 0.35, "novidade": 0.15, "desbloqueio": 0.15}
INTERESSE_MINIMO = 0.5     # afinidade mínima com a área para a regra de desbloqueio


class ModeloConhecimento(Recomendador):
    nome = "Conhecimento"

    def __init__(self, alunos: pd.DataFrame, materiais: pd.DataFrame, n_usuarios: int, n_itens: int,
                 limiar_conclusao: int = 3):
        self.alunos, self.limiar_conclusao = alunos, limiar_conclusao
        self.obras = (materiais.groupby("id_obra")[["titulo", "tipo", "area", "nivel"]].first()
                      .sort_index().reset_index())
        self.regras = [ontologia.regra_da_obra(t) for t in self.obras["titulo"]]
        self.nivel_obra = [ontologia.NIVEIS.index(n) if n in ontologia.NIVEIS else 1 for n in self.obras["nivel"]]
        self.obra_do_material = np.full(n_itens + 1, -1, dtype=np.int32)
        self.obra_do_material[materiais.index.to_numpy()] = materiais["id_obra"].to_numpy()
        self.tem_perfil = np.zeros(n_usuarios + 1, dtype=bool)
        self.tem_perfil[alunos.index.to_numpy()] = True
        self.concluidas: dict[int, frozenset[int]] = {}
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def treinar(self, treino: pd.DataFrame) -> "ModeloConhecimento":
        material = treino["id_material"].to_numpy()
        obra = np.where(material < len(self.obra_do_material),
                        self.obra_do_material[np.minimum(material, len(self.obra_do_material) - 1)], -1)
        aluno = treino["id_aluno"].to_numpy()
        valido = ((treino["avaliacao"].to_numpy() >= self.limiar_conclusao) & (obra >= 0)
                  & (aluno < len(self.tem_perfil)) & self.tem_perfil[np.minimum(aluno, len(self.tem_perfil) - 1)])
        pares = pd.DataFrame({"aluno": aluno[valido], "obra": obra[valido]}).drop_duplicates()
        self.concluidas = {int(a): frozenset(g.tolist()) for a, g in pares.groupby("aluno")["obra"]}
        self._cache.clear()
        return self

    def possui_perfil(self, ids_alunos) -> np.ndarray:
        ids_alunos = np.asarray(ids_alunos)
        dentro = ids_alunos < len(self.tem_perfil)
        return np.where(dentro, self.tem_perfil[np.where(dentro, ids_alunos, 0)], False)

    def competencias(self, id_aluno: int) -> set[str]:
        competencias = ontologia.competencias_declaradas(self.alunos.at[id_aluno, "disciplinas"])
        for obra in self.concluidas.get(id_aluno, ()):
            competencias |= self.regras[obra].ensina
        return competencias

    def _aplicar_regras(self, id_aluno: int) -> list[dict]:
        """As cinco regras para cada obra, sem pandas no caminho (é chamada para milhares de alunos)."""
        aluno = self.alunos.loc[id_aluno]
        competencias = self.competencias(id_aluno)
        cursadas = ontologia.competencias_declaradas(aluno["disciplinas"])
        interesse = ontologia.afinidade_areas(aluno["interesses"])
        relevancia_curso = ontologia.CURSO_AREAS.get(aluno["curso"], {})
        periodo = int(aluno["periodo"])
        titulos, areas = self.obras["titulo"].tolist(), self.obras["area"].tolist()
        faltantes = [ontologia.prerequisitos_faltantes(regra, competencias) for regra in self.regras]
        bloqueadas_de_interesse = [w for w, falta in enumerate(faltantes)
                                   if falta and interesse.get(areas[w], 0) >= INTERESSE_MINIMO]
        linhas = []
        for w, regra in enumerate(self.regras):
            curso = relevancia_curso.get(areas[w], 0.0)
            nivel = ontologia.adequacao_nivel(self.nivel_obra[w], periodo)
            redundante = bool(regra.ensina) and regra.ensina <= cursadas
            desbloqueia = [titulos[alvo] for alvo in bloqueadas_de_interesse
                           if any(regra.ensina & grupo for grupo in faltantes[alvo])]
            elegivel = not faltantes[w]
            pontuacao = elegivel * (PESOS_REGRAS["curso"] * curso + PESOS_REGRAS["nivel"] * nivel
                                    + PESOS_REGRAS["novidade"] * (not redundante)
                                    + PESOS_REGRAS["desbloqueio"] * bool(desbloqueia))
            linhas.append({"id_obra": w, "titulo": titulos[w], "elegivel": elegivel, "faltantes": faltantes[w],
                           "curso": curso, "nivel": nivel, "redundante": redundante, "desbloqueia": desbloqueia,
                           "pontuacao": pontuacao})
        return linhas

    def avaliar_obras(self, id_aluno: int) -> pd.DataFrame:
        """Aplica as regras a todas as obras para um aluno, com as justificativas de cada uma."""
        aluno = self.alunos.loc[id_aluno]
        tabela = pd.DataFrame(self._aplicar_regras(id_aluno))
        tabela["justificativas"] = [self._justificar(aluno, self.obras.loc[linha.id_obra], linha)
                                    for linha in tabela.itertuples()]
        return tabela.drop(columns="faltantes")

    def _justificar(self, aluno, obra, linha) -> list[str]:
        textos = []
        if linha.faltantes:
            grupos = [" ou ".join(sorted(g)) for g in linha.faltantes]
            textos.append("Faltam pré-requisitos: " + "; ".join(grupos))
        elif self.regras[linha.id_obra].prerequisitos:
            textos.append("Pré-requisitos atendidos")
        if linha.curso >= 0.8:
            textos.append(f"Área central do curso {aluno['curso']}")
        elif linha.curso >= 0.4:
            textos.append(f"Área complementar ao curso {aluno['curso']}")
        else:
            textos.append(f"Pouca relação com o curso {aluno['curso']}")
        esperado = ontologia.NIVEIS[ontologia.nivel_esperado(int(aluno["periodo"]))]
        if obra["nivel"] == esperado:
            textos.append(f"Nível {obra['nivel']} adequado ao {aluno['periodo']}º período")
        else:
            textos.append(f"Nível {obra['nivel']} (esperado para o {aluno['periodo']}º período: {esperado})")
        if linha.redundante:
            textos.append("Conteúdo de disciplina já cursada")
        if linha.desbloqueia:
            textos.append("Desbloqueia: " + ", ".join(linha.desbloqueia))
        return textos

    def _por_obra(self, id_aluno: int) -> tuple[np.ndarray, np.ndarray]:
        """(pontuação, elegibilidade) de cada obra; alunos com o mesmo perfil e as
        mesmas obras concluídas compartilham o resultado em cache."""
        chave = (int(self.alunos.at[id_aluno, "id_perfil"]), self.concluidas.get(id_aluno, frozenset()))
        if chave not in self._cache:
            linhas = self._aplicar_regras(id_aluno)
            self._cache[chave] = (np.array([l["pontuacao"] for l in linhas], dtype=np.float32),
                                  np.array([l["elegivel"] for l in linhas]))
        return self._cache[chave]

    def _matriz(self, ids_alunos, ids_materiais, coluna: int, padrao):
        ids_materiais = np.asarray(ids_materiais)
        obra = np.where(ids_materiais < len(self.obra_do_material),
                        self.obra_do_material[np.minimum(ids_materiais, len(self.obra_do_material) - 1)], -1)
        conhecida = obra >= 0
        saida = np.full((len(ids_alunos), len(ids_materiais)), padrao)
        for linha, (id_aluno, tem) in enumerate(zip(ids_alunos, self.possui_perfil(ids_alunos))):
            if tem:
                valores = self._por_obra(int(id_aluno))[coluna]
                saida[linha, conhecida] = valores[obra[conhecida]]
        return saida

    def pontuar(self, ids_alunos, ids_materiais):
        return self._matriz(ids_alunos, ids_materiais, 0, np.float32(0)).astype(np.float32)

    def pontuar_pares(self, ids_alunos, ids_materiais) -> np.ndarray:
        """Pontuação de pares alinhados aluno/material (0 onde as regras não se aplicam)."""
        ids_alunos, ids_materiais = np.asarray(ids_alunos), np.asarray(ids_materiais)
        obra = np.where(ids_materiais < len(self.obra_do_material),
                        self.obra_do_material[np.minimum(ids_materiais, len(self.obra_do_material) - 1)], -1)
        saida = np.zeros(len(ids_alunos), dtype=np.float32)
        aplica = np.flatnonzero(self.possui_perfil(ids_alunos) & (obra >= 0))
        ordem = aplica[np.argsort(ids_alunos[aplica], kind="stable")]
        for grupo in np.split(ordem, np.flatnonzero(np.diff(ids_alunos[ordem])) + 1):
            if len(grupo):
                saida[grupo] = self._por_obra(int(ids_alunos[grupo[0]]))[0][obra[grupo]]
        return saida

    def elegibilidade(self, ids_alunos, ids_materiais) -> np.ndarray:
        """True onde o aluno cumpre os pré-requisitos (ou onde as regras não se aplicam)."""
        return self._matriz(ids_alunos, ids_materiais, 1, True).astype(bool)

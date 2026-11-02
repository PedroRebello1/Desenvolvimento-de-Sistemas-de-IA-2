"""Cenário com sinal controlado e teste A/B simulado.

As interações fornecidas são ruído uniforme (ver analise.py), então nenhum modelo
pode mostrar ganho real com elas. Para verificar que a implementação funciona,
o simulador cria um "mundo" com os MESMOS alunos e materiais dos CSVs, mas em que
as interações seguem uma utilidade conhecida, desconhecida pelos modelos:

    u(aluno, material) = 1,0·tópico + 0,8·adequação + 0,8·latente + 0,4·qualidade

- tópico:    gosto do perfil pela área da obra. 70% vem dos interesses declarados
             (via ontologia) e 30% é idiossincrasia sorteada por perfil — o perfil
             declarado informa, mas não determina o gosto;
- adequação: pré-requisitos, nível x período e redundância — o que as regras codificam;
- latente:   produto de vetores aleatórios do aluno e do material (4 dimensões):
             gosto individual que nenhum metadado explica (só a FC pode aprender);
- qualidade: viés sorteado por material (cópias da mesma obra não são idênticas).
Cada termo é padronizado antes da soma.

Interações: N ~ Poisson(25) por aluno ao longo de 2023; 70% escolhidas proporcionalmente
a exp(u / τ) (o aluno procura o que gosta), 30% ao acaso (exposição).
Nota = arredondar(3 + 1,1·z(u) + viés do aluno + ruído), limitada a 1–5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, ontologia
from .dados import Base, dia_de
from .modelos import Recomendador
from .modelos.base import valores_por_id

PESOS_UTILIDADE = {"topico": 1.0, "adequacao": 0.8, "latente": 0.8, "qualidade": 0.4}


def _padronizar(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


class SimuladorPreferencias:
    def __init__(self, base: Base, semente: int = config.SEMENTE, dimensoes_latentes: int = 4,
                 fracao_idiossincrasia: float = 0.3, temperatura: float = 1.0):
        self.base, self.temperatura = base, temperatura
        rng = np.random.default_rng(semente)
        alunos, materiais = base.alunos, base.materiais
        self.ids_alunos = alunos.index.to_numpy()
        self.ids_materiais = materiais.index.to_numpy()
        n_u, n_i = base.n_usuarios, base.n_itens

        # Tópico: perfil x área.
        perfis = alunos.groupby("id_perfil").first()
        declarado = np.array([[ontologia.afinidade_areas(p.interesses)[a] for a in ontologia.AREAS]
                              for p in perfis.itertuples()])
        idiossincrasia = rng.normal(size=declarado.shape)
        topico = ((1 - fracao_idiossincrasia) * _padronizar(declarado)
                  + fracao_idiossincrasia * _padronizar(idiossincrasia))
        # Adequação acadêmica: perfil x obra (só competências declaradas).
        obras = materiais.groupby("id_obra")[["titulo", "area", "nivel"]].first().sort_index()
        adequacao = np.zeros((len(perfis), len(obras)))
        for p, perfil in enumerate(perfis.itertuples()):
            cursadas = ontologia.competencias_declaradas(perfil.disciplinas)
            for w, obra in enumerate(obras.itertuples()):
                regra = ontologia.regra_da_obra(obra.titulo)
                nivel = ontologia.NIVEIS.index(obra.nivel) if obra.nivel in ontologia.NIVEIS else 1
                adequacao[p, w] = (-1.5 * bool(ontologia.prerequisitos_faltantes(regra, cursadas))
                                   + ontologia.adequacao_nivel(nivel, int(perfil.periodo))
                                   - 0.5 * (bool(regra.ensina) and regra.ensina <= cursadas))
        area_da_obra = np.array([ontologia.AREAS.index(a) for a in obras["area"]])
        # Matriz perfil x obra com os dois termos de conteúdo já padronizados e ponderados.
        self.perfil_obra = (PESOS_UTILIDADE["topico"] * _padronizar(topico[:, area_da_obra])
                            + PESOS_UTILIDADE["adequacao"] * _padronizar(adequacao)).astype(np.float32)

        self.perfil_do_aluno = np.full(n_u + 1, -1, dtype=np.int32)
        self.perfil_do_aluno[self.ids_alunos] = alunos["id_perfil"].to_numpy()
        self.obra_do_material = np.full(n_i + 1, -1, dtype=np.int32)
        self.obra_do_material[self.ids_materiais] = materiais["id_obra"].to_numpy()
        self.latente_aluno = np.zeros((n_u + 1, dimensoes_latentes), dtype=np.float32)
        self.latente_aluno[self.ids_alunos] = rng.normal(size=(len(self.ids_alunos), dimensoes_latentes))
        self.latente_material = np.zeros((n_i + 1, dimensoes_latentes), dtype=np.float32)
        self.latente_material[self.ids_materiais] = rng.normal(size=(len(self.ids_materiais), dimensoes_latentes))
        self.escala_latente = 1 / np.sqrt(dimensoes_latentes)   # produto de normais padrão tem desvio sqrt(d)
        self.qualidade = np.zeros(n_i + 1, dtype=np.float32)
        self.qualidade[self.ids_materiais] = rng.normal(size=len(self.ids_materiais))
        self.vies_nota = np.zeros(n_u + 1, dtype=np.float32)
        self.vies_nota[self.ids_alunos] = rng.normal(0, 0.3, size=len(self.ids_alunos))
        # Média e desvio de u na população, para converter utilidade em nota.
        amostra = rng.choice(self.ids_alunos, 300, replace=False)
        u = self.utilidade(amostra, self.ids_materiais)
        self.media_u, self.desvio_u = float(u.mean()), float(u.std())

    def utilidade(self, ids_alunos, ids_materiais) -> np.ndarray:
        """Utilidade verdadeira (alunos x materiais)."""
        perfil = valores_por_id(self.perfil_do_aluno, ids_alunos)
        obra = valores_por_id(self.obra_do_material, ids_materiais)
        u = self.perfil_obra[perfil][:, obra]
        u = u + PESOS_UTILIDADE["latente"] * self.escala_latente * (
            valores_por_id(self.latente_aluno, ids_alunos) @ valores_por_id(self.latente_material, ids_materiais).T)
        return u + PESOS_UTILIDADE["qualidade"] * valores_por_id(self.qualidade, ids_materiais)[None, :]

    def utilidade_padronizada(self, ids_alunos, ids_materiais) -> np.ndarray:
        return (self.utilidade(ids_alunos, ids_materiais) - self.media_u) / self.desvio_u

    def gerar_interacoes(self, media_por_aluno: float = 25, fracao_preferencia: float = 0.7,
                         semente: int = config.SEMENTE + 1, lote: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(semente)
        materiais = self.base.materiais
        tipo_material = materiais["tipo"].reindex(self.ids_materiais).to_numpy()
        prob_leitura = np.select([tipo_material == "livro", tipo_material == "artigo"], [0.75, 0.65], 0.2)
        partes = []
        for inicio in range(0, len(self.ids_alunos), lote):
            ids = self.ids_alunos[inicio:inicio + lote]
            n = np.maximum(rng.poisson(media_por_aluno, len(ids)), 1)
            n_preferidas = rng.binomial(n, fracao_preferencia)
            z = self.utilidade_padronizada(ids, self.ids_materiais)
            # Gumbel-top-k: amostra sem reposição proporcional a exp(z / τ).
            chaves = z / self.temperatura + rng.gumbel(size=z.shape)
            ordem_preferencia = np.argsort(-chaves, axis=1)
            ordem_aleatoria = np.argsort(rng.random(z.shape), axis=1)
            for linha, aluno in enumerate(ids):
                preferidas = ordem_preferencia[linha, :n_preferidas[linha]]
                restantes = ordem_aleatoria[linha][~np.isin(ordem_aleatoria[linha], preferidas)]
                escolhidas = np.concatenate([preferidas, restantes[:n[linha] - n_preferidas[linha]]])
                zz = z[linha, escolhidas]
                nota = np.clip(np.rint(3 + 1.1 * zz + self.vies_nota[aluno] + rng.normal(0, 0.7, len(zz))), 1, 5)
                partes.append(pd.DataFrame({
                    "id_aluno": np.int32(aluno),
                    "id_material": self.ids_materiais[escolhidas].astype(np.int32),
                    "tipo_interacao": np.where(rng.random(len(zz)) < prob_leitura[escolhidas], "leitura",
                                               "visualizacao"),
                    "avaliacao": nota.astype(np.int8),
                    "duracao_minutos": np.clip(np.rint(rng.normal(60 + 35 * zz, 25)), 5, 240).astype(np.int16),
                    "dia": rng.integers(dia_de("2023-01-01"), dia_de("2023-12-31") + 1, len(zz)).astype(np.int32),
                }))
        interacoes = pd.concat(partes, ignore_index=True)
        interacoes["tipo_interacao"] = pd.Categorical(interacoes["tipo_interacao"],
                                                      categories=["visualizacao", "leitura"])
        return interacoes


class Oraculo(Recomendador):
    """Ranqueia pela utilidade verdadeira: o teto que um modelo perfeito alcançaria."""
    nome = "Oráculo (teto)"

    def __init__(self, simulador: SimuladorPreferencias):
        self.simulador = simulador

    def pontuar(self, ids_alunos, ids_materiais):
        return self.simulador.utilidade(ids_alunos, ids_materiais).astype(np.float32)


def simular_teste_ab(simulador: SimuladorPreferencias, modelos: list[Recomendador], treino: pd.DataFrame,
                     sessoes_por_modelo: int = 5000, k: int = 10, taxa_resposta: float = 0.3,
                     semente: int = config.SEMENTE + 2, lote: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teste A/B simulado: cada sessão mostra k recomendações de um modelo a um aluno sorteado.

    Comportamento do aluno simulado (z = utilidade padronizada):
    - examina a posição p com probabilidade 1 / log2(p + 1) (viés de posição);
    - clica no item examinado com probabilidade sigmoide(1,5·(z − 1));
    - cada clique gera N(20 + 15·z, 8) minutos de engajamento (limitado a 1–120);
    - com probabilidade `taxa_resposta` responde à pesquisa de satisfação (1–5), que
      sobe com a utilidade média dos itens examinados e com a variedade de obras da lista.
    Devolve (resumo por modelo com IC de 95%, tabela de sessões).
    """
    rng = np.random.default_rng(semente)
    candidatos = simulador.ids_materiais
    exame = 1 / np.log2(np.arange(2, k + 2))
    historico = treino[treino["id_material"].isin(candidatos).to_numpy()]
    vistos_por_aluno = historico.groupby("id_aluno")["id_material"].apply(np.asarray)
    posicao = np.full(candidatos.max() + 1, -1)
    posicao[candidatos] = np.arange(len(candidatos))
    sessoes = []
    for modelo in modelos:
        alunos = rng.choice(simulador.ids_alunos, sessoes_por_modelo)
        for inicio in range(0, len(alunos), lote):
            ids = alunos[inicio:inicio + lote]
            vistos = np.zeros((len(ids), len(candidatos)), dtype=bool)
            for linha, aluno in enumerate(ids):
                if aluno in vistos_por_aluno.index:
                    vistos[linha, posicao[vistos_por_aluno[aluno]]] = True
            listas = modelo.recomendar(ids, candidatos, k, vistos)
            z = np.take_along_axis(simulador.utilidade_padronizada(ids, candidatos), posicao[listas], axis=1)
            examinou = rng.random(z.shape) < exame
            clicou = examinou & (rng.random(z.shape) < 1 / (1 + np.exp(-1.5 * (z - 1))))
            minutos = np.where(clicou, np.clip(rng.normal(20 + 15 * z, 8), 1, 120), 0).sum(axis=1)
            obras_distintas = np.array([len(np.unique(simulador.obra_do_material[lista])) for lista in listas])
            qualidade = (z * exame).sum(axis=1) / exame.sum()
            satisfacao = np.clip(np.rint(3 + 0.9 * qualidade + 0.15 * (obras_distintas - 5)
                                         + rng.normal(0, 0.6, len(ids))), 1, 5)
            respondeu = rng.random(len(ids)) < taxa_resposta
            sessoes.append(pd.DataFrame({"modelo": modelo.nome, "id_aluno": ids, "impressoes": k,
                                         "cliques": clicou.sum(axis=1), "minutos": minutos,
                                         "satisfacao": np.where(respondeu, satisfacao, np.nan)}))
    sessoes = pd.concat(sessoes, ignore_index=True)
    return resumir_online(sessoes), sessoes


def resumir_online(sessoes: pd.DataFrame) -> pd.DataFrame:
    """CTR, minutos por sessão e satisfação média por modelo, com IC de 95% (aprox. normal)."""
    linhas = []
    for modelo, grupo in sessoes.groupby("modelo", sort=False):
        impressoes, cliques = grupo["impressoes"].sum(), grupo["cliques"].sum()
        ctr = cliques / impressoes
        respostas = grupo["satisfacao"].dropna()
        linhas.append({
            "modelo": modelo, "sessoes": len(grupo), "impressoes": int(impressoes), "cliques": int(cliques),
            "CTR": ctr, "CTR IC95": 1.96 * np.sqrt(ctr * (1 - ctr) / impressoes),
            "minutos por sessão": grupo["minutos"].mean(),
            "minutos IC95": 1.96 * grupo["minutos"].std() / np.sqrt(len(grupo)),
            "satisfação (1–5)": respostas.mean() if len(respostas) else np.nan,
            "satisfação IC95": 1.96 * respostas.std() / np.sqrt(len(respostas)) if len(respostas) > 1 else np.nan,
            "respostas": len(respostas)})
    return pd.DataFrame(linhas)

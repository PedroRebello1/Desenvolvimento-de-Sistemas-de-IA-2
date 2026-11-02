"""Experimento completo sobre um conjunto de interações.

1. Validação (treino jan–ago, validação set–out):
   - número de iterações da filtragem colaborativa (parada antecipada);
   - busca em grade dos pesos do híbrido (NDCG@10);
   - coeficientes do empilhamento de notas e da calibração das regras.
2. Teste (treino jan–out, teste nov–dez): todos os modelos avaliados no mesmo protocolo.
3. Opcional: cold start (alunos cujo histórico de treino é removido) e a curva
   precisão x diversidade do MMR.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .avaliacao import (Protocolo, acertos, avaliar_notas, avaliar_ranqueamento, montar_protocolo,
                        precisao_recall_ndcg)
from .dados import Base, dia_de
from .modelos import (COMPONENTES, Aleatorio, BaselineVieses, FiltragemColaborativa, FiltragemConteudo, Hibrido,
                      MediaGlobal, ModeloConhecimento, Popularidade, Recomendador)

COLUNAS_TREINO = ["id_aluno", "id_material", "avaliacao"]


@dataclass
class Componentes:
    baseline: BaselineVieses
    popularidade: Popularidade
    conteudo: FiltragemConteudo
    colaborativo: FiltragemColaborativa
    conhecimento: ModeloConhecimento

    def hibrido(self, **parametros) -> Hibrido:
        return Hibrido(self.conteudo, self.colaborativo, self.conhecimento, self.popularidade, **parametros)


def treinar_componentes(base: Base, treino: pd.DataFrame, validacao: pd.DataFrame | None = None,
                        iteracoes_fc: int | None = None, log=print) -> Componentes:
    n_u, n_i = base.n_usuarios, base.n_itens
    inicio = time.time()
    baseline = BaselineVieses(n_u, n_i).treinar(treino)
    popularidade = Popularidade(n_i).treinar(treino)
    conteudo = FiltragemConteudo(base.alunos, base.materiais, n_u, n_i, baseline=baseline).treinar(treino)
    conhecimento = ModeloConhecimento(base.alunos, base.materiais, n_u, n_i).treinar(treino)
    colaborativo = FiltragemColaborativa(n_u, n_i).treinar(treino, validacao=validacao, iteracoes=iteracoes_fc)
    log(f"    componentes treinados em {time.time() - inicio:.0f}s "
        f"({len(treino):,} interações; FC: {colaborativo.melhor_iteracao} iterações)")
    return Componentes(baseline, popularidade, conteudo, colaborativo, conhecimento)


class ConhecimentoCalibrado(Recomendador):
    """Transforma a pontuação das regras em nota: baseline + β0 + β1·pontuação (β da validação)."""
    nome = "Conhecimento"

    def __init__(self, conhecimento: ModeloConhecimento, baseline: BaselineVieses, coef=None):
        self.conhecimento, self.baseline = conhecimento, baseline
        self.coef = np.zeros(2) if coef is None else np.asarray(coef)

    def ajustar(self, pares: pd.DataFrame) -> "ConhecimentoCalibrado":
        a, m = pares["id_aluno"].to_numpy(), pares["id_material"].to_numpy()
        atributos = np.column_stack([np.ones(len(a)), self.conhecimento.pontuar_pares(a, m)])
        alvo = pares["avaliacao"].to_numpy() - self.baseline.prever_avaliacao(a, m)
        self.coef = np.linalg.lstsq(atributos, alvo, rcond=None)[0]
        return self

    def pontuar(self, ids_alunos, ids_materiais):
        return self.conhecimento.pontuar(ids_alunos, ids_materiais)

    def prever_avaliacao(self, ids_alunos, ids_materiais):
        pontos = self.conhecimento.pontuar_pares(ids_alunos, ids_materiais)
        previsto = self.baseline.prever_avaliacao(ids_alunos, ids_materiais) + self.coef[0] + self.coef[1] * pontos
        return np.clip(previsto, 1, 5).astype(np.float32)


def combinacoes_pesos(passo: float = 0.1, minimo: float = config.PESO_MINIMO) -> list[tuple[float, float, float]]:
    """Pontos do simplex (conteúdo, colaborativo, conhecimento) com passo 0,1 e nenhum peso abaixo do mínimo."""
    valores = np.round(np.arange(minimo, 1 - 2 * minimo + 1e-9, passo), 10)
    return [(a, b, round(1 - a - b, 10)) for a in valores for b in valores if 1 - a - b >= minimo - 1e-9]


def buscar_pesos(hibrido: Hibrido, protocolo: Protocolo, k: int = config.K_PRINCIPAL,
                 lote: int = 500) -> tuple[pd.DataFrame, np.ndarray]:
    """NDCG@k de cada combinação de pesos (sem MMR): a grade e a matriz (combinação x aluno).

    Os componentes são pontuados uma vez por lote e só a soma ponderada muda entre combinações.
    """
    combinacoes = combinacoes_pesos()
    ndcg = np.zeros((len(combinacoes), len(protocolo.ids_alunos)))
    n_relevantes = np.array([len(r) for r in protocolo.relevantes])
    pesos_originais = dict(hibrido.pesos)
    for inicio in range(0, len(protocolo.ids_alunos), lote):
        fim = inicio + lote
        ids = protocolo.ids_alunos[inicio:fim]
        componentes = hibrido.componentes(ids, protocolo.ids_candidatos)
        elegivel = hibrido.conhecimento.elegibilidade(ids, protocolo.ids_candidatos)
        vistos = protocolo.vistos[inicio:fim].toarray()
        relevantes = protocolo.relevantes[inicio:fim]
        for c, pesos in enumerate(combinacoes):
            hibrido.pesos = dict(zip(COMPONENTES, pesos))
            pontuacoes = hibrido.combinar(componentes, hibrido.pesos_efetivos(ids), elegivel, ids,
                                          protocolo.ids_candidatos)
            listas = Recomendador.selecionar(hibrido, pontuacoes, protocolo.ids_candidatos, k, vistos,
                                             semente=int(ids[0]))
            ndcg[c, inicio:fim] = precisao_recall_ndcg(acertos(listas, relevantes), n_relevantes[inicio:fim], k)[2]
    hibrido.pesos = pesos_originais
    grade = pd.DataFrame(combinacoes, columns=list(COMPONENTES))
    grade[f"NDCG@{k}"] = ndcg.mean(axis=1)
    return grade, ndcg


def escolher_pesos(grade: pd.DataFrame, ndcg_por_aluno: np.ndarray, padrao: dict = config.PESOS_PADRAO,
                   reamostras: int = 2000) -> tuple[dict, dict]:
    """Troca os pesos padrão pela melhor combinação só se a vantagem for significativa.

    Compara, aluno a aluno, o NDCG da melhor combinação com o da combinação padrão
    (bootstrap pareado). Se o intervalo de 95% da diferença contém zero, a diferença
    é indistinguível do ruído e os pesos padrão (escolhidos por conhecimento do domínio)
    são mantidos — evita "aprender" pesos a partir de flutuação aleatória.
    """
    coluna = [c for c in grade.columns if c.startswith("NDCG")][0]
    melhor = int(grade[coluna].to_numpy().argmax())
    linha_padrao = int(np.flatnonzero(np.isclose(grade[list(COMPONENTES)].to_numpy(),
                                                 [padrao[c] for c in COMPONENTES]).all(axis=1))[0])
    diferenca = ndcg_por_aluno[melhor] - ndcg_por_aluno[linha_padrao]
    rng = np.random.default_rng(config.SEMENTE)
    medias = np.array([diferenca[rng.integers(0, len(diferenca), len(diferenca))].mean() for _ in range(reamostras)])
    intervalo = (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5)))
    significativo = melhor != linha_padrao and intervalo[0] > 0
    escolhida = melhor if significativo else linha_padrao
    decisao = {"melhor": {c: float(grade.at[melhor, c]) for c in COMPONENTES},
               f"{coluna} melhor": float(grade.at[melhor, coluna]),
               f"{coluna} padrão": float(grade.at[linha_padrao, coluna]),
               "diferenca_media": float(diferenca.mean()), "IC95_diferenca": intervalo,
               "significativo": bool(significativo),
               "escolha": "melhor da grade" if significativo else "padrão (diferença não significativa)"}
    return {c: float(grade.at[escolhida, c]) for c in COMPONENTES}, decisao


def comparacoes_pareadas(por_aluno: dict[str, dict], referencia: str = "Híbrido",
                         metricas=(f"NDCG@{config.K_PRINCIPAL}", "Obra P@3"), reamostras: int = 2000) -> pd.DataFrame:
    """Diferença média (referência − modelo), aluno a aluno, com IC de 95% por bootstrap pareado.

    Comparar intervalos de confiança separados é conservador e pode enganar; aqui cada
    reamostragem sorteia os mesmos alunos para os dois modelos. p = bilateral do bootstrap.
    """
    linhas = []
    for modelo, valores in por_aluno.items():
        if modelo == referencia:
            continue
        for metrica in metricas:
            diferenca = por_aluno[referencia][metrica] - valores[metrica]
            rng = np.random.default_rng(config.SEMENTE)
            medias = np.array([diferenca[rng.integers(0, len(diferenca), len(diferenca))].mean()
                               for _ in range(reamostras)])
            inferior, superior = np.percentile(medias, [2.5, 97.5])
            p = min(1.0, 2 * min((medias <= 0).mean(), (medias >= 0).mean()))
            linhas.append({"modelo": modelo, "metrica": metrica, "diferenca": float(diferenca.mean()),
                           "IC95 inf": float(inferior), "IC95 sup": float(superior), "p": float(p),
                           "significativo": bool(inferior > 0 or superior < 0)})
    return pd.DataFrame(linhas)


@dataclass
class ResultadoCenario:
    nome: str
    ranqueamento: pd.DataFrame
    notas: pd.DataFrame
    pesos: dict
    grade_pesos: pd.DataFrame
    coef_notas: dict
    historico_fc: pd.DataFrame
    curva_mmr: pd.DataFrame
    cold_start: pd.DataFrame | None
    comparacoes: pd.DataFrame
    info: dict
    por_aluno: dict = field(repr=False)
    componentes: Componentes = field(repr=False)
    hibrido: Hibrido = field(repr=False)
    protocolo: Protocolo = field(repr=False)
    treino: pd.DataFrame = field(repr=False)


def _tabela(resultados: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in resultados])


def _separar(interacoes: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """(treino jan–ago, validação set–out, treino jan–out, teste nov–dez), só com as colunas usadas."""
    dia = interacoes["dia"].to_numpy()
    teste = dia >= dia_de(config.DATA_CORTE_TESTE)
    validacao = (dia >= dia_de(config.DATA_CORTE_VALIDACAO)) & ~teste
    return (interacoes.loc[~teste & ~validacao, COLUNAS_TREINO], interacoes.loc[validacao, COLUNAS_TREINO],
            interacoes.loc[~teste, COLUNAS_TREINO], interacoes.loc[teste, COLUNAS_TREINO])


def executar_cenario(nome: str, base: Base, interacoes: pd.DataFrame, cold_start: bool = False,
                     lambdas_mmr=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), modelos_extras: tuple = (),
                     log=print) -> ResultadoCenario:
    """`modelos_extras` entram só na tabela de ranqueamento (ex.: o oráculo do cenário simulado)."""
    inicio_total = time.time()
    alunos = base.ids_alunos_cadastrados
    candidatos = base.ids_materiais_cadastrados
    treino_v, validacao, treino, teste = _separar(interacoes)
    log(f"[{nome}] treino {len(treino):,} | teste {len(teste):,} | validação {len(validacao):,}")

    # 1) Validação: ajustes que não podem olhar o teste.
    log("  validação (jan–ago -> set–out)")
    comp_v = treinar_componentes(base, treino_v, validacao=validacao, log=log)
    obra_do_material = comp_v.conhecimento.obra_do_material
    protocolo_v = montar_protocolo(treino_v, validacao, alunos, candidatos, obra_do_material)
    hibrido_v = comp_v.hibrido()
    grade, ndcg_grade = buscar_pesos(hibrido_v, protocolo_v)
    pesos, decisao_pesos = escolher_pesos(grade, ndcg_grade)
    grade = grade.sort_values(grade.columns[-1], ascending=False, ignore_index=True)
    log(f"    pesos: {pesos} -> {decisao_pesos['escolha']} "
        f"(melhor {decisao_pesos['melhor']}, diferença {decisao_pesos['diferenca_media']:+.5f}, "
        f"IC95 [{decisao_pesos['IC95_diferenca'][0]:+.5f}, {decisao_pesos['IC95_diferenca'][1]:+.5f}])")
    pares_v = validacao[base.mascara_cadastradas(validacao)]
    coef_notas = hibrido_v.ajustar_notas(pares_v["id_aluno"].to_numpy(), pares_v["id_material"].to_numpy(),
                                         pares_v["avaliacao"].to_numpy())
    coef_kb = ConhecimentoCalibrado(comp_v.conhecimento, comp_v.baseline).ajustar(pares_v).coef
    iteracoes = comp_v.colaborativo.melhor_iteracao
    historico_fc = pd.DataFrame(comp_v.colaborativo.historico)
    del comp_v, hibrido_v, protocolo_v, treino_v, validacao, pares_v, ndcg_grade
    gc.collect()

    # 2) Treino final e teste.
    log("  teste (jan–out -> nov–dez)")
    comp = treinar_componentes(base, treino, iteracoes_fc=iteracoes, log=log)
    hibrido = comp.hibrido(pesos=pesos)
    hibrido.coef_notas = coef_notas
    protocolo = montar_protocolo(treino, teste, alunos, candidatos, obra_do_material)
    vetores = comp.conteudo.vetores_itens
    modelos = [Aleatorio(), comp.popularidade, comp.conteudo, comp.colaborativo, comp.conhecimento, hibrido]
    resultados = []
    for modelo in [*modelos, *modelos_extras]:
        resultados.append(avaliar_ranqueamento(modelo, protocolo, vetores, obra_do_material))
        log(f"    {modelo.nome:<14} P@10={resultados[-1]['P@10']:.5f}  NDCG@10={resultados[-1]['NDCG@10']:.5f}  "
            f"Obra P@3={resultados[-1]['Obra P@3']:.4f}  Div@10={resultados[-1]['Diversidade@10']:.3f}")

    curva = []
    resultado_hibrido = resultados[len(modelos) - 1]
    for lambda_ in lambdas_mmr:
        hibrido.lambda_mmr = lambda_
        r = resultado_hibrido if lambda_ == config.LAMBDA_MMR else avaliar_ranqueamento(hibrido, protocolo, vetores,
                                                                                       obra_do_material)
        curva.append({"lambda": lambda_, "P@10": r["P@10"], "NDCG@10": r["NDCG@10"],
                      "Diversidade@10": r["Diversidade@10"], "Cobertura@10": r["Cobertura@10"]})
    hibrido.lambda_mmr = config.LAMBDA_MMR

    pares = teste[base.mascara_cadastradas(teste)]
    calibrado = ConhecimentoCalibrado(comp.conhecimento, comp.baseline, coef_kb)
    media = MediaGlobal().treinar(treino)
    notas = [avaliar_notas(m, pares) | {"pares": "aluno e material cadastrados"}
             for m in (media, comp.baseline, comp.conteudo, comp.colaborativo, calibrado, hibrido)]
    if len(teste) > len(pares):
        notas += [avaliar_notas(m, teste) | {"pares": "todo o teste"} for m in (media, comp.baseline, comp.colaborativo)]

    tabela_cold = None
    if cold_start:
        tabela_cold = executar_cold_start(base, treino, teste, pesos, iteracoes, obra_do_material, log=log)

    info = {"n_treino": len(treino), "n_teste": len(teste), "n_pares_notas": len(pares),
            "n_alunos_avaliados": len(protocolo.ids_alunos), "n_candidatos": len(candidatos),
            "iteracoes_fc": iteracoes, "decisao_pesos": decisao_pesos, "segundos": round(time.time() - inicio_total, 1)}
    log(f"  concluído em {info['segundos']:.0f}s")
    por_aluno = {r["modelo"]: r["_por_aluno"] for r in resultados}
    return ResultadoCenario(nome, _tabela(resultados), pd.DataFrame(notas), pesos, grade,
                            dict(zip(["colaborativo", "conteudo", "conhecimento", "intercepto"], coef_notas)),
                            historico_fc, pd.DataFrame(curva), tabela_cold, comparacoes_pareadas(por_aluno), info,
                            por_aluno, comp, hibrido, protocolo, treino)


def executar_cold_start(base: Base, treino: pd.DataFrame, teste: pd.DataFrame, pesos: dict, iteracoes: int,
                        obra_do_material: np.ndarray, fracao: float = 0.2, log=print) -> pd.DataFrame:
    """Sorteia `fracao` dos alunos cadastrados, apaga todo o histórico de treino deles e
    avalia os modelos só nesses alunos — como se tivessem acabado de entrar na plataforma."""
    rng = np.random.default_rng(config.SEMENTE)
    alunos = base.ids_alunos_cadastrados
    novos = np.sort(rng.choice(alunos, size=int(len(alunos) * fracao), replace=False))
    log(f"  cold start: {len(novos):,} alunos sem histórico")
    treino_reduzido = treino[~treino["id_aluno"].isin(novos).to_numpy()]
    comp = treinar_componentes(base, treino_reduzido, iteracoes_fc=iteracoes, log=log)
    hibrido = comp.hibrido(pesos=pesos)
    protocolo = montar_protocolo(treino_reduzido, teste, novos, base.ids_materiais_cadastrados, obra_do_material)
    modelos = [Aleatorio(), comp.popularidade, comp.conteudo, comp.colaborativo, comp.conhecimento, hibrido]
    return _tabela([avaliar_ranqueamento(m, protocolo, comp.conteudo.vetores_itens, obra_do_material)
                    for m in modelos])

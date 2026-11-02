"""Análise exploratória e testes estatísticos de qualidade dos dados.

A pergunta central é: as interações carregam algum sinal que um recomendador
possa aprender? Cada teste abaixo mede um tipo de sinal diferente:
- integridade referencial entre as três bases;
- distribuição de cada coluna (aderência à uniforme);
- dependência entre avaliação e as demais variáveis;
- efeito de aluno e de material (ANOVA) e sua estabilidade no tempo (meia-divisão).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .dados import Base, dia_de


def _qui_quadrado_aderencia(contagens: np.ndarray) -> dict:
    """H0: todas as categorias são igualmente prováveis."""
    contagens = np.asarray(contagens, dtype=float)
    estatistica, p = stats.chisquare(contagens)
    return {"qui2": float(estatistica), "gl": int(len(contagens) - 1), "p": float(p)}


def _qui_quadrado_independencia(tabela: np.ndarray) -> dict:
    tabela = np.asarray(tabela, dtype=float)
    estatistica, p, gl, _ = stats.chi2_contingency(tabela, correction=False)
    n = tabela.sum()
    v_cramer = np.sqrt(estatistica / (n * (min(tabela.shape) - 1)))
    return {"qui2": float(estatistica), "gl": int(gl), "p": float(p), "v_cramer": float(v_cramer)}


def _anova_um_fator(grupos: np.ndarray, valores: np.ndarray) -> dict:
    """ANOVA de um fator calculada por somas agregadas (viável para 100 mil grupos).

    eta² = fração da variância das avaliações explicada pelo grupo (aluno ou material).
    """
    valores = valores.astype(np.float64)
    n = np.bincount(grupos)
    soma = np.bincount(grupos, weights=valores)
    usados = n > 0
    n, soma = n[usados], soma[usados]
    media_global = valores.mean()
    ss_total = float(((valores - media_global) ** 2).sum())
    ss_entre = float((soma ** 2 / n).sum() - len(valores) * media_global ** 2)
    k, total = len(n), len(valores)
    gl_entre, gl_dentro = k - 1, total - k
    f = (ss_entre / gl_entre) / ((ss_total - ss_entre) / gl_dentro)
    return {"F": float(f), "gl_entre": gl_entre, "gl_dentro": gl_dentro,
            "p": float(stats.f.sf(f, gl_entre, gl_dentro)), "eta2": ss_entre / ss_total}


def _correlacao_meias(grupos: np.ndarray, valores: np.ndarray, primeira_metade: np.ndarray,
                      minimo: int = 20) -> dict:
    """Correlação entre a média de cada grupo no 1º e no 2º semestre.

    Se alunos (ou materiais) tivessem gosto (ou qualidade) estável, quem avalia
    acima da média em jan–jun também avaliaria acima da média em jul–dez.
    """
    medias, contagens = [], []
    for mascara in (primeira_metade, ~primeira_metade):
        n = np.bincount(grupos[mascara])
        s = np.bincount(grupos[mascara], weights=valores[mascara].astype(np.float64))
        contagens.append(n)
        medias.append(s / np.maximum(n, 1))
    tamanho = min(len(contagens[0]), len(contagens[1]))
    validos = (contagens[0][:tamanho] >= minimo) & (contagens[1][:tamanho] >= minimo)
    a, b = medias[0][:tamanho][validos], medias[1][:tamanho][validos]
    r, p = stats.pearsonr(a, b)
    return {"r": float(r), "p": float(p), "n_grupos": int(validos.sum()), "minimo_por_metade": minimo}


def integridade(base: Base) -> dict:
    inter = base.interacoes
    aluno_ok = inter["id_aluno"].isin(base.alunos.index).to_numpy()
    material_ok = inter["id_material"].isin(base.materiais.index).to_numpy()
    n = len(inter)
    return {
        "n_interacoes": n,
        "n_alunos_cadastrados": len(base.alunos),
        "n_materiais_cadastrados": len(base.materiais),
        "alunos_distintos_nas_interacoes": int(inter["id_aluno"].nunique()),
        "materiais_distintos_nas_interacoes": int(inter["id_material"].nunique()),
        "id_aluno_max": int(inter["id_aluno"].max()),
        "id_material_max": int(inter["id_material"].max()),
        "com_aluno_cadastrado": int(aluno_ok.sum()),
        "com_material_cadastrado": int(material_ok.sum()),
        "com_ambos_cadastrados": int((aluno_ok & material_ok).sum()),
        "ambos_esperado_se_independente": float(aluno_ok.mean() * material_ok.mean() * n),
        "valores_ausentes": int(inter.isna().sum().sum()),
    }


def catalogo_e_perfis(base: Base) -> dict:
    obras = (base.materiais.groupby("id_obra")
             .agg(titulo=("titulo", "first"), tipo=("tipo", "first"), area=("area", "first"),
                  nivel=("nivel", "first"), autor=("autor", "first"), copias=("id_material", "size"))
             .reset_index())
    perfis = (base.alunos.groupby("id_perfil")
              .agg(curso=("curso", "first"), periodo=("periodo", "first"),
                   disciplinas=("disciplinas_cursadas", "first"), interesses=("areas_interesse", "first"),
                   alunos=("id_aluno", "size"))
              .reset_index())
    return {"n_obras": len(obras), "n_perfis": len(perfis), "obras": obras, "perfis": perfis}


def distribuicoes(base: Base) -> dict:
    inter = base.interacoes
    avaliacao = inter["avaliacao"].to_numpy()
    duracao = inter["duracao_minutos"].to_numpy()
    dia = inter["dia"].to_numpy()
    cont_aval = np.bincount(avaliacao, minlength=6)[1:]
    cont_tipo = inter["tipo_interacao"].value_counts().sort_index()
    faixa_dur = np.arange(duracao.min(), duracao.max() + 1)
    cont_dur = np.bincount(duracao - duracao.min(), minlength=len(faixa_dur))
    cont_dia = np.bincount(dia - dia.min())
    por_aluno = np.bincount(inter["id_aluno"].to_numpy())[1:]
    por_material = np.bincount(inter["id_material"].to_numpy())[1:]
    meses = pd.to_datetime(dia, unit="D").month
    return {
        "avaliacao": {"contagens": cont_aval.tolist(), **_qui_quadrado_aderencia(cont_aval)},
        "tipo_interacao": {"contagens": cont_tipo.to_dict(), **_qui_quadrado_aderencia(cont_tipo.to_numpy())},
        "duracao": {"min": int(duracao.min()), "max": int(duracao.max()), "media": float(duracao.mean()),
                    "desvio": float(duracao.std()), **_qui_quadrado_aderencia(cont_dur)},
        "data": {"inicio": str(pd.to_datetime(dia.min(), unit="D").date()),
                 "fim": str(pd.to_datetime(dia.max(), unit="D").date()),
                 "por_mes": np.bincount(meses, minlength=13)[1:].tolist(), **_qui_quadrado_aderencia(cont_dia)},
        # Índice de dispersão = variância / média; ~1 quando a atividade é Poisson (sorteio uniforme).
        "interacoes_por_aluno": {"media": float(por_aluno.mean()), "min": int(por_aluno.min()),
                                 "max": int(por_aluno.max()), "dispersao": float(por_aluno.var() / por_aluno.mean())},
        "interacoes_por_material": {"media": float(por_material.mean()), "min": int(por_material.min()),
                                    "max": int(por_material.max()),
                                    "dispersao": float(por_material.var() / por_material.mean())},
    }


def sinal(base: Base, semente: int = 0) -> dict:
    inter = base.interacoes
    avaliacao = inter["avaliacao"].to_numpy()
    resultado = {}

    tabela_tipo = pd.crosstab(inter["tipo_interacao"], inter["avaliacao"]).to_numpy()
    resultado["avaliacao_x_tipo"] = _qui_quadrado_independencia(tabela_tipo)
    meses = pd.to_datetime(inter["dia"].to_numpy(), unit="D").month
    resultado["avaliacao_x_mes"] = _qui_quadrado_independencia(pd.crosstab(meses, avaliacao).to_numpy())

    amostra = np.random.default_rng(semente).choice(len(inter), size=min(2_000_000, len(inter)), replace=False)
    rho, p = stats.spearmanr(avaliacao[amostra], inter["duracao_minutos"].to_numpy()[amostra])
    resultado["avaliacao_x_duracao"] = {"spearman_rho": float(rho), "p": float(p), "n_amostra": len(amostra)}

    resultado["efeito_aluno"] = _anova_um_fator(inter["id_aluno"].to_numpy(), avaliacao)
    resultado["efeito_material"] = _anova_um_fator(inter["id_material"].to_numpy(), avaliacao)
    primeira_metade = inter["dia"].to_numpy() < dia_de("2023-07-01")
    resultado["estabilidade_aluno"] = _correlacao_meias(inter["id_aluno"].to_numpy(), avaliacao, primeira_metade)
    resultado["estabilidade_material"] = _correlacao_meias(inter["id_material"].to_numpy(), avaliacao,
                                                           primeira_metade)

    # Sinal entre perfil acadêmico e conteúdo (só onde aluno e material existem nos cadastros).
    sub = inter[base.mascara_cadastradas()]
    perfil = base.alunos["id_perfil"].reindex(sub["id_aluno"]).to_numpy()
    obra = base.materiais["id_obra"].reindex(sub["id_material"]).to_numpy()
    celula = perfil.astype(np.int64) * (obra.max() + 1) + obra
    resultado["avaliacao_x_perfil_obra"] = _anova_um_fator(pd.factorize(celula)[0], sub["avaliacao"].to_numpy())
    escolhas = pd.crosstab(perfil, obra).to_numpy()
    resultado["escolha_obra_x_perfil"] = _qui_quadrado_independencia(escolhas)
    resultado["media_perfil_obra"] = (pd.DataFrame({"perfil": perfil, "obra": obra, "avaliacao": sub["avaliacao"]})
                                      .pivot_table(index="perfil", columns="obra", values="avaliacao",
                                                   aggfunc="mean"))

    pares = inter["id_aluno"].to_numpy().astype(np.int64) * (base.n_itens + 1) + inter["id_material"].to_numpy()
    _, repeticoes = np.unique(pares, return_counts=True)
    n_alunos = inter["id_aluno"].nunique()
    n_materiais = inter["id_material"].nunique()
    esperado = len(inter) ** 2 / (2 * n_alunos * n_materiais)
    resultado["pares_repetidos"] = {"observado": int((repeticoes > 1).sum()),
                                    "esperado_se_aleatorio": float(esperado)}
    return resultado


def analisar(base: Base) -> dict:
    return {"integridade": integridade(base), "catalogo": catalogo_e_perfis(base),
            "distribuicoes": distribuicoes(base), "sinal": sinal(base)}

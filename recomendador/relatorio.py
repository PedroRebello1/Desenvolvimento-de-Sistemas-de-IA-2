"""Gera o relatório a partir de `resultados/`.

1. Tabelas Markdown (formato brasileiro) em relatorio/tabelas/, incluídas no .qmd.
2. relatorio/variaveis.yml com os números citados no texto ({{< meta ... >}}).
3. Renderização com o Quarto local: relatorio/Relatorio.docx e relatorio/Relatorio.pdf (Typst).
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pandas as pd

from . import config, ontologia
from .figuras import br

DIR_RELATORIO = config.RAIZ / "relatorio"
DIR_TABELAS = DIR_RELATORIO / "tabelas"
ORDEM = ["Aleatório", "Popularidade", "Conteúdo", "Colaborativo", "Conhecimento", "Híbrido", "Oráculo (teto)"]


def _ler(nome: str) -> pd.DataFrame:
    return pd.read_csv(config.DIR_RESULTADOS / nome)


def _json(nome: str) -> dict:
    return json.loads((config.DIR_RESULTADOS / nome).read_text(encoding="utf-8"))


def _comprimento(texto: str) -> float:
    """Largura aproximada em caracteres médios: maiúsculas, @ e % são mais largos, o negrito
    ocupa ~8% a mais e o código (monoespaçado) ~25% a mais."""
    negrito = "**" in texto
    texto = texto.replace("**", "").replace("~", "")
    codigo = texto.startswith("`")
    largura = sum(1.4 if (c.isupper() or c in "@%") else 1.0 for c in texto.replace("`", ""))
    return largura * (1.25 if codigo else 1.0) * (1.08 if negrito else 1.0)


def _larguras(textos: list[list[str]], total: int = 96, folga: int = 3) -> list[int]:
    """Larguras relativas das colunas, como num layout de tabela: cada coluna recebe ao menos
    a palavra mais longa (não dá para quebrar uma palavra), e o espaço restante vai para as
    colunas cujo texto completo é mais longo que isso, que então quebram linha.
    Só o espaço comum separa palavras: o espaço não separável mantém "Obra P@3" junto."""
    n = len(textos[0])
    minimo = [max(max((_comprimento(p) for p in linha[j].split(" ") if p), default=1) for linha in textos) + folga
              for j in range(n)]
    maximo = [max(_comprimento(linha[j]) for linha in textos) + folga for j in range(n)]
    sobra = total - sum(minimo)
    extra = [mx - mn for mx, mn in zip(maximo, minimo)]
    if sobra <= 0 or sum(extra) == 0:
        return [max(1, round(m)) for m in minimo]
    return [max(1, round(mn + min(e, sobra * e / sum(extra)))) for mn, e in zip(minimo, extra)]


def _markdown(tabela: pd.DataFrame, alinhamento: list[str] | None = None) -> str:
    """Tabela pipe do Pandoc. Em tabelas largas, o Pandoc usa o número de traços da linha
    separadora como largura relativa das colunas; aqui ele vem de `_larguras`."""
    alinhamento = alinhamento or ["l"] + ["r"] * (tabela.shape[1] - 1)
    textos = [[str(c) for c in tabela.columns]] + [[str(v) for v in linha] for linha in tabela.itertuples(index=False)]
    textos = [[c.replace("Obra P@3", "Obra P@3") for c in linha] for linha in textos]
    larguras = [max(n, 3) for n in _larguras(textos)]

    def separador(a, n):
        return {"l": ":" + "-" * (n - 1), "r": "-" * (n - 1) + ":", "c": ":" + "-" * (n - 2) + ":"}[a]

    linhas = ["| " + " | ".join(textos[0]) + " |",
              "| " + " | ".join(separador(a, n) for a, n in zip(alinhamento, larguras)) + " |"]
    linhas += ["| " + " | ".join(linha) + " |" for linha in textos[1:]]
    return "\n".join(linhas) + "\n"


def _salvar(nome: str, tabela: pd.DataFrame, legenda: str, alinhamento: list[str] | None = None):
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    (DIR_TABELAS / f"{nome}.md").write_text(_markdown(tabela, alinhamento) + f"\n: {legenda}\n", encoding="utf-8")


def _p(valor: float) -> str:
    return "< 0,001" if valor < 0.001 else br(valor, 3)


def _ordenar(tabela: pd.DataFrame) -> pd.DataFrame:
    return tabela.set_index("modelo").reindex([m for m in ORDEM if m in set(tabela["modelo"])]).reset_index()


def _negrito_hibrido(tabela: pd.DataFrame) -> pd.DataFrame:
    tabela = tabela.copy()
    linha = tabela["Modelo"] == "Híbrido"
    tabela.loc[linha] = tabela.loc[linha].map(lambda v: f"**{v}**")
    return tabela


def tabelas_dados():
    q = _json("qualidade_dados.json")
    i, d, s = q["integridade"], q["distribuicoes"], q["sinal"]
    n = i["n_interacoes"]
    integridade = pd.DataFrame([
        ("Todas as interações", br(n, 0), "100%"),
        ("Com aluno cadastrado", br(i["com_aluno_cadastrado"], 0), f"{br(100 * i['com_aluno_cadastrado'] / n, 1)}%"),
        ("Com material cadastrado", br(i["com_material_cadastrado"], 0),
         f"{br(100 * i['com_material_cadastrado'] / n, 1)}%"),
        ("Com aluno **e** material cadastrados", br(i["com_ambos_cadastrados"], 0),
         f"{br(100 * i['com_ambos_cadastrados'] / n, 2)}%"),
        ("Esperado para os dois, se independentes", br(i["ambos_esperado_se_independente"], 0),
         f"{br(100 * i['ambos_esperado_se_independente'] / n, 2)}%"),
    ], columns=["Recorte", "Interações", "Parcela"])
    _salvar("integridade", integridade, "Integridade referencial de interacoes.csv {#tbl-integridade}")

    testes = pd.DataFrame([
        ("Notas uniformes em 1–5", "χ² aderência", f"χ² = {br(d['avaliacao']['qui2'], 1)}", _p(d["avaliacao"]["p"]),
         "uniforme"),
        ("Tipo de interação 50/50", "χ² aderência", f"χ² = {br(d['tipo_interacao']['qui2'], 2)}",
         _p(d["tipo_interacao"]["p"]), "uniforme"),
        ("Duração uniforme em 5–240 min", "χ² aderência", f"χ² = {br(d['duracao']['qui2'], 0)}",
         _p(d["duracao"]["p"]), "uniforme"),
        ("Datas uniformes em 2023", "χ² aderência", f"χ² = {br(d['data']['qui2'], 0)}", _p(d["data"]["p"]),
         "uniforme"),
        ("Nota × tipo de interação", "χ² independência", f"V = {br(s['avaliacao_x_tipo']['v_cramer'], 4)}",
         _p(s["avaliacao_x_tipo"]["p"]), "independentes"),
        ("Nota × mês", "χ² independência", f"V = {br(s['avaliacao_x_mes']['v_cramer'], 4)}",
         _p(s["avaliacao_x_mes"]["p"]), "independentes"),
        ("Nota × duração", "Spearman (2 mi)", f"ρ = {br(s['avaliacao_x_duracao']['spearman_rho'], 4)}",
         _p(s["avaliacao_x_duracao"]["p"]), "sem correlação"),
        ("Efeito do aluno na nota", "ANOVA, 100 mil grupos", f"F = {br(s['efeito_aluno']['F'], 3)}",
         _p(s["efeito_aluno"]["p"]), "nenhum"),
        ("Efeito do material na nota", "ANOVA, 50 mil grupos", f"F = {br(s['efeito_material']['F'], 3)}",
         _p(s["efeito_material"]["p"]), "nenhum"),
        ("Média do aluno, 1º × 2º semestre", "Pearson", f"r = {br(s['estabilidade_aluno']['r'], 4)}",
         _p(s["estabilidade_aluno"]["p"]), "sem estabilidade"),
        ("Média do material, 1º × 2º semestre", "Pearson", f"r = {br(s['estabilidade_material']['r'], 4)}",
         _p(s["estabilidade_material"]["p"]), "sem estabilidade"),
        ("Nota × (perfil, obra)", "ANOVA, 80 células", f"F = {br(s['avaliacao_x_perfil_obra']['F'], 3)}",
         _p(s["avaliacao_x_perfil_obra"]["p"]), "nenhum"),
        ("Escolha da obra × perfil", "χ² independência", f"V = {br(s['escolha_obra_x_perfil']['v_cramer'], 4)}",
         _p(s["escolha_obra_x_perfil"]["p"]), "independentes"),
    ], columns=["Hipótese testada", "Teste", "Estatística", "p", "Conclusão"])
    _salvar("testes", testes, "Testes estatísticos sobre interacoes.csv (10 milhões de linhas) {#tbl-testes}",
            ["l", "l", "r", "r", "l"])

    obras = _ler("obras.csv")
    linhas = []
    for o in obras.itertuples():
        regra = ontologia.regra_da_obra(o.titulo)
        requisitos = " e ".join("(" + " ou ".join(sorted(g)) + ")" if len(g) > 1 else next(iter(g))
                                for g in regra.prerequisitos) or "—"
        linhas.append((o.titulo, o.tipo, o.area, o.nivel, o.autor, requisitos))
    copias = br(int(obras["copias"].iloc[0]), 0) if obras["copias"].nunique() == 1 else "várias"
    _salvar("obras", pd.DataFrame(linhas, columns=["Obra", "Tipo", "Área", "Nível", "Autor", "Pré-requisitos"]),
            f"As 10 obras do catálogo ({copias} cópias de cada) e os pré-requisitos da base de conhecimento "
            "{#tbl-obras}", ["l"] * 6)

    perfis = _ler("perfis.csv")
    _salvar("perfis", pd.DataFrame({"Curso": perfis["curso"], "Período": perfis["periodo"].astype(str) + "º",
                                    "Disciplinas cursadas": perfis["disciplinas"],
                                    "Interesses": perfis["interesses"], "Alunos": perfis["alunos"].map(lambda v: br(v, 0))}),
            "Os 8 perfis acadêmicos distintos de dados_alunos.csv {#tbl-perfis}", ["l", "r", "l", "l", "r"])


def _tabela_ranqueamento(prefixo: str, legenda: str):
    r = _ordenar(_ler(f"{prefixo}_ranqueamento.csv"))
    tabela = pd.DataFrame({"Modelo": r["modelo"],
                           "P@10": r["P@10"].map(lambda v: br(v, 4)), "R@10": r["R@10"].map(lambda v: br(v, 4)),
                           "F1@10": r["F1@10"].map(lambda v: br(v, 4)),
                           "NDCG@10": r["NDCG@10"].map(lambda v: br(v, 4)),
                           "IC 95%": [f"{br(a, 4)}–{br(b, 4)}" for a, b in
                                      zip(r["NDCG@10 IC95 inf"], r["NDCG@10 IC95 sup"])],
                           "Obra P@3": r["Obra P@3"].map(lambda v: br(v, 3)),
                           "Cobertura": r["Cobertura@10"].map(lambda v: f"{br(100 * v, 1)}%"),
                           "Diversidade": r["Diversidade@10"].map(lambda v: br(v, 2))})
    _salvar(f"{prefixo}_ranqueamento", _negrito_hibrido(tabela), legenda)


def _tabela_k(prefixo: str, legenda: str):
    r = _ordenar(_ler(f"{prefixo}_ranqueamento.csv"))
    colunas = {"Modelo": r["modelo"]}
    for k in config.LISTA_K:
        for metrica in ("P", "R", "F1"):
            colunas[f"{metrica}@{k}"] = r[f"{metrica}@{k}"].map(lambda v: br(v, 4))
    _salvar(f"{prefixo}_ranqueamento_k", _negrito_hibrido(pd.DataFrame(colunas)), legenda)


ROTULO_NOTAS = {"Vieses (μ + b_u + b_i)": "Vieses μ + b~u~ + b~i~"}


def _tabela_notas(prefixo: str, legenda: str):
    """Uma linha por modelo; colunas de RMSE e MAE por recorte de pares do teste."""
    n = _ler(f"{prefixo}_notas.csv")
    recortes = list(dict.fromkeys(n["pares"]))
    nomes = {"aluno e material cadastrados": "cadastrados", "todo o teste": "todo o teste"}
    tabela = pd.DataFrame({"Modelo": list(dict.fromkeys(n["modelo"]))})
    for recorte in recortes:
        parte = n[n["pares"] == recorte].set_index("modelo")
        for metrica in ("RMSE", "MAE"):
            coluna = f"{metrica} ({nomes[recorte]})" if len(recortes) > 1 else metrica
            tabela[coluna] = [br(parte.at[m, metrica], 4) if m in parte.index else "—" for m in tabela["Modelo"]]
    tamanhos = " e ".join(f"{br(n.loc[n['pares'] == r, 'n_pares'].iloc[0], 0)} pares ({nomes[r]})" for r in recortes)
    tabela["Modelo"] = tabela["Modelo"].replace(ROTULO_NOTAS)
    _salvar(f"{prefixo}_notas", _negrito_hibrido(tabela), legenda.replace("{#", f"— {tamanhos} {{#"))


CASAS_METRICA = {"NDCG@10": 4, "Obra P@3": 3}
CASAS_DIFERENCA = {"NDCG@10": 5, "Obra P@3": 4}
CHAVE_MODELO = {"Aleatório": "aleatorio", "Popularidade": "popularidade", "Conteúdo": "conteudo",
                "Colaborativo": "colaborativo", "Conhecimento": "conhecimento", "Oráculo (teto)": "oraculo"}
CHAVE_METRICA = {"NDCG@10": "ndcg10", "Obra P@3": "obra_p3"}


def _com_sinal(valor: float, casas: int) -> str:
    texto = br(valor, casas).replace("-", "−")
    return texto if texto.startswith("−") or float(valor) == 0 else "+" + texto


def _tabela_comparacoes(prefixo: str, legenda: str):
    c = _ler(f"{prefixo}_comparacoes.csv")
    c = c[c["modelo"] != "Oráculo (teto)"]
    c = c.assign(ordem=c["modelo"].map(ORDEM.index)).sort_values(["metrica", "ordem"])
    tabela = pd.DataFrame({
        "Híbrido menos": c["modelo"], "Métrica": c["metrica"],
        "Diferença média": [_com_sinal(v, CASAS_DIFERENCA[m]) for v, m in zip(c["diferenca"], c["metrica"])],
        "IC 95% pareado": [f"{_com_sinal(a, CASAS_DIFERENCA[m])} a {_com_sinal(b, CASAS_DIFERENCA[m])}"
                           for a, b, m in zip(c["IC95 inf"], c["IC95 sup"], c["metrica"])],
        "p": c["p"].map(_p), "Significativa": c["significativo"].map({True: "sim", False: "não"})})
    _salvar(f"{prefixo}_comparacoes", tabela, legenda, ["l", "l", "r", "r", "r", "c"])


def tabelas_resultados():
    _tabela_comparacoes("real", "Comparações pareadas do híbrido com cada modelo nos dados fornecidos "
                                "{#tbl-real-comparacoes}")
    _tabela_comparacoes("controlado", "Comparações pareadas do híbrido com cada modelo no cenário controlado "
                                      "{#tbl-controlado-comparacoes}")
    _tabela_ranqueamento("real", "Ranqueamento nos dados fornecidos (teste nov–dez/2023; 10 mil materiais candidatos) "
                                 "{#tbl-real}")
    _tabela_ranqueamento("controlado", "Ranqueamento no cenário com sinal controlado {#tbl-controlado}")
    _tabela_k("real", "Precisão, recall e F1 para K = 5, 10 e 20 nos dados fornecidos {#tbl-real-k}")
    _tabela_k("controlado", "Precisão, recall e F1 para K = 5, 10 e 20 no cenário controlado {#tbl-controlado-k}")
    _tabela_notas("real", "Erro na previsão de notas nos dados fornecidos {#tbl-real-notas}")
    _tabela_notas("controlado", "Erro na previsão de notas no cenário controlado {#tbl-controlado-notas}")

    cold = _ordenar(_ler("controlado_cold_start.csv"))
    todos = _ordenar(_ler("controlado_ranqueamento.csv")).set_index("modelo")
    tabela = pd.DataFrame({"Modelo": cold["modelo"],
                           "NDCG@10 sem histórico": cold["NDCG@10"].map(lambda v: br(v, 4)),
                           "Obra P@3 sem histórico": cold["Obra P@3"].map(lambda v: br(v, 3)),
                           "Obra P@3 com histórico": [br(todos.at[m, "Obra P@3"], 3) for m in cold["modelo"]],
                           "Cobertura sem histórico": cold["Cobertura@10"].map(lambda v: f"{br(100 * v, 1)}%")})
    _salvar("controlado_cold_start", _negrito_hibrido(tabela),
            "Cold start: 2.000 alunos com todo o histórico de treino apagado {#tbl-cold}")

    curva = _ler("controlado_curva_mmr.csv")
    _salvar("controlado_curva_mmr", pd.DataFrame({
        "λ": curva["lambda"].map(lambda v: br(v, 1)), "P@10": curva["P@10"].map(lambda v: br(v, 4)),
        "NDCG@10": curva["NDCG@10"].map(lambda v: br(v, 4)),
        "Diversidade@10": curva["Diversidade@10"].map(lambda v: br(v, 3)),
        "Cobertura@10": curva["Cobertura@10"].map(lambda v: f"{br(100 * v, 1)}%")}),
        "Efeito do λ do MMR no híbrido (cenário controlado) {#tbl-mmr}")

    linhas = []
    for prefixo, nome in (("real", "Dados fornecidos"), ("controlado", "Sinal controlado")):
        resumo = _json(f"{prefixo}_resumo.json")
        decisao = resumo["decisao_pesos"]
        melhor = decisao["melhor"]
        linhas.append((nome, " / ".join(br(melhor[c], 1) for c in ("conteudo", "colaborativo", "conhecimento")),
                       f"{_com_sinal(decisao['diferenca_media'], 5)} ({_com_sinal(decisao['IC95_diferenca'][0], 5)} "
                       f"a {_com_sinal(decisao['IC95_diferenca'][1], 5)})",
                       " / ".join(br(resumo["pesos"][c], 1) for c in ("conteudo", "colaborativo", "conhecimento")),
                       str(resumo["iteracoes_fc"])))
    _salvar("pesos", pd.DataFrame(linhas, columns=["Cenário", "Melhor da grade", "Ganho de NDCG@10 (IC 95%)",
                                                   "Pesos adotados", "Iterações da FC"]),
            "Escolhas feitas na validação (set–out): pesos do híbrido (conteúdo / colaborativo / conhecimento) "
            "e iterações da FC {#tbl-pesos}", ["l", "r", "r", "r", "r"])

    ab = _ordenar(_ler("online_ab.csv"))
    tabela = pd.DataFrame({"Modelo": ab["modelo"],
                           "CTR": [f"{br(100 * c, 1)}% ± {br(100 * e, 1)}" for c, e in zip(ab["CTR"], ab["CTR IC95"])],
                           "Minutos por sessão": [f"{br(m, 0)} ± {br(e, 0)}" for m, e in
                                                  zip(ab["minutos por sessão"], ab["minutos IC95"])],
                           "Satisfação (1–5)": [f"{br(m, 2)} ± {br(e, 2)}" for m, e in
                                                zip(ab["satisfação (1–5)"], ab["satisfação IC95"])],
                           "Respostas": ab["respostas"].map(lambda v: br(v, 0))})
    _salvar("online_ab", _negrito_hibrido(tabela),
            "Teste A/B simulado: métricas online por modelo (± = meia-largura do IC 95%) {#tbl-ab}")


def variaveis() -> dict:
    """Números citados no texto do relatório, já formatados."""
    q = _json("qualidade_dados.json")
    v = {"integridade": {k: br(val, 0) if isinstance(val, (int, float)) else val
                         for k, val in q["integridade"].items()}}
    v["integridade"]["pct_ambos"] = br(100 * q["integridade"]["com_ambos_cadastrados"] / q["integridade"]["n_interacoes"], 1)
    s = q["sinal"]
    v["sinal"] = {"F_aluno": br(s["efeito_aluno"]["F"], 3), "p_aluno": br(s["efeito_aluno"]["p"], 2),
                  "F_material": br(s["efeito_material"]["F"], 3), "p_material": br(s["efeito_material"]["p"], 2),
                  "eta2_aluno": br(s["efeito_aluno"]["eta2"], 4), "eta2_material": br(s["efeito_material"]["eta2"], 4),
                  "r_aluno": br(s["estabilidade_aluno"]["r"], 4), "r_material": br(s["estabilidade_material"]["r"], 4),
                  "F_perfil_obra": br(s["avaliacao_x_perfil_obra"]["F"], 2),
                  "p_perfil_obra": br(s["avaliacao_x_perfil_obra"]["p"], 2),
                  "pares_obs": br(s["pares_repetidos"]["observado"], 0),
                  "pares_esp": br(s["pares_repetidos"]["esperado_se_aleatorio"], 0)}
    for prefixo in ("real", "controlado"):
        r = _ler(f"{prefixo}_ranqueamento.csv").set_index("modelo")
        n = _ler(f"{prefixo}_notas.csv")
        n = n[n["pares"] == "aluno e material cadastrados"].set_index("modelo")
        resumo = _json(f"{prefixo}_resumo.json")
        bloco = {"n_alunos": br(resumo["n_alunos_avaliados"], 0), "n_treino": br(resumo["n_treino"], 0),
                 "n_teste": br(resumo["n_teste"], 0), "iteracoes_fc": str(resumo["iteracoes_fc"]),
                 "pesos": " / ".join(br(resumo["pesos"][c], 1) for c in ("conteudo", "colaborativo", "conhecimento"))}
        for modelo, chave in (("Aleatório", "aleatorio"), ("Popularidade", "popularidade"), ("Conteúdo", "conteudo"),
                              ("Colaborativo", "colaborativo"), ("Conhecimento", "conhecimento"),
                              ("Híbrido", "hibrido"), ("Oráculo (teto)", "oraculo")):
            if modelo in r.index:
                bloco[chave] = {"p10": br(r.at[modelo, "P@10"], 4), "ndcg10": br(r.at[modelo, "NDCG@10"], 4),
                                "obra_p3": br(r.at[modelo, "Obra P@3"], 3),
                                "cobertura": br(100 * r.at[modelo, "Cobertura@10"], 1),
                                "diversidade": br(r.at[modelo, "Diversidade@10"], 2)}
        for modelo, chave in (("Média global", "media"), ("Vieses (μ + b_u + b_i)", "vieses"), ("Conteúdo", "conteudo"),
                              ("Colaborativo", "colaborativo"), ("Conhecimento", "conhecimento"), ("Híbrido", "hibrido")):
            bloco.setdefault(chave, {})["rmse"] = br(n.at[modelo, "RMSE"], 3)
        componentes = ["Conteúdo", "Colaborativo", "Conhecimento"]
        for metrica, chave in (("NDCG@10", "ndcg10"), ("Obra P@3", "obra_p3")):
            melhor = r.loc[componentes, metrica].idxmax()
            bloco[f"melhor_{chave}"] = melhor.lower()
            bloco[f"ganho_{chave}"] = br(100 * (r.at["Híbrido", metrica] / r.at[melhor, metrica] - 1), 0)
        bloco["ganho_ndcg_aleatorio"] = br(r.at["Híbrido", "NDCG@10"] / r.at["Aleatório", "NDCG@10"], 1)
        comparacoes = _ler(f"{prefixo}_comparacoes.csv").rename(columns={"IC95 inf": "inferior",
                                                                           "IC95 sup": "superior"})
        bloco["comparacoes"] = {
            f"{CHAVE_MODELO[c.modelo]}_{CHAVE_METRICA[c.metrica]}": {
                "dif": _com_sinal(c.diferenca, CASAS_DIFERENCA[c.metrica]),
                "ic": f"{_com_sinal(c.inferior, CASAS_DIFERENCA[c.metrica])} a "
                      f"{_com_sinal(c.superior, CASAS_DIFERENCA[c.metrica])}",
                "p": _p(c.p), "sig": "sim" if c.significativo else "não"}
            for c in comparacoes.itertuples()}
        competidores = comparacoes[comparacoes["modelo"] != "Oráculo (teto)"]
        bloco["n_comparacoes"] = str(len(competidores))
        bloco["n_significativas"] = str(int(competidores["significativo"].sum()))
        bloco["menor_p"], bloco["maior_p"] = _p(competidores["p"].min()), _p(competidores["p"].max())
        bloco["limiar_bonferroni"] = br(0.05 / len(competidores), 3)
        v[prefixo] = bloco
    curva = _ler("controlado_curva_mmr.csv").set_index("lambda")
    sem, adotado, menor = curva.loc[1.0], curva.loc[config.LAMBDA_MMR], curva.loc[round(config.LAMBDA_MMR - 0.1, 1)]
    v["mmr"] = {"div_sem": br(sem["Diversidade@10"], 2), "div_adotado": br(adotado["Diversidade@10"], 2),
                "vezes": br(adotado["Diversidade@10"] / sem["Diversidade@10"], 0),
                "perda_adotado": br(100 * (1 - adotado["NDCG@10"] / sem["NDCG@10"]), 0),
                "lambda_menor": br(round(config.LAMBDA_MMR - 0.1, 1), 1),
                "perda_menor": br(100 * (1 - menor["NDCG@10"] / sem["NDCG@10"]), 0)}
    ab = _ler("online_ab.csv").set_index("modelo")
    v["online"] = {chave: {"ctr": br(100 * ab.at[modelo, "CTR"], 1), "minutos": br(ab.at[modelo, "minutos por sessão"], 0),
                           "satisfacao": br(ab.at[modelo, "satisfação (1–5)"], 2)}
                   for modelo, chave in (("Aleatório", "aleatorio"), ("Popularidade", "popularidade"),
                                         ("Conteúdo", "conteudo"), ("Colaborativo", "colaborativo"),
                                         ("Conhecimento", "conhecimento"), ("Híbrido", "hibrido"),
                                         ("Oráculo (teto)", "oraculo"))}
    v["online"]["sessoes"] = br(int(ab["sessoes"].iloc[0]), 0)
    return v


def _yaml(dados: dict, nivel: int = 0) -> str:
    linhas = []
    for chave, valor in dados.items():
        if isinstance(valor, dict):
            linhas.append("  " * nivel + f"{chave}:")
            linhas.append(_yaml(valor, nivel + 1))
        else:
            linhas.append("  " * nivel + f'{chave}: "{valor}"')
    return "\n".join(linhas)


def renderizar():
    quarto = shutil.which("quarto")
    if quarto is None:
        print("Quarto não encontrado: tabelas geradas, mas o .docx/.pdf não foi renderizado.")
        return
    fonte = DIR_RELATORIO / "relatorio.qmd"
    for formato, extensao in (("docx", "docx"), ("typst", "pdf")):
        subprocess.run([quarto, "render", str(fonte), "--to", formato, "--output", f"Relatorio.{extensao}"],
                       cwd=DIR_RELATORIO, check=True)
    for intermediario in (DIR_RELATORIO / "relatorio.typ", DIR_RELATORIO / "relatorio_files"):
        if intermediario.is_dir():
            shutil.rmtree(intermediario)
        elif intermediario.exists():
            intermediario.unlink()


def exemplo_linha_de_comando(argumentos=("--aluno", "1", "--regras", "--k", "3")):
    """Saída real de `recomendar.py`, incluída no relatório como exemplo de demonstração."""
    import sys
    saida = subprocess.run([sys.executable, "-X", "utf8", str(config.RAIZ / "recomendar.py"), *argumentos],
                           cwd=config.RAIZ, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    comando = "python recomendar.py " + " ".join(argumentos)
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    (DIR_TABELAS / "demo_regras.md").write_text(f"```text\n$ {comando}\n{saida.strip()}\n```\n", encoding="utf-8")


def gerar(renderizar_documentos: bool = True):
    tabelas_dados()
    tabelas_resultados()
    exemplo_linha_de_comando()
    # O Typst só lê arquivos dentro da pasta do documento: as figuras vão junto com o relatório.
    shutil.copytree(config.DIR_FIGURAS, DIR_RELATORIO / "figuras", dirs_exist_ok=True)
    (DIR_RELATORIO / "variaveis.yml").write_text(_yaml(variaveis()) + "\n", encoding="utf-8")
    if renderizar_documentos:
        renderizar()

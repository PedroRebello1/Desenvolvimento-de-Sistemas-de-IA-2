"""Figuras do relatório, geradas a partir de `resultados/`.

Estilo único para todas: o híbrido em azul (destaque), os demais modelos em cinza,
o teto do oráculo vazado; grade e eixos discretos; o título diz a conclusão e o
subtítulo diz o que está plotado. Números no formato brasileiro.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker, transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config, dados

TINTA, TINTA_2, MUDO = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"
DESTAQUE, DESTAQUE_CLARO, DESTAQUE_ESCURO = "#2a78d6", "#86b6ef", "#1c5cab"
CONTEXTO, REFERENCIA = "#c9c7bf", "#6f6d66"
HIBRIDO, ORACULO = "Híbrido", "Oráculo (teto)"
ORDEM_MODELOS = ["Aleatório", "Popularidade", "Conteúdo", "Colaborativo", "Conhecimento", HIBRIDO, ORACULO]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": TINTA,
    "axes.edgecolor": EIXO, "axes.labelcolor": TINTA_2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "grid.color": GRADE, "grid.linewidth": 0.7, "xtick.color": MUDO, "ytick.color": MUDO,
    "xtick.labelcolor": TINTA_2, "ytick.labelcolor": TINTA_2, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


def br(valor: float, casas: int = 2) -> str:
    """Número no formato brasileiro: 1.234,56."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def _eixo_numerico(eixo, marcas: int = 5):
    """Marcas em passos 1-2-5, com as casas decimais necessárias para distingui-las."""
    eixo.set_major_locator(ticker.MaxNLocator(nbins=marcas, steps=[1, 2, 5, 10]))

    def formatar(valor, _):
        locais = eixo.get_majorticklocs()
        passo = np.min(np.diff(locais)) if len(locais) > 1 else 1.0
        casas = 0
        while casas < 6 and abs(passo * 10 ** casas - round(passo * 10 ** casas)) > 1e-6:
            casas += 1
        return br(valor, casas)

    eixo.set_major_formatter(ticker.FuncFormatter(formatar))


def _titulo(fig, titulo: str, subtitulo: str):
    """Título (a conclusão) e subtítulo (o que está plotado) acima do conteúdo, alinhados à esquerda dele."""
    fig.canvas.draw()
    caixa = fig.get_tightbbox(fig.canvas.get_renderer())
    largura, altura = fig.get_size_inches()
    x, topo = caixa.x0 / largura, caixa.y1 / altura

    def acima(pontos):
        return transforms.offset_copy(fig.transFigure, fig=fig, y=pontos, units="points")

    fig.text(x, topo, subtitulo, transform=acima(8), fontsize=9.5, color=TINTA_2, ha="left", va="bottom")
    fig.text(x, topo, titulo, transform=acima(26), fontsize=12.5, fontweight="semibold", color=TINTA,
             ha="left", va="bottom")


def _salvar(fig, nome: str):
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.DIR_FIGURAS / nome)
    plt.close(fig)


def _cor(modelo: str) -> str:
    return DESTAQUE if modelo == HIBRIDO else CONTEXTO


def _barras_modelos(ax, tabela: pd.DataFrame, coluna: str, casas: int, ic: tuple[str, str] | None = None,
                    referencia: str | None = "Aleatório", titulo: str = ""):
    """Barras horizontais por modelo (híbrido em destaque) com IC opcional e linha de referência."""
    tabela = tabela.set_index("modelo").reindex([m for m in ORDEM_MODELOS if m in set(tabela["modelo"])])
    y = np.arange(len(tabela))[::-1]
    for posicao, (modelo, linha) in zip(y, tabela.iterrows()):
        valor = linha[coluna]
        if modelo == ORACULO:
            ax.barh(posicao, valor, height=0.56, color="white", edgecolor=REFERENCIA, linewidth=1.0, hatch="////")
        else:
            ax.barh(posicao, valor, height=0.56, color=_cor(modelo))
        fim = valor
        if ic is not None and pd.notna(linha.get(ic[0])):
            inferior, superior = linha[ic[0]], linha[ic[1]]
            ax.plot([inferior, superior], [posicao, posicao], color=TINTA_2, linewidth=1.1)
            for extremo in (inferior, superior):
                ax.plot([extremo] * 2, [posicao - 0.14, posicao + 0.14], color=TINTA_2, linewidth=1.1)
            fim = max(valor, superior)
        ax.text(fim, posicao, "  " + br(valor, casas), va="center", ha="left", fontsize=8.5,
                color=TINTA if modelo == HIBRIDO else TINTA_2, fontweight="semibold" if modelo == HIBRIDO else None)
    if referencia is not None and referencia in tabela.index:
        ax.axvline(tabela.at[referencia, coluna], color=REFERENCIA, linewidth=0.9, zorder=0)
    ax.set_yticks(y, tabela.index)
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo)
    limite = tabela[ic[1]].max() if ic is not None and ic[1] in tabela else tabela[coluna].max()
    ax.set_xlim(0, max(limite, tabela[coluna].max()) * 1.32)
    _eixo_numerico(ax.xaxis, 4)


def figura_integridade(qualidade: dict):
    q = qualidade["integridade"]
    rotulos = ["Todas as interações", "Com aluno cadastrado", "Com material cadastrado",
               "Com aluno e material cadastrados"]
    valores = np.array([q["n_interacoes"], q["com_aluno_cadastrado"], q["com_material_cadastrado"],
                        q["com_ambos_cadastrados"]]) / 1e6
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    y = np.arange(len(valores))[::-1]
    ax.barh(y, valores, height=0.56, color=[CONTEXTO, CONTEXTO, CONTEXTO, DESTAQUE])
    for posicao, valor in zip(y, valores):
        parcela = valor / valores[0]
        ax.text(valor, posicao, f"  {br(valor, 2)} mi" + ("" if parcela == 1 else f"  ({br(100 * parcela, 1)}%)"),
                va="center", fontsize=9, color=TINTA)
    ax.set_yticks(y, rotulos)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 12.5)
    _eixo_numerico(ax.xaxis)
    ax.set_xlabel("milhões de interações")
    _titulo(fig, "Só 2% das interações ligam um aluno e um material cadastrados",
            "interacoes.csv cita ids de aluno até 100.000 e de material até 50.000; os cadastros têm 10.000 de cada")
    _salvar(fig, "fig01_integridade.png")


def figura_distribuicoes(qualidade: dict, base: dados.Base):
    d = qualidade["distribuicoes"]
    fig, eixos = plt.subplots(1, 4, figsize=(11, 2.7), gridspec_kw={"width_ratios": [5, 2, 6, 6]})
    ax = eixos[0]
    contagens = np.array(d["avaliacao"]["contagens"]) / 1e6
    ax.bar(np.arange(1, 6), contagens, width=0.6, color=DESTAQUE)
    ax.set_title("Nota (milhões)")
    ax.set_xticks(np.arange(1, 6))
    ax.set_xlabel(f"p = {br(d['avaliacao']['p'], 2)}")

    ax = eixos[1]
    tipos = d["tipo_interacao"]["contagens"]
    ax.bar([0, 1], np.array(list(tipos.values())) / 1e6, width=0.6, color=DESTAQUE)
    ax.set_xticks([0, 1], ["visual.", "leitura"] if list(tipos)[0] == "visualizacao" else ["leitura", "visual."])
    ax.set_title("Tipo (milhões)")
    ax.set_xlabel(f"p = {br(d['tipo_interacao']['p'], 2)}")

    ax = eixos[2]
    duracao = base.interacoes["duracao_minutos"].to_numpy()
    minimo = int(duracao.min())
    por_minuto = np.bincount(duracao - minimo) / 1e3
    ax.fill_between(np.arange(minimo, minimo + len(por_minuto)), por_minuto, step="mid", color=DESTAQUE,
                    alpha=0.15, linewidth=0)
    ax.step(np.arange(minimo, minimo + len(por_minuto)), por_minuto, where="mid", color=DESTAQUE, linewidth=1.2)
    ax.set_ylim(0, por_minuto.max() * 1.15)
    ax.set_title("Interações por minuto de duração (mil)")
    ax.set_xlabel(f"minutos · p = {br(d['duracao']['p'], 2)}")

    ax = eixos[3]
    meses = np.array(d["data"]["por_mes"])
    dias = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    ax.bar(np.arange(1, 13), meses / dias / 1e3, width=0.6, color=DESTAQUE)
    ax.set_xticks(np.arange(1, 13), list("JFMAMJJASOND"))
    ax.set_title("Interações por dia, em cada mês (mil)")
    ax.set_xlabel(f"2023 · p = {br(d['data']['p'], 2)}")
    for ax in eixos:
        ax.grid(axis="x", visible=False)
        _eixo_numerico(ax.yaxis, 4)
    fig.tight_layout(w_pad=2.2)
    _titulo(fig, "Todas as colunas das interações seguem distribuições uniformes",
            "Contagens nas 10 milhões de interações; p = teste qui-quadrado de aderência à distribuição uniforme")
    _salvar(fig, "fig02_distribuicoes.png")


def figura_estabilidade(qualidade: dict, base: dados.Base):
    inter = base.interacoes
    metade = inter["dia"].to_numpy() < dados.dia_de("2023-07-01")
    nota = inter["avaliacao"].to_numpy().astype(float)
    fig, eixos = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, coluna, nome, chave in ((eixos[0], "id_aluno", "Alunos", "estabilidade_aluno"),
                                    (eixos[1], "id_material", "Materiais", "estabilidade_material")):
        ids = inter[coluna].to_numpy()
        medias = [np.bincount(ids[m], weights=nota[m]) / np.maximum(np.bincount(ids[m]), 1) for m in (metade, ~metade)]
        tamanho = min(map(len, medias))
        a, b = medias[0][1:tamanho], medias[1][1:tamanho]
        amostra = np.random.default_rng(0).choice(len(a), 3000, replace=False)
        ax.scatter(a[amostra], b[amostra], s=6, color=DESTAQUE, alpha=0.35, linewidths=0)
        ax.set_title(f"{nome}: r = {br(qualidade['sinal'][chave]['r'], 3)}")
        ax.set_xlabel("nota média, jan–jun")
        ax.set_ylabel("nota média, jul–dez")
        ax.set_xlim(2.2, 3.8)
        ax.set_ylim(2.2, 3.8)
        ax.set_aspect("equal")
        _eixo_numerico(ax.xaxis, 4)
        _eixo_numerico(ax.yaxis, 4)
    fig.tight_layout(w_pad=3)
    _titulo(fig, "A média de um aluno ou material num semestre não prevê a do outro",
            "Cada ponto é um aluno (ou material); amostra de 3.000; r = correlação de Pearson em todos os grupos")
    _salvar(fig, "fig03_estabilidade.png")


def figura_perfil_obra():
    media = pd.read_csv(config.DIR_RESULTADOS / "media_perfil_obra.csv", index_col=0)
    obras = pd.read_csv(config.DIR_RESULTADOS / "obras.csv")
    perfis = pd.read_csv(config.DIR_RESULTADOS / "perfis.csv")
    abreviacao = {"Engenharia de Software": "Eng. Software", "Ciência de Dados": "Ciência de Dados",
                  "Engenharia de Computação": "Eng. Computação", "Sistemas de Informação": "Sist. Informação"}
    rotulos_perfis = [f"{abreviacao.get(c, c)}, {p}º per." for c, p in zip(perfis["curso"], perfis["periodo"])]
    cores = LinearSegmentedColormap.from_list("divergente", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c03a3a"])
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    imagem = ax.imshow(media.to_numpy(), cmap=cores, vmin=2.5, vmax=3.5, aspect="auto")
    for (i, j), valor in np.ndenumerate(media.to_numpy()):
        ax.text(j, i, br(valor, 2), ha="center", va="center", fontsize=7.5, color=TINTA_2)
    ax.set_xticks(np.arange(len(obras)), obras["titulo"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(perfis)), rotulos_perfis)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    barra = fig.colorbar(imagem, ax=ax, fraction=0.03, pad=0.02, ticks=[2.5, 3, 3.5])
    barra.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: br(v, 1)))
    barra.outline.set_visible(False)
    _titulo(fig, "Nenhuma combinação de perfil e obra se afasta da nota média 3",
            "Nota média nas 200.622 interações com aluno e material cadastrados; escala de 2,5 a 3,5")
    _salvar(fig, "fig04_perfil_obra.png")


def _ranqueamento(prefixo: str) -> pd.DataFrame:
    return pd.read_csv(config.DIR_RESULTADOS / f"{prefixo}_ranqueamento.csv")


def figura_modelos(prefixo: str, titulo: str, subtitulo: str, nome: str):
    ranq = _ranqueamento(prefixo)
    notas = pd.read_csv(config.DIR_RESULTADOS / f"{prefixo}_notas.csv")
    notas = notas[notas["pares"] == "aluno e material cadastrados"]
    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.1))
    _barras_modelos(eixos[0], ranq, "NDCG@10", 4, ("NDCG@10 IC95 inf", "NDCG@10 IC95 sup"),
                    titulo="NDCG@10 por material")
    _barras_modelos(eixos[1], ranq, "Obra P@3", 3, ("Obra P@3 IC95 inf", "Obra P@3 IC95 sup"),
                    titulo="Precisão@3 por obra")
    ordem = ["Média global", "Vieses (μ + b_u + b_i)", "Conteúdo", "Colaborativo", "Conhecimento", HIBRIDO]
    notas = notas.set_index("modelo").reindex(ordem)
    ax = eixos[2]
    y = np.arange(len(notas))[::-1]
    ax.barh(y, notas["RMSE"], height=0.56, color=[_cor(m) for m in notas.index])
    for posicao, (modelo, valor) in zip(y, notas["RMSE"].items()):
        ax.text(valor, posicao, "  " + br(valor, 3), va="center", fontsize=8.5,
                color=TINTA if modelo == HIBRIDO else TINTA_2, fontweight="semibold" if modelo == HIBRIDO else None)
    ax.axvline(notas.at["Média global", "RMSE"], color=REFERENCIA, linewidth=0.9, zorder=0)
    ax.set_yticks(y, ["Média global", "Vieses μ+bᵤ+bᵢ", "Conteúdo", "Colaborativo", "Conhecimento", HIBRIDO])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, notas["RMSE"].max() * 1.3)
    _eixo_numerico(ax.xaxis, 4)
    ax.set_title("RMSE da nota prevista (menor é melhor)")
    fig.tight_layout(w_pad=2.5)
    _titulo(fig, titulo, subtitulo)
    _salvar(fig, nome)


def figura_curva_mmr(prefixo: str = "controlado"):
    curva = pd.read_csv(config.DIR_RESULTADOS / f"{prefixo}_curva_mmr.csv").sort_values("lambda", ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(curva["Diversidade@10"], curva["NDCG@10"], color=DESTAQUE, linewidth=2, zorder=2)
    for _, linha in curva.iterrows():
        escolhido = np.isclose(linha["lambda"], config.LAMBDA_MMR)
        ax.scatter(linha["Diversidade@10"], linha["NDCG@10"], s=46 if escolhido else 26, zorder=3,
                   color=DESTAQUE_ESCURO if escolhido else DESTAQUE, edgecolors="white", linewidths=1.5)
        ax.annotate(f"λ = {br(linha['lambda'], 1)}" + ("  (adotado)" if escolhido else ""),
                    (linha["Diversidade@10"], linha["NDCG@10"]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8.5, color=TINTA if escolhido else TINTA_2)
    ax.set_xlabel("diversidade intra-lista @10 (0 = cópias da mesma obra)")
    ax.set_ylabel("NDCG@10")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, curva["NDCG@10"].max() * 1.25)
    _eixo_numerico(ax.xaxis)
    _eixo_numerico(ax.yaxis)
    inicial, adotado = curva.iloc[0], curva[np.isclose(curva["lambda"], config.LAMBDA_MMR)].iloc[0]
    vezes = adotado["Diversidade@10"] / max(inicial["Diversidade@10"], 1e-9)
    perda = 1 - adotado["NDCG@10"] / inicial["NDCG@10"]
    _titulo(fig, f"Com λ = {br(config.LAMBDA_MMR, 1)}, a lista fica {br(vezes, 0)}× mais diversa "
                 f"por {br(100 * perda, 0)}% de NDCG",
            "Re-ranqueamento MMR do híbrido no cenário com sinal controlado; λ = 1 desliga a diversificação")
    _salvar(fig, "fig07_curva_mmr.png")


def figura_cold_start(prefixo: str = "controlado"):
    todos = _ranqueamento(prefixo).set_index("modelo")
    novos = pd.read_csv(config.DIR_RESULTADOS / f"{prefixo}_cold_start.csv").set_index("modelo")
    modelos = [m for m in ORDEM_MODELOS if m in novos.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = np.arange(len(modelos))[::-1]
    for posicao, modelo in zip(y, modelos):
        com, sem = todos.at[modelo, "Obra P@3"], novos.at[modelo, "Obra P@3"]
        ax.plot([com, sem], [posicao, posicao], color=DESTAQUE_CLARO if modelo == HIBRIDO else GRADE,
                linewidth=2.5, zorder=1)
        ax.scatter(com, posicao, s=42, color=DESTAQUE_ESCURO, zorder=2, edgecolors="white", linewidths=1.5,
                   label="com histórico (todos os alunos)" if posicao == y[0] else None)
        ax.scatter(sem, posicao, s=42, color=DESTAQUE_CLARO, zorder=2, edgecolors="white", linewidths=1.5,
                   label="sem histórico (cold start)" if posicao == y[0] else None)
        ax.text(max(com, sem) + 0.012, posicao, br(sem, 3), va="center", fontsize=8.5,
                color=TINTA if modelo == HIBRIDO else TINTA_2, fontweight="semibold" if modelo == HIBRIDO else None)
    ax.set_yticks(y, modelos)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.6)
    _eixo_numerico(ax.xaxis)
    ax.set_xlabel("Precisão@3 por obra")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, fontsize=8.5, handletextpad=0.3,
              columnspacing=1.5, borderaxespad=0.2)
    _titulo(fig, "Sem histórico, o híbrido mantém a precisão apoiado em conteúdo e regras",
            "2.000 alunos com o histórico de treino apagado versus todos os alunos; rótulo = valor sem histórico")
    _salvar(fig, "fig08_cold_start.png")


def figura_online():
    ab = pd.read_csv(config.DIR_RESULTADOS / "online_ab.csv")
    ab["CTR %"] = 100 * ab["CTR"]
    ab["CTR % inf"], ab["CTR % sup"] = 100 * (ab["CTR"] - ab["CTR IC95"]), 100 * (ab["CTR"] + ab["CTR IC95"])
    for coluna, ic in (("minutos por sessão", "minutos IC95"), ("satisfação (1–5)", "satisfação IC95")):
        ab[coluna + " inf"], ab[coluna + " sup"] = ab[coluna] - ab[ic], ab[coluna] + ab[ic]
    fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.1))
    _barras_modelos(eixos[0], ab, "CTR %", 1, ("CTR % inf", "CTR % sup"), titulo="CTR (%)")
    _barras_modelos(eixos[1], ab, "minutos por sessão", 0, ("minutos por sessão inf", "minutos por sessão sup"),
                    titulo="Tempo de engajamento (min por sessão)")
    _barras_modelos(eixos[2], ab, "satisfação (1–5)", 2, ("satisfação (1–5) inf", "satisfação (1–5) sup"),
                    titulo="Satisfação média (1–5)")
    eixos[2].set_xlim(0, 5.9)
    fig.tight_layout(w_pad=2.5)
    sessoes = int(ab["sessoes"].iloc[0])
    _titulo(fig, "No teste A/B simulado, o híbrido lidera CTR, engajamento e satisfação",
            f"{br(sessoes, 0)} sessões por modelo, 10 recomendações cada, alunos simulados do cenário controlado; "
            "barras finas = IC de 95%; linha vertical = aleatório")
    _salvar(fig, "fig09_online.png")


def figura_validacao_fc():
    fig, eixos = plt.subplots(1, 2, figsize=(8.6, 2.9))
    for ax, prefixo, nome in ((eixos[0], "real", "Dados fornecidos"), (eixos[1], "controlado", "Sinal controlado")):
        historico = pd.read_csv(config.DIR_RESULTADOS / f"{prefixo}_historico_fc.csv")
        usada = json.loads((config.DIR_RESULTADOS / f"{prefixo}_resumo.json").read_text(encoding="utf-8"))["iteracoes_fc"]
        ponto = historico.set_index("iteracao").loc[usada]
        ax.plot(historico["iteracao"], historico["rmse_validacao"], color=DESTAQUE, linewidth=2)
        ax.scatter(usada, ponto["rmse_validacao"], s=40, color=DESTAQUE_ESCURO, zorder=3, edgecolors="white",
                   linewidths=1.5)
        a_direita = usada > historico["iteracao"].max() * 0.6
        ax.annotate(f"iteração usada: {usada}", (usada, ponto["rmse_validacao"]), textcoords="offset points",
                    xytext=(-6 if a_direita else 6, 8), ha="right" if a_direita else "left", fontsize=8.5, color=TINTA)
        ax.set_title(nome)
        ax.set_xlabel("iteração do ALS")
        ax.set_ylabel("RMSE na validação")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        _eixo_numerico(ax.yaxis, 4)
    fig.tight_layout(w_pad=3)
    _titulo(fig, "A parada antecipada evita que a filtragem colaborativa decore ruído",
            "RMSE de validação (set–out) por iteração do ALS; o treino final usa a iteração marcada")
    _salvar(fig, "fig10_validacao_fc.png")


def figura_arquitetura():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def caixa(x, y, largura, altura, titulo, linhas=(), destaque=False, tom=None):
        cor = DESTAQUE if destaque else (tom or EIXO)
        ax.add_patch(FancyBboxPatch((x, y), largura, altura, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=(DESTAQUE + "1f") if destaque else "white", edgecolor=cor,
                                    linewidth=1.6 if destaque else 1.0))
        ax.text(x + largura / 2, y + altura - 2.6, titulo, ha="center", va="center", fontsize=9.5,
                fontweight="semibold", color=TINTA)
        for i, linha in enumerate(linhas):
            ax.text(x + largura / 2, y + altura - 6.2 - 3.1 * i, linha, ha="center", va="center", fontsize=8,
                    color=TINTA_2)

    def seta(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=MUDO,
                                     linewidth=1.0, shrinkA=0, shrinkB=0))

    caixa(1, 38, 16, 14, "Dados (CSV)", ["dados_alunos", "materiais_didaticos", "interacoes (10 mi)"])
    caixa(21, 38, 19, 14, "Preparação", ["normaliza termos", "integridade · obras", "Parquet · divisão temporal"])
    caixa(46, 42, 22, 11, "Conteúdo", ["TF-IDF + áreas + formato", "perfil Rocchio · kNN"])
    caixa(46, 28, 22, 11, "Colaborativo", ["fatoração de matrizes (ALS)", "μ + bᵤ + bᵢ + pᵤ·qᵢ"])
    caixa(46, 14, 22, 11, "Conhecimento", ["pré-requisitos, nível,", "curso, novidade, trilha"])
    caixa(46, 0.5, 22, 10.5, "Base de conhecimento", ["termo → área · curso → área", "pré-requisitos · competências"],
          tom=REFERENCIA)
    caixa(73, 14, 17, 39, "Híbrido", ["1 ponderação", "2 troca (cold start)", "3 restrição", "4 MMR", "",
                                      "notas:", "empilhamento"], destaque=True)
    caixa(94, 42, 15, 11, "Recomendação", ["top-K + motivos", "nota prevista"])
    caixa(94, 26.5, 15, 11, "Demonstração", ["Streamlit e CLI", "registra eventos"])
    caixa(94, 1, 15, 21, "Avaliação", ["offline: P/R/F1,", "NDCG, RMSE,", "cobertura, divers.", "online: CTR,",
                                      "tempo, satisfação"])
    seta(17, 45, 21, 45)
    ax.plot([40, 43], [45, 45], color=MUDO, linewidth=1.0)          # barramento da preparação
    ax.plot([43, 43], [19.5, 47.5], color=MUDO, linewidth=1.0)
    for y in (47.5, 33.5, 19.5):
        seta(43, y, 46, y)
        seta(68, y, 73, y)
    seta(57, 11, 57, 14)
    seta(90, 47.5, 94, 47.5)
    seta(90, 17, 94, 17)
    seta(101.5, 42, 101.5, 37.5)
    seta(101.5, 26.5, 101.5, 22)
    ax.text(102.5, 24.2, "eventos", fontsize=7.5, color=MUDO, va="center")
    _titulo(fig, "Três componentes independentes alimentam um híbrido de quatro etapas",
            "Fluxo de dados do sistema; a base de conhecimento, escrita à mão, também orienta o perfil de conteúdo "
            "(termo → área)")
    _salvar(fig, "fig00_arquitetura.png")


def gerar_todas(base: dados.Base | None = None):
    qualidade = json.loads((config.DIR_RESULTADOS / "qualidade_dados.json").read_text(encoding="utf-8"))
    base = base or dados.carregar_base()
    figura_arquitetura()
    figura_integridade(qualidade)
    figura_distribuicoes(qualidade, base)
    figura_estabilidade(qualidade, base)
    figura_perfil_obra()
    figura_modelos("real", "Nos dados fornecidos, nenhum modelo se distingue do acaso",
                   "Teste nov–dez/2023; barras finas = IC de 95% (bootstrap por aluno); linha vertical = aleatório "
                   "(ranqueamento) e média global (RMSE)", "fig05_modelos_real.png")
    figura_modelos("controlado", "Com sinal controlado, o híbrido supera cada componente isolado",
                   "Mesmos alunos e materiais, interações simuladas; barras finas = IC de 95%; "
                   "oráculo = ranqueamento pela utilidade verdadeira (teto)", "fig06_modelos_controlado.png")
    figura_curva_mmr()
    figura_cold_start()
    figura_online()
    figura_validacao_fc()

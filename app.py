"""Demonstração funcional do sistema de recomendação híbrido.

Executar (depois de `python pipeline.py`):
    streamlit run app.py
"""
from __future__ import annotations

import copy
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from recomendador import config, dados, ontologia
from recomendador.modelos import COMPONENTES, Aleatorio
from recomendador.online import RegistroEventos, metricas_online, nova_sessao

st.set_page_config(page_title="Recomendador de materiais didáticos", page_icon="📚", layout="wide")

CENARIOS = {
    "Dados fornecidos": ("real", config.ARQ_INTERACOES_CACHE),
    "Sinal controlado (simulado)": ("controlado", config.DIR_ARTEFATOS / "interacoes_simuladas.parquet"),
}
NOMES_COMPONENTES = {"conteudo": "Conteúdo", "colaborativo": "Colaborativo", "conhecimento": "Conhecimento"}
MODO_AB = "Teste A/B (sorteado por sessão)"


@st.cache_resource(show_spinner="Carregando modelos treinados…")
def carregar_modelos(prefixo: str) -> dict:
    with open(config.DIR_ARTEFATOS / f"modelos_{prefixo}.pkl", "rb") as arquivo:
        salvo = pickle.load(arquivo)
    comp = salvo["componentes"]
    salvo["modelos"] = {"Híbrido": salvo["hibrido"], "Conteúdo": comp.conteudo, "Colaborativo": comp.colaborativo,
                        "Conhecimento": comp.conhecimento, "Popularidade": comp.popularidade,
                        "Aleatório": Aleatorio()}
    return salvo


@st.cache_resource(show_spinner="Carregando histórico de interações…")
def carregar_historico(caminho: str) -> tuple[pd.DataFrame, np.ndarray]:
    interacoes = pd.read_parquet(caminho)
    ordem = np.argsort(interacoes["id_aluno"].to_numpy(), kind="stable")
    interacoes = interacoes.iloc[ordem].reset_index(drop=True)
    return interacoes, interacoes["id_aluno"].to_numpy()


@st.cache_data
def carregar_cadastros() -> tuple[pd.DataFrame, pd.DataFrame]:
    return dados.carregar_alunos(), dados.carregar_materiais()


def historico_do_aluno(interacoes: pd.DataFrame, ids_ordenados: np.ndarray, id_aluno: int) -> pd.DataFrame:
    inicio, fim = np.searchsorted(ids_ordenados, [id_aluno, id_aluno + 1])
    return interacoes.iloc[inicio:fim]


def registro() -> RegistroEventos:
    return RegistroEventos()


def estado_inicial():
    st.session_state.setdefault("sessao", nova_sessao())
    st.session_state.setdefault("braco_ab", None)
    st.session_state.setdefault("impressoes_registradas", set())
    st.session_state.setdefault("aberto", None)
    st.session_state.setdefault("avaliacoes_enviadas", set())


def artefatos_ausentes() -> list[str]:
    exigidos = [config.DIR_ARTEFATOS / "modelos_real.pkl", config.ARQ_INTERACOES_CACHE]
    return [str(p.relative_to(config.RAIZ)) for p in exigidos if not p.exists()]


# ----------------------------------------------------------------------------- barra lateral
def barra_lateral(alunos: pd.DataFrame) -> dict:
    """Controles da barra lateral. Aceitam valores iniciais pela URL, ex.: ?cenario=controlado&aluno=1&modelo=Conhecimento."""
    st.sidebar.header("Configuração")
    url = st.query_params
    disponiveis = [c for c, (p, arq) in CENARIOS.items()
                   if (config.DIR_ARTEFATOS / f"modelos_{p}.pkl").exists() and arq.exists()]
    prefixos = [CENARIOS[c][0] for c in disponiveis]
    cenario = st.sidebar.radio("Cenário", disponiveis,
                               index=prefixos.index(url["cenario"]) if url.get("cenario") in prefixos else 0,
                               help="'Dados fornecidos' usa os CSVs do projeto. "
                                    "'Sinal controlado' usa interações simuladas em que há preferências reais.")
    exemplos = {f"{a.id_aluno} · {a.curso}, {a.periodo}º per.": a.id_aluno for a in alunos.head(8).itertuples()}
    rotulos = list(exemplos)
    if "id_aluno" not in st.session_state:   # primeira execução: a URL define o aluno inicial
        st.session_state.id_aluno = int(url["aluno"]) if url.get("aluno", "").isdigit() else exemplos[rotulos[1]]
        inicial = [r for r in rotulos if exemplos[r] == st.session_state.id_aluno]
        st.session_state.exemplo = inicial[0] if inicial else rotulos[1]

    def ao_escolher_exemplo():
        st.session_state.id_aluno = exemplos[st.session_state.exemplo]

    st.sidebar.selectbox("Aluno de exemplo (um por perfil)", rotulos, key="exemplo", on_change=ao_escolher_exemplo)
    maximo = 100_000 if cenario == "Dados fornecidos" else int(alunos.index.max())
    st.session_state.id_aluno = min(max(int(st.session_state.id_aluno), 1), maximo)
    id_aluno = st.sidebar.number_input("ou digite um id de aluno", 1, maximo, step=1, key="id_aluno",
                                       help="Nos dados fornecidos, ids acima de 10.000 não têm perfil "
                                            "cadastrado: o sistema usa só o histórico.")
    nomes_modelos = ["Híbrido", "Conteúdo", "Colaborativo", "Conhecimento", "Popularidade", "Aleatório", MODO_AB]
    modelo = st.sidebar.selectbox("Modelo", nomes_modelos,
                                  index=nomes_modelos.index(url["modelo"]) if url.get("modelo") in nomes_modelos else 0)
    k = st.sidebar.slider("Tamanho da lista (K)", 5, 20, config.K_PRINCIPAL)
    return {"cenario": cenario, "id_aluno": int(id_aluno), "modelo": modelo, "k": k}


def ajustes_hibrido(hibrido) -> object:
    with st.sidebar.expander("Ajustar o híbrido"):
        st.caption("Pesos antes das regras de troca (são renormalizados para somar 1).")
        pesos = {c: st.slider(NOMES_COMPONENTES[c], 0.0, 1.0, float(hibrido.pesos[c]), 0.05, key=f"peso_{c}")
                 for c in COMPONENTES}
        lambda_mmr = st.slider("λ do MMR (1 = sem diversificação)", 0.0, 1.0, float(hibrido.lambda_mmr), 0.05)
        penalidade = st.slider("Penalidade por pré-requisito faltante", 0.0, 1.0,
                               float(hibrido.penalidade_inelegivel), 0.05)
    if sum(pesos.values()) == 0:
        st.sidebar.warning("Todos os pesos em zero: usando os pesos treinados.")
        return hibrido
    ajustado = copy.copy(hibrido)
    total = sum(pesos.values())
    ajustado.pesos = {c: v / total for c, v in pesos.items()}
    ajustado.lambda_mmr, ajustado.penalidade_inelegivel = lambda_mmr, penalidade
    return ajustado


# ----------------------------------------------------------------------------- aba de recomendações
def cartao_perfil(id_aluno: int, alunos: pd.DataFrame, salvo: dict, historico: pd.DataFrame,
                  materiais: pd.DataFrame):
    conhecimento = salvo["componentes"].conhecimento
    col_perfil, col_hist = st.columns([1.1, 1])
    with col_perfil:
        with st.container(border=True):
            if id_aluno in alunos.index:
                aluno = alunos.loc[id_aluno]
                st.subheader(f"{aluno['nome']} · aluno {id_aluno}")
                st.write(f"**{aluno['curso']}**, {aluno['periodo']}º período "
                         f"(nível esperado: {ontologia.NIVEIS[ontologia.nivel_esperado(int(aluno['periodo']))]})")
                st.write("**Disciplinas cursadas:** " + ", ".join(aluno["disciplinas"]))
                st.write("**Interesses:** " + ", ".join(aluno["interesses"]))
                adquiridas = conhecimento.competencias(id_aluno) - ontologia.competencias_declaradas(aluno["disciplinas"])
                st.write("**Competências vindas do histórico:** " + (", ".join(sorted(adquiridas)) or "nenhuma"))
            else:
                st.subheader(f"Aluno {id_aluno}")
                st.info("Sem perfil cadastrado: as regras de conhecimento não se aplicam e o perfil de conteúdo "
                        "vem só do histórico.")
    with col_hist:
        with st.container(border=True):
            st.subheader("Histórico")
            cadastrados = historico[historico["id_material"].isin(materiais.index)]
            c1, c2, c3 = st.columns(3)
            c1.metric("Interações", f"{len(historico):,}".replace(",", "."))
            c2.metric("Com material cadastrado", f"{len(cadastrados):,}".replace(",", "."))
            c3.metric("Nota média", f"{historico['avaliacao'].mean():.2f}".replace(".", ",") if len(historico) else "–")
            if len(cadastrados):
                obras = (cadastrados.assign(obra=materiais.loc[cadastrados["id_material"], "titulo"].to_numpy())
                         .groupby("obra")["avaliacao"].agg(["count", "mean"])
                         .sort_values("count", ascending=False).head(5))
                obras.columns = ["interações", "nota média"]
                st.dataframe(obras.style.format({"nota média": "{:.1f}"}), use_container_width=True)


def explicacoes(id_aluno: int, id_material: int, salvo: dict, alunos: pd.DataFrame, materiais: pd.DataFrame,
                componentes_norm: dict | None, posicao_candidato: int, regras: pd.DataFrame | None) -> list[str]:
    comp = salvo["componentes"]
    motivos = []
    if regras is not None:
        obra = int(materiais.at[id_material, "id_obra"])
        motivos += [f"Regra: {j}" for j in regras.loc[regras["id_obra"] == obra, "justificativas"].iloc[0]]
    contribuicao = comp.conteudo.contribuicoes(id_aluno, id_material)
    if contribuicao["area"] > 0.2:
        motivos.append(f"Conteúdo: área {materiais.at[id_material, 'area']} alinhada ao seu perfil")
    termos = comp.conteudo.termos_em_comum(id_aluno, id_material) if id_aluno in alunos.index else []
    if termos:
        motivos.append("Conteúdo: termos em comum com seu perfil — " + ", ".join(termos))
    nota = float(comp.colaborativo.prever_avaliacao(np.array([id_aluno]), np.array([id_material]))[0])
    motivos.append(f"Colaborativo: nota prevista {nota:.1f}".replace(".", ","))
    if componentes_norm is not None:
        partes = [f"{NOMES_COMPONENTES[c]} {componentes_norm[c][posicao_candidato]:.2f}".replace(".", ",")
                  for c in COMPONENTES]
        motivos.append("Pontuação normalizada: " + " · ".join(partes))
    return motivos


def aba_recomendacoes(opcoes: dict, alunos: pd.DataFrame, materiais: pd.DataFrame):
    prefixo, caminho = CENARIOS[opcoes["cenario"]]
    salvo = carregar_modelos(prefixo)
    interacoes, ids_ordenados = carregar_historico(str(caminho))
    id_aluno = opcoes["id_aluno"]
    historico = historico_do_aluno(interacoes, ids_ordenados, id_aluno)
    hibrido = ajustes_hibrido(salvo["hibrido"])
    modelos = dict(salvo["modelos"], **{"Híbrido": hibrido})

    if opcoes["modelo"] == MODO_AB:
        if st.session_state.braco_ab is None:
            st.session_state.braco_ab = str(np.random.default_rng().choice(list(modelos)))
        nome_modelo = st.session_state.braco_ab
    else:
        nome_modelo = opcoes["modelo"]
    modelo = modelos[nome_modelo]

    cartao_perfil(id_aluno, alunos, salvo, historico, materiais)

    candidatos = materiais.index.to_numpy()
    vistos = np.isin(candidatos, historico["id_material"].to_numpy())[None, :]
    lista = modelo.recomendar(np.array([id_aluno]), candidatos, opcoes["k"], vistos)[0]
    posicao = {m: i for i, m in enumerate(candidatos)}
    componentes_norm, pesos = None, None
    if nome_modelo == "Híbrido":
        explicacao = hibrido.explicar(id_aluno, candidatos)
        componentes_norm, pesos = explicacao["componentes"], explicacao["pesos"]
    regras = (salvo["componentes"].conhecimento.avaliar_obras(id_aluno) if id_aluno in alunos.index else None)

    titulo = f"Recomendações — {nome_modelo}" + (" (variante sorteada no teste A/B)" if opcoes["modelo"] == MODO_AB else "")
    st.header(titulo)
    if pesos is not None:
        st.caption("Peso efetivo de cada componente para este aluno (após as regras de troca): "
                   + " · ".join(f"{NOMES_COMPONENTES[c]} {pesos[c]:.0%}" for c in COMPONENTES))
    chave_lista = (st.session_state.sessao, opcoes["cenario"], id_aluno, nome_modelo, tuple(lista))
    if chave_lista not in st.session_state.impressoes_registradas:
        registro().registrar_impressoes(st.session_state.sessao, id_aluno, nome_modelo, lista)
        st.session_state.impressoes_registradas.add(chave_lista)

    for rank, id_material in enumerate(lista, start=1):
        material = materiais.loc[id_material]
        with st.container(border=True):
            esquerda, direita = st.columns([4, 1.3])
            with esquerda:
                st.markdown(f"**{rank}. {material['titulo']}** — {material['autor']}  \n"
                            f"`{material['tipo']}` · `{material['area']}` · `{material['nivel']}` · material {id_material}")
                with st.expander("Por que este material?"):
                    for motivo in explicacoes(id_aluno, id_material, salvo, alunos, materiais, componentes_norm,
                                              posicao[id_material], regras):
                        st.write("• " + motivo)
            with direita:
                if st.button("Abrir material", key=f"abrir_{rank}_{id_material}", use_container_width=True):
                    registro().registrar(st.session_state.sessao, id_aluno, nome_modelo, "clique", int(id_material), rank)
                    st.session_state.aberto = (int(id_material), rank, nome_modelo, id_aluno)
                if st.session_state.aberto and st.session_state.aberto[0] == id_material:
                    minutos = st.number_input("Minutos de estudo", 1, 600, 30, key=f"min_{id_material}")
                    if st.button("Concluí", key=f"concluir_{id_material}", use_container_width=True):
                        registro().registrar(st.session_state.sessao, id_aluno, nome_modelo, "tempo", int(id_material),
                                             rank, float(minutos))
                        st.session_state.aberto = None
                        st.toast(f"{minutos} minutos registrados")

    with st.container(border=True):
        st.subheader("Pesquisa de satisfação")
        chave_resposta = chave_lista[:4]
        if chave_resposta in st.session_state.avaliacoes_enviadas:
            st.success("Obrigado! Sua avaliação desta lista foi registrada.")
        else:
            nota = st.radio("Quão úteis foram estas recomendações?", [1, 2, 3, 4, 5], index=3, horizontal=True,
                            format_func=lambda n: "★" * n, key=f"satisfacao_{'_'.join(map(str, chave_resposta[1:]))}")
            if st.button("Enviar avaliação"):
                registro().registrar(st.session_state.sessao, id_aluno, nome_modelo, "satisfacao", valor=float(nota))
                st.session_state.avaliacoes_enviadas.add(chave_resposta)
                st.rerun()

    if nome_modelo in ("Colaborativo", "Híbrido"):
        with st.expander("Alunos com padrão de avaliação parecido (fatores latentes da filtragem colaborativa)"):
            vizinhos = salvo["componentes"].colaborativo.vizinhos(id_aluno)
            if vizinhos.empty:
                st.write("Este aluno não tem histórico no treino.")
            else:
                vizinhos["perfil"] = [f"{alunos.at[v, 'curso']}, {alunos.at[v, 'periodo']}º per."
                                      if v in alunos.index else "sem perfil cadastrado" for v in vizinhos["id_aluno"]]
                st.dataframe(vizinhos, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------- abas de métricas
def tabela_csv(nome: str) -> pd.DataFrame | None:
    caminho = config.DIR_RESULTADOS / nome
    return pd.read_csv(caminho) if caminho.exists() else None


def figura(nome: str):
    caminho = config.DIR_FIGURAS / nome
    if caminho.exists():
        st.image(str(caminho), use_container_width=True)


def aba_offline():
    st.header("Métricas offline")
    st.caption("Divisão temporal: treino jan–out/2023, teste nov–dez/2023. Relevante = nota ≥ 4. "
               "Detalhes no relatório (relatorio/Relatorio.pdf).")
    colunas = ["modelo", "P@10", "R@10", "F1@10", "NDCG@10", "Cobertura@10", "Diversidade@10", "Obra P@3"]
    for prefixo, nome, arquivo_figura in (("real", "Dados fornecidos", "fig05_modelos_real.png"),
                                          ("controlado", "Sinal controlado (simulado)", "fig06_modelos_controlado.png")):
        st.subheader(nome)
        ranqueamento, notas = tabela_csv(f"{prefixo}_ranqueamento.csv"), tabela_csv(f"{prefixo}_notas.csv")
        if ranqueamento is None:
            st.info("Execute `python pipeline.py` para gerar os resultados.")
            continue
        figura(arquivo_figura)
        st.dataframe(ranqueamento[colunas].style.format(precision=4), hide_index=True, use_container_width=True)
        st.dataframe(notas.style.format(precision=4), hide_index=True, use_container_width=True)


def aba_online():
    st.header("Métricas online")
    st.subheader("Desta demonstração")
    st.caption("Calculadas a partir de artefatos/eventos_online.csv: cada lista exibida gera impressões; "
               "'Abrir material' gera clique; 'Concluí' registra o tempo; as estrelas registram a satisfação.")
    eventos = registro().carregar()
    if eventos.empty:
        st.info("Nenhum evento ainda: gere recomendações e interaja com elas na aba ao lado.")
    else:
        st.dataframe(metricas_online(eventos).style.format({"CTR": "{:.1%}", "minutos por sessão": "{:.1f}",
                                                            "satisfação (1–5)": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
    st.subheader("Teste A/B simulado (cenário com sinal controlado)")
    figura("fig09_online.png")
    ab = tabela_csv("online_ab.csv")
    if ab is not None:
        st.dataframe(ab.style.format(precision=3), hide_index=True, use_container_width=True)


def aba_dados():
    st.header("Qualidade dos dados")
    caminho = config.DIR_RESULTADOS / "qualidade_dados.json"
    if not caminho.exists():
        st.info("Execute `python pipeline.py --etapas analise`.")
        return
    qualidade = json.loads(caminho.read_text(encoding="utf-8"))
    q = qualidade["integridade"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interações", f"{q['n_interacoes']:,}".replace(",", "."))
    c2.metric("Alunos citados / cadastrados", f"{q['alunos_distintos_nas_interacoes']:,} / {q['n_alunos_cadastrados']:,}".replace(",", "."))
    c3.metric("Materiais citados / cadastrados", f"{q['materiais_distintos_nas_interacoes']:,} / {q['n_materiais_cadastrados']:,}".replace(",", "."))
    c4.metric("Obras distintas no catálogo", qualidade["catalogo"]["n_obras"])
    for nome in ("fig01_integridade.png", "fig02_distribuicoes.png", "fig03_estabilidade.png", "fig04_perfil_obra.png"):
        figura(nome)


def main():
    estado_inicial()
    st.title("📚 Recomendador híbrido de materiais didáticos")
    faltando = artefatos_ausentes()
    if faltando:
        st.error("Modelos ainda não treinados. Execute `python pipeline.py` na pasta do projeto. "
                 "Arquivos ausentes: " + ", ".join(faltando))
        st.stop()
    alunos, materiais = carregar_cadastros()
    opcoes = barra_lateral(alunos)
    abas = st.tabs(["Recomendações", "Métricas offline", "Métricas online", "Qualidade dos dados"])
    with abas[0]:
        aba_recomendacoes(opcoes, alunos, materiais)
    with abas[1]:
        aba_offline()
    with abas[2]:
        aba_online()
    with abas[3]:
        aba_dados()


main()

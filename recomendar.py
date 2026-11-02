"""Demonstração em linha de comando: recomendações explicadas para um aluno.

Uso (depois de `python pipeline.py`):
    python recomendar.py --aluno 2
    python recomendar.py --aluno 1 --modelo Conhecimento --k 5
    python recomendar.py --aluno 35000                  # aluno sem perfil cadastrado
    python recomendar.py --aluno 6 --cenario controlado
    python recomendar.py --aluno 1 --regras             # as regras de conhecimento para as 10 obras
"""
from __future__ import annotations

import argparse
import pickle
import sys
import textwrap

import numpy as np
import pandas as pd

from recomendador import config, dados, ontologia
from recomendador.modelos import COMPONENTES, Aleatorio

ARQUIVOS = {"real": config.ARQ_INTERACOES_CACHE, "controlado": config.DIR_ARTEFATOS / "interacoes_simuladas.parquet"}
MODELOS = ["Híbrido", "Conteúdo", "Colaborativo", "Conhecimento", "Popularidade", "Aleatório"]
LARGURA = 88   # colunas de texto da saída


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aluno", type=int, required=True)
    parser.add_argument("--modelo", choices=MODELOS, default="Híbrido")
    parser.add_argument("--k", type=int, default=config.K_PRINCIPAL)
    parser.add_argument("--cenario", choices=list(ARQUIVOS), default="real")
    parser.add_argument("--regras", action="store_true", help="mostra a avaliação das regras para cada obra")
    opcoes = parser.parse_args(argumentos)

    with open(config.DIR_ARTEFATOS / f"modelos_{opcoes.cenario}.pkl", "rb") as arquivo:
        salvo = pickle.load(arquivo)
    comp, hibrido = salvo["componentes"], salvo["hibrido"]
    modelos = dict(zip(MODELOS, [hibrido, comp.conteudo, comp.colaborativo, comp.conhecimento, comp.popularidade,
                                 Aleatorio()]))
    alunos, materiais = dados.carregar_alunos(), dados.carregar_materiais()
    interacoes = pd.read_parquet(ARQUIVOS[opcoes.cenario], columns=["id_aluno", "id_material", "avaliacao"],
                                 filters=[("id_aluno", "==", opcoes.aluno)])
    a = opcoes.aluno

    print(f"\nAluno {a}")
    if a in alunos.index:
        aluno = alunos.loc[a]
        print(f"  {aluno['curso']}, {aluno['periodo']}º período")
        print(f"  Disciplinas: {', '.join(aluno['disciplinas'])}")
        print(f"  Interesses:  {', '.join(aluno['interesses'])}")
        adquiridas = comp.conhecimento.competencias(a) - ontologia.competencias_declaradas(aluno["disciplinas"])
        print(f"  Competências pelo histórico: {', '.join(sorted(adquiridas)) or 'nenhuma'}")
    else:
        print("  sem perfil cadastrado (regras de conhecimento não se aplicam)")
    cadastrados = interacoes[interacoes["id_material"].isin(materiais.index)]
    print(f"  Histórico: {len(interacoes)} interações, {len(cadastrados)} com material cadastrado")

    candidatos = materiais.index.to_numpy()
    vistos = np.isin(candidatos, interacoes["id_material"].to_numpy())[None, :]
    lista = modelos[opcoes.modelo].recomendar(np.array([a]), candidatos, opcoes.k, vistos)[0]
    if opcoes.modelo == "Híbrido":
        pesos = hibrido.pesos_efetivos(np.array([a]))[0]
        nomes = {"conteudo": "conteúdo", "colaborativo": "colaborativo", "conhecimento": "conhecimento"}
        print("  Pesos efetivos: " + " · ".join(f"{nomes[c]} {p:.0%}" for c, p in zip(COMPONENTES, pesos)))
    regras = comp.conhecimento.avaliar_obras(a).set_index("id_obra") if a in alunos.index else None

    def recuado(texto: str, margem: int) -> str:
        return textwrap.fill(texto, width=LARGURA, initial_indent=" " * margem, subsequent_indent=" " * margem)

    if opcoes.regras and regras is not None:
        print("\nRegras de conhecimento por obra (pontuação 0–1; 0 = pré-requisito faltante)\n")
        for linha in regras.sort_values("pontuacao", ascending=False).itertuples():
            print(f"  {linha.pontuacao:.2f}  {linha.titulo}".replace(".", ",", 1))
            print(recuado(" · ".join(linha.justificativas), 8))

    print(f"\nTop {opcoes.k} — {opcoes.modelo}\n")
    for posicao, m in enumerate(lista, start=1):
        material = materiais.loc[m]
        nota = comp.colaborativo.prever_avaliacao(np.array([a]), np.array([m]))[0]
        print(f"{posicao:>2}. {material['titulo']} — {material['tipo']}, {material['area']}, {material['nivel']} "
              f"(material {m})")
        motivos = [f"nota prevista {nota:.1f}".replace(".", ",")]
        if regras is not None:
            motivos += regras.at[int(material["id_obra"]), "justificativas"]
        print(recuado(" · ".join(motivos), 6))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

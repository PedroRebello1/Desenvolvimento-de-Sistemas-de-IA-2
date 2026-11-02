"""Pipeline completo do sistema de recomendação híbrido.

Uso:
    python pipeline.py                          # todas as etapas
    python pipeline.py --etapas analise real    # só as etapas indicadas

Etapas (nesta ordem):
    analise     qualidade dos dados e testes estatísticos       -> resultados/qualidade_dados.json, ...
    real        experimento com as interações fornecidas         -> resultados/real_*, artefatos/modelos_real.pkl
    controlado  experimento com sinal controlado (simulado)      -> resultados/controlado_*, artefatos/modelos_controlado.pkl
    online      teste A/B simulado sobre o cenário controlado    -> resultados/online_ab*.csv
    figuras     gráficos a partir dos resultados                 -> resultados/figuras/*.png
    relatorio   tabelas Markdown + Relatorio.docx/.pdf (Quarto)  -> relatorio/
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from recomendador import analise, config, dados, experimentos, simulacao

ETAPAS = ["analise", "real", "controlado", "online", "figuras", "relatorio"]


def _json(objeto, caminho):
    def converter(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=converter), encoding="utf-8")


def _salvar_cenario(prefixo: str, resultado: experimentos.ResultadoCenario) -> None:
    pasta = config.DIR_RESULTADOS
    ranqueamento = resultado.ranqueamento.copy()
    for coluna in [c for c in ranqueamento.columns if c.endswith("IC95")]:
        ranqueamento[coluna + " inf"] = ranqueamento[coluna].map(lambda t: t[0])
        ranqueamento[coluna + " sup"] = ranqueamento[coluna].map(lambda t: t[1])
        ranqueamento = ranqueamento.drop(columns=coluna)
    ranqueamento.to_csv(pasta / f"{prefixo}_ranqueamento.csv", index=False)
    resultado.notas.to_csv(pasta / f"{prefixo}_notas.csv", index=False)
    resultado.grade_pesos.to_csv(pasta / f"{prefixo}_grade_pesos.csv", index=False)
    resultado.curva_mmr.to_csv(pasta / f"{prefixo}_curva_mmr.csv", index=False)
    resultado.historico_fc.to_csv(pasta / f"{prefixo}_historico_fc.csv", index=False)
    if resultado.cold_start is not None:
        cold = resultado.cold_start.drop(columns=[c for c in resultado.cold_start.columns if c.endswith("IC95")])
        cold.to_csv(pasta / f"{prefixo}_cold_start.csv", index=False)
    resultado.comparacoes.to_csv(pasta / f"{prefixo}_comparacoes.csv", index=False)
    _json({"pesos": resultado.pesos, "coef_notas": resultado.coef_notas, **resultado.info},
          pasta / f"{prefixo}_resumo.json")


def _salvar_modelos(prefixo: str, resultado: experimentos.ResultadoCenario, **extras) -> None:
    with open(config.DIR_ARTEFATOS / f"modelos_{prefixo}.pkl", "wb") as arquivo:
        pickle.dump({"componentes": resultado.componentes, "hibrido": resultado.hibrido, **extras}, arquivo,
                    protocol=pickle.HIGHEST_PROTOCOL)


def etapa_analise() -> None:
    base = dados.carregar_base()
    resultado = analise.analisar(base)
    catalogo = resultado.pop("catalogo")
    catalogo["obras"].to_csv(config.DIR_RESULTADOS / "obras.csv", index=False)
    catalogo["perfis"].to_csv(config.DIR_RESULTADOS / "perfis.csv", index=False)
    media = resultado["sinal"].pop("media_perfil_obra")
    media.to_csv(config.DIR_RESULTADOS / "media_perfil_obra.csv")
    resultado["catalogo"] = {"n_obras": catalogo["n_obras"], "n_perfis": catalogo["n_perfis"]}
    _json(resultado, config.DIR_RESULTADOS / "qualidade_dados.json")
    print(json.dumps(resultado["integridade"], ensure_ascii=False))


def etapa_real() -> None:
    base = dados.carregar_base()
    resultado = experimentos.executar_cenario("Dados fornecidos", base, base.interacoes)
    _salvar_cenario("real", resultado)
    _salvar_modelos("real", resultado)


def _simulador() -> simulacao.SimuladorPreferencias:
    alunos, materiais = dados.carregar_alunos(), dados.carregar_materiais()
    vazio = pd.DataFrame({"id_aluno": np.array([], dtype=np.int32), "id_material": np.array([], dtype=np.int32)})
    return simulacao.SimuladorPreferencias(dados.Base(alunos, materiais, vazio))


def etapa_controlado() -> None:
    simulador = _simulador()
    interacoes = simulador.gerar_interacoes()
    interacoes.to_parquet(config.DIR_ARTEFATOS / "interacoes_simuladas.parquet", index=False)
    base = dados.Base(simulador.base.alunos, simulador.base.materiais, interacoes)
    resultado = experimentos.executar_cenario("Sinal controlado", base, interacoes, cold_start=True,
                                              modelos_extras=(simulacao.Oraculo(simulador),))
    _salvar_cenario("controlado", resultado)
    _salvar_modelos("controlado", resultado, simulador=simulador)


def etapa_online() -> None:
    with open(config.DIR_ARTEFATOS / "modelos_controlado.pkl", "rb") as arquivo:
        salvo = pickle.load(arquivo)
    comp, simulador = salvo["componentes"], salvo["simulador"]
    interacoes = pd.read_parquet(config.DIR_ARTEFATOS / "interacoes_simuladas.parquet")
    treino = interacoes[interacoes["dia"] < dados.dia_de(config.DATA_CORTE_TESTE)]
    modelos = [experimentos.Aleatorio(), comp.popularidade, comp.conteudo, comp.colaborativo, comp.conhecimento,
               salvo["hibrido"], simulacao.Oraculo(simulador)]
    resumo, sessoes = simulacao.simular_teste_ab(simulador, modelos, treino)
    resumo.to_csv(config.DIR_RESULTADOS / "online_ab.csv", index=False)
    sessoes.to_csv(config.DIR_RESULTADOS / "online_ab_sessoes.csv", index=False)
    print(resumo.round(4).to_string())


def etapa_figuras() -> None:
    from recomendador import figuras
    figuras.gerar_todas()


def etapa_relatorio() -> None:
    from recomendador import relatorio
    relatorio.gerar()


def main(argumentos=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS, default=ETAPAS)
    etapas = parser.parse_args(argumentos).etapas
    for pasta in (config.DIR_ARTEFATOS, config.DIR_RESULTADOS, config.DIR_FIGURAS):
        pasta.mkdir(parents=True, exist_ok=True)
    for nome in ETAPAS:
        if nome in etapas:
            inicio = time.time()
            print(f"\n=== {nome} ===", flush=True)
            globals()[f"etapa_{nome}"]()
            gc.collect()
            print(f"=== {nome}: {time.time() - inicio:.0f}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

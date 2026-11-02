import numpy as np
import pandas as pd
import pytest

from recomendador import dados, ontologia
from recomendador.modelos import FiltragemConteudo, ModeloConhecimento


@pytest.fixture(scope="module")
def cadastros():
    return dados.carregar_alunos(), dados.carregar_materiais()


def test_normalizacao_de_abreviacoes():
    assert dados.separar_lista("BD, IA, Web") == ["Banco de Dados", "Inteligência Artificial", "Web"]
    assert dados.separar_lista("Banco de Dados, BD") == ["Banco de Dados"]


def test_prerequisitos():
    regra = ontologia.regra_da_obra("ML Avançado")
    completo = ontologia.competencias_declaradas(["Estatística", "Python", "Machine Learning"])
    sem_estatistica = ontologia.competencias_declaradas(["Algoritmos", "Banco de Dados", "Inteligência Artificial"])
    assert ontologia.prerequisitos_faltantes(regra, completo) == []
    assert ontologia.prerequisitos_faltantes(regra, sem_estatistica) == [frozenset({"Estatística"})]


def test_adequacao_de_nivel_por_periodo():
    assert ontologia.adequacao_nivel(0, periodo=2) == 1.0      # iniciante no 2º período
    assert ontologia.adequacao_nivel(2, periodo=2) == 0.2      # avançado no 2º período
    assert ontologia.adequacao_nivel(2, periodo=7) == 1.0


def test_regras_por_perfil(cadastros):
    alunos, materiais = cadastros
    modelo = ModeloConhecimento(alunos, materiais, n_usuarios=10_000, n_itens=10_000).treinar(
        pd.DataFrame(columns=["id_aluno", "id_material", "avaliacao"], dtype=int))
    # aluno 2: Ciência de Dados, cursou Estatística, Python e ML -> pode ML Avançado, não pode NoSQL
    tabela = modelo.avaliar_obras(2).set_index("titulo")
    assert tabela.at["ML Avançado", "elegivel"]
    assert not tabela.at["Banco de Dados NoSQL", "elegivel"]
    assert tabela.at["Banco de Dados NoSQL", "pontuacao"] == 0
    # aluno 1: Eng. de Software sem Estatística -> Estatística Básica desbloqueia ML Avançado
    tabela = modelo.avaliar_obras(1).set_index("titulo")
    assert not tabela.at["ML Avançado", "elegivel"]
    assert "ML Avançado" in tabela.at["Estatística Básica", "desbloqueia"]


def test_historico_concede_competencias(cadastros):
    alunos, materiais = cadastros
    modelo = ModeloConhecimento(alunos, materiais, n_usuarios=10_000, n_itens=10_000)
    # aluno 1 conclui Estatística Básica (material 4) com nota 4 -> ML Avançado passa a ser elegível
    modelo.treinar(pd.DataFrame({"id_aluno": [1], "id_material": [4], "avaliacao": [4]}))
    assert modelo.avaliar_obras(1).set_index("titulo").at["ML Avançado", "elegivel"]


def test_conteudo_aproxima_interesse_e_area(cadastros):
    alunos, materiais = cadastros
    modelo = FiltragemConteudo(alunos, materiais, n_usuarios=10_000, n_itens=10_000)
    # aluno 6 (interesses NLP e Deep Learning): Deep Learning (material 5) supera Gestão Ágil (material 6)
    pontuacoes = modelo.pontuar(np.array([6]), np.array([5, 6]))[0]
    assert pontuacoes[0] > pontuacoes[1]
    assert "learning" in modelo.termos_em_comum(6, 5)

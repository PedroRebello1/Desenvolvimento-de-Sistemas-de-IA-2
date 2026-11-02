"""Base de conhecimento do domínio acadêmico.

Tudo aqui é conhecimento explícito, escrito à mão (não aprendido dos dados):
- a relação entre termos do perfil (disciplinas, interesses) e as áreas do catálogo;
- a relevância de cada área para cada curso;
- os pré-requisitos de cada obra e as competências que ela ensina;
- o nível de dificuldade esperado para cada período.

É usada pelo modelo de conhecimento (regras) e pelo modelo de conteúdo (para
projetar o perfil declarado do aluno no espaço de áreas dos materiais).
"""
from __future__ import annotations

from dataclasses import dataclass

AREAS = ["Programação", "IA", "Estatística", "BD", "Gestão", "Segurança", "UX", "IoT"]
NIVEIS = ["Iniciante", "Intermediário", "Avançado"]
TIPOS = ["livro", "artigo", "vídeo"]

# Termo do perfil (nome canônico, ver dados.CANONICO) -> {área do catálogo: peso 0..1}.
TERMO_AREAS: dict[str, dict[str, float]] = {
    # disciplinas
    "Algoritmos": {"Programação": 1.0},
    "Banco de Dados": {"BD": 1.0},
    "Inteligência Artificial": {"IA": 1.0},
    "Estatística": {"Estatística": 1.0},
    "Python": {"Programação": 1.0, "IA": 0.3},
    "Machine Learning": {"IA": 1.0, "Estatística": 0.5},
    "Redes": {"Segurança": 0.7, "IoT": 0.5},
    "Sistemas Operacionais": {"Programação": 0.5, "Segurança": 0.4},
    "Arquitetura de Computadores": {"IoT": 0.8},
    "Gestão": {"Gestão": 1.0},
    "Web": {"Programação": 0.8, "UX": 0.5},
    "Compiladores": {"Programação": 0.9},
    "UX": {"UX": 1.0},
    "Gestão de Projetos": {"Gestão": 1.0},
    # interesses
    "Programação": {"Programação": 1.0},
    "Cloud": {"BD": 0.5, "Programação": 0.5, "Segurança": 0.4},
    "Visualização": {"Estatística": 0.8, "UX": 0.5},
    "Big Data": {"BD": 0.8, "Estatística": 0.6, "IA": 0.4},
    "IoT": {"IoT": 1.0},
    "Segurança": {"Segurança": 1.0},
    "Hardware": {"IoT": 0.9},
    "Analytics": {"Estatística": 0.9, "BD": 0.4},
    "Mobile": {"Programação": 0.8, "UX": 0.5},
    "DevOps": {"Programação": 0.6, "Gestão": 0.4, "Segurança": 0.3},
    "Qualidade de Software": {"Programação": 0.6, "Gestão": 0.5},
    "NLP": {"IA": 1.0},
    "Deep Learning": {"IA": 1.0},
    "Robótica": {"IoT": 0.8, "IA": 0.5},
    "Embedded": {"IoT": 1.0},
    "Marketing Digital": {"Gestão": 0.6, "UX": 0.6, "Estatística": 0.3},
}

# Relevância curricular de cada área do catálogo para cada curso (0..1).
CURSO_AREAS: dict[str, dict[str, float]] = {
    "Engenharia de Software": {"Programação": 1.0, "BD": 0.8, "Gestão": 0.6, "UX": 0.5, "IA": 0.4,
                               "Segurança": 0.3, "Estatística": 0.2, "IoT": 0.1},
    "Ciência de Dados": {"IA": 1.0, "Estatística": 1.0, "Programação": 0.8, "BD": 0.7, "UX": 0.2,
                         "Gestão": 0.2, "Segurança": 0.1, "IoT": 0.1},
    "Engenharia de Computação": {"IoT": 1.0, "Programação": 0.8, "Segurança": 0.7, "IA": 0.4, "BD": 0.3,
                                 "Estatística": 0.3, "Gestão": 0.1, "UX": 0.1},
    "Sistemas de Informação": {"Gestão": 1.0, "BD": 0.8, "UX": 0.7, "Programação": 0.5, "Estatística": 0.5,
                               "Segurança": 0.4, "IA": 0.3, "IoT": 0.1},
}

# Competências implícitas: cursar Algoritmos também comprova base de programação, etc.
IMPLICACOES: dict[str, set[str]] = {
    "Algoritmos": {"Programação"},
    "Python": {"Programação"},
    "Web": {"Programação"},
    "Compiladores": {"Programação"},
    "Machine Learning": {"Inteligência Artificial"},
    "Gestão de Projetos": {"Gestão"},
    "Arquitetura de Computadores": {"Hardware"},
}


@dataclass(frozen=True)
class RegraObra:
    """`prerequisitos`: todos os grupos são exigidos; dentro de um grupo, basta uma competência."""
    prerequisitos: tuple[frozenset[str], ...] = ()
    ensina: frozenset[str] = frozenset()


def _regra(*grupos: set[str], ensina: set[str]) -> RegraObra:
    return RegraObra(tuple(frozenset(g) for g in grupos), frozenset(ensina))


REGRAS_OBRA: dict[str, RegraObra] = {
    "Introdução a Algoritmos": _regra(ensina={"Algoritmos", "Programação"}),
    "Aprendendo Python": _regra(ensina={"Python", "Programação"}),
    "Estatística Básica": _regra(ensina={"Estatística"}),
    "ML Avançado": _regra({"Machine Learning", "Inteligência Artificial"}, {"Estatística"}, {"Programação"},
                          ensina={"Machine Learning", "Machine Learning Avançado"}),
    "Deep Learning": _regra({"Programação"}, {"Machine Learning", "Inteligência Artificial", "Estatística"},
                            ensina={"Deep Learning", "Inteligência Artificial"}),
    "Gestão Ágil": _regra(ensina={"Metodologias Ágeis"}),
    "Segurança em Redes": _regra({"Redes"}, ensina={"Segurança de Redes"}),
    "UX Moderno": _regra(ensina={"UX"}),
    "Banco de Dados NoSQL": _regra({"Banco de Dados"}, ensina={"NoSQL"}),
    "IoT Prático": _regra(ensina={"IoT"}),
}


def regra_da_obra(titulo: str) -> RegraObra:
    """Obras fora da base de conhecimento não têm pré-requisitos e ensinam o próprio título."""
    return REGRAS_OBRA.get(titulo, RegraObra(ensina=frozenset({titulo})))


def competencias_declaradas(disciplinas: list[str]) -> set[str]:
    competencias = set(disciplinas)
    for disciplina in disciplinas:
        competencias |= IMPLICACOES.get(disciplina, set())
    return competencias


def prerequisitos_faltantes(regra: RegraObra, competencias: set[str]) -> list[frozenset[str]]:
    return [grupo for grupo in regra.prerequisitos if not grupo & competencias]


def afinidade_areas(termos: list[str]) -> dict[str, float]:
    """Afinidade de um conjunto de termos com cada área (máximo entre os termos)."""
    afinidade = dict.fromkeys(AREAS, 0.0)
    for termo in termos:
        for area, peso in TERMO_AREAS.get(termo, {}).items():
            afinidade[area] = max(afinidade[area], peso)
    return afinidade


def nivel_esperado(periodo: int) -> int:
    """1º–2º período: Iniciante; 3º–5º: Intermediário; 6º em diante: Avançado."""
    if periodo <= 2:
        return 0
    return 1 if periodo <= 5 else 2


def adequacao_nivel(nivel: int, periodo: int) -> float:
    """1 no nível esperado; 0,6 um degrau acima ou abaixo; 0,2 dois degraus."""
    return {0: 1.0, 1: 0.6, 2: 0.2}[abs(nivel - nivel_esperado(periodo))]

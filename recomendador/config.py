"""Configurações centrais: caminhos, protocolo de avaliação e hiperparâmetros."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Os três CSVs fornecidos ficam na raiz do projeto.
ARQ_ALUNOS = RAIZ / "dados_alunos.csv"
ARQ_MATERIAIS = RAIZ / "materiais_didaticos.csv"
ARQ_INTERACOES = RAIZ / "interacoes.csv"

# Saídas geradas pelo pipeline.
DIR_ARTEFATOS = RAIZ / "artefatos"        # cache de dados e modelos treinados
DIR_RESULTADOS = RAIZ / "resultados"      # tabelas e métricas
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
ARQ_INTERACOES_CACHE = DIR_ARTEFATOS / "interacoes.parquet"
ARQ_EVENTOS_ONLINE = DIR_ARTEFATOS / "eventos_online.csv"

SEMENTE = 42

# --- Protocolo de avaliação offline -------------------------------------------
# Divisão temporal: treino = jan–out/2023, teste = nov–dez/2023.
# Para ajustar os pesos do híbrido, o treino é subdividido: jan–ago / set–out.
DATA_CORTE_TESTE = "2023-11-01"
DATA_CORTE_VALIDACAO = "2023-09-01"
LIMIAR_RELEVANCIA = 4          # avaliação >= 4 => material relevante para o aluno
LISTA_K = (5, 10, 20)
K_PRINCIPAL = 10

# --- Filtragem colaborativa (fatoração de matrizes com vieses, ALS) -----------
FC_FATORES = 8
FC_ITERACOES = 40
FC_REGULARIZACAO = 5.0         # ridge dos fatores; valores altos colapsam os fatores em zero
FC_REGULARIZACAO_VIES = 10.0
FC_ESCALA_INICIAL = 0.1
FC_PACIENCIA = 8               # iterações sem melhora na validação antes de parar

# --- Híbrido ------------------------------------------------------------------
PESOS_PADRAO = {"conteudo": 0.4, "colaborativo": 0.3, "conhecimento": 0.3}
PESO_MINIMO = 0.1              # na busca em grade, nenhum componente é desligado
LAMBDA_MMR = 0.8               # 1.0 = só relevância; menor = mais diversidade
PENALIDADE_INELEGIVEL = 0.5    # material sem pré-requisitos cumpridos desce na lista
HISTORICO_MINIMO_FC = 5        # abaixo disso o peso colaborativo é reduzido (cold start)

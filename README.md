# Sistema de Recomendação Híbrido para Plataforma de Ensino

Recomenda materiais didáticos (livros, artigos e vídeos) a alunos de ensino superior combinando
**filtragem por conteúdo**, **filtragem colaborativa** e um **modelo baseado em conhecimento**
(regras de pré-requisitos e requisitos acadêmicos). Projeto prático da AV2 de Desenvolvimento de Sistemas de IA (CEUB).

O relatório completo está em [`relatorio/Relatorio.pdf`](relatorio/Relatorio.pdf)
(versão editável: [`relatorio/Relatorio.docx`](relatorio/Relatorio.docx)).

## Como executar

> [!IMPORTANT]
> O arquivo `interacoes.csv` (466 MB) é grande demais para o GitHub (limite de 100 MB por arquivo)
> e por isso **não está no repositório**. Antes de executar, insira-o manualmente na raiz do projeto,
> ao lado de `dados_alunos.csv` e `materiais_didaticos.csv`. Sem ele, o `pipeline.py` não roda.

Requer Python 3.10+.

```bash
pip install -r requirements.txt

python pipeline.py              # análise, experimentos, teste A/B simulado, figuras e relatório
streamlit run app.py            # demonstração funcional (abre no navegador)
python recomendar.py --aluno 1 --regras   # demonstração na linha de comando
python -m pytest                # testes automatizados
```

A demonstração aceita a configuração inicial pela URL, por exemplo
`http://localhost:8501/?cenario=controlado&aluno=6&modelo=Híbrido`.

O pipeline completo leva cerca de 12 minutos num notebook e grava tudo localmente. Etapas isoladas:

```bash
python pipeline.py --etapas analise          # qualidade dos dados e testes estatísticos
python pipeline.py --etapas real             # experimento com as interações fornecidas
python pipeline.py --etapas controlado       # experimento com sinal controlado (simulado)
python pipeline.py --etapas online           # teste A/B simulado (CTR, engajamento, satisfação)
python pipeline.py --etapas figuras relatorio
```

A etapa `relatorio` usa o [Quarto](https://quarto.org) para gerar o .docx e o .pdf; sem ele,
as tabelas são geradas e só a renderização é pulada.

## Estrutura

```
dados_alunos.csv, materiais_didaticos.csv, interacoes.csv   dados fornecidos
pipeline.py               executa todas as etapas
app.py                    demonstração em Streamlit
recomendar.py             demonstração na linha de comando
recomendador/
  config.py               caminhos, protocolo de avaliação e hiperparâmetros
  dados.py                carga, normalização, cache Parquet, divisão temporal
  analise.py              qualidade dos dados e testes estatísticos
  ontologia.py            base de conhecimento: termo→área, curso→área, pré-requisitos
  modelos/
    baselines.py          aleatório, popularidade, média global, vieses μ + b_u + b_i
    conteudo.py           TF-IDF + áreas + formato; perfil Rocchio; kNN de conteúdo
    colaborativo.py       fatoração de matrizes com vieses, treinada por ALS
    conhecimento.py       regras: pré-requisitos, nível, curso, novidade, trilha
    hibrido.py            ponderação + troca (cold start) + restrição + MMR + empilhamento
  avaliacao.py            Precisão/Recall/F1/NDCG@K, RMSE, cobertura, diversidade, IC
  experimentos.py         validação (pesos, iterações) → treino final → teste
  simulacao.py            cenário com sinal controlado e teste A/B simulado
  online.py               registro de eventos e métricas online da demonstração
  figuras.py, relatorio.py
tests/                    testes automatizados (pytest)
relatorio/
  relatorio.qmd           fonte do relatório (Quarto); estilo.typ ajusta o PDF
  capturas/               captura de tela da demonstração
  tabelas/, figuras/, variaveis.yml   gerados pela etapa `relatorio`
  Relatorio.pdf, Relatorio.docx       o relatório renderizado
resultados/               métricas em CSV/JSON e figuras (gerados)
artefatos/                cache Parquet, modelos treinados e log de eventos (gerados)
```

## O que os dados mostraram

As interações fornecidas não têm sinal aprendível: 90% citam alunos e 80% citam materiais
inexistentes nos cadastros, todas as colunas são uniformes, e nem o aluno, nem o material, nem a
combinação perfil × obra afetam a nota (testes no relatório, seção 2). Os 10 mil alunos são 8 perfis
repetidos e os 10 mil materiais são 10 obras repetidas. Por isso o relatório avalia o sistema em
dois cenários: os dados fornecidos (todos os modelos empatam com o acaso, como deveriam) e um
cenário com sinal controlado — mesmos alunos e materiais, interações simuladas a partir de uma
utilidade conhecida — que confirma que cada componente captura o sinal para o qual foi projetado e
que o híbrido supera cada um deles.

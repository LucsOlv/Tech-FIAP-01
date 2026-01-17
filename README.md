# Sistema de Diagnóstico de Hipertensão com Machine Learning

**Tech Challenge - Fase 1 - FIAP**

Sistema inteligente de suporte ao diagnóstico de hipertensão utilizando técnicas de Machine Learning para análise de dados clínicos e demográficos de pacientes.

---

## Descrição do Projeto

Este projeto faz parte do Tech Challenge da FIAP e tem como objetivo desenvolver um sistema de IA capaz de auxiliar médicos e equipes clínicas na análise inicial de exames e diagnóstico de hipertensão.

O sistema utiliza algoritmos de Machine Learning para classificação binária de pacientes (portadores ou não de hipertensão), baseando-se em variáveis clínicas e demográficas:

- Idade
- Consumo de sal
- Nível de estresse
- Histórico de pressão arterial
- Duração do sono
- IMC (Índice de Massa Corporal)
- Medicação atual
- Histórico familiar
- Nível de exercício físico
- Status de fumante

---

## Objetivos

1. **Análise Exploratória**: Compreender padrões e correlações nos dados clínicos
2. **Pré-processamento**: Limpar e preparar dados para modelagem
3. **Modelagem**: Treinar e comparar modelos de Machine Learning
4. **Avaliação**: Medir performance com métricas adequadas ao contexto médico
5. **Interpretabilidade**: Identificar quais variáveis são mais importantes para o diagnóstico

---

## Instalação e Execução

### Opção 1: Execução Local

#### 1. Criar ambiente virtual (recomendado)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OU
.venv\Scripts\activate  # Windows
```

#### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

#### 3. Executar Jupyter Notebook
```bash
jupyter notebook index.ipynb
```

#### 4. Executar todas as células
No Jupyter: `Kernel → Restart & Run All`

---

### Opção 2: Execução com Docker

#### 1. Construir a imagem
```bash
docker build -t hipertensao-ml .
```

#### 2. Executar o container
```bash
docker run -p 8888:8888 hipertensao-ml
```

#### 3. Acessar o notebook
Copiar o link com token que aparecer no terminal.
Exemplo: `http://127.0.0.1:8888/?token=abc123...`

---

## Modelos Implementados

### 1. Regressão Logística
- **Tipo**: Modelo linear
- **Vantagens**: Simples, interpretável, rápido
- **Uso**: Baseline e comparação

### 2. Árvore de Decisão
- **Tipo**: Modelo baseado em regras
- **Vantagens**: Captura não-linearidades, feature importance
- **Configuração**: `max_depth=5`, `min_samples_split=20`

---

## Estrutura do Projeto

```
projeto-hipertensao/
├── data/
│   └── hypertension_dataset.csv       # Dataset original
├── index.ipynb                        # Notebook principal
├── Dockerfile                         # Container Docker
├── requirements.txt                   # Dependências Python
├── README.md                          # Este arquivo
├── RELATORIO_TECNICO.md              # Relatório detalhado
└── .gitignore                         # Arquivos ignorados pelo Git
```

---

## Métricas de Avaliação

O projeto utiliza as seguintes métricas para avaliar os modelos:

- **Acurácia**: Percentual total de acertos
- **Precisão**: Proporção de predições positivas corretas
- **Recall (Sensibilidade)**: Proporção de casos positivos identificados
- **F1-Score**: Média harmônica entre precisão e recall

A escolha das métricas considera o contexto médico, onde falsos negativos (não identificar hipertensão) são particularmente críticos.

---

## Resultados

Os modelos apresentaram performance comparável, com destaque para:

- Identificação das principais variáveis preditoras (idade, BMI, histórico familiar, pressão arterial prévia)
- Análise de Feature Importance revelando padrões consistentes com literatura médica
- Discussão crítica sobre limitações e viabilidade de uso prático

Detalhes completos no arquivo `RELATORIO_TECNICO.md`.

---

## Limitações e Considerações Éticas

### Limitações Técnicas
- Dataset limitado (1.985 registros)
- Valores ausentes em variável "Medication" (~40%)
- Ausência de variáveis como etnia e região geográfica
- Necessidade de validação externa

### Considerações Éticas
- Viés algorítmico em subgrupos populacionais
- Responsabilidade médica sempre com profissional habilitado
- Conformidade com LGPD/HIPAA para dados de saúde
- Transparência e explicabilidade das decisões

---

## Aviso Importante

Este sistema é uma ferramenta educacional e de apoio à decisão.

**NÃO deve ser utilizado como substituto de diagnóstico médico profissional.**

Sempre consulte um médico qualificado para avaliação e tratamento de condições de saúde.

---

## Autor

Desenvolvido como parte do Tech Challenge - Fase 1 - FIAP

---

## Licença

Este projeto é de uso acadêmico.

# TECH CHALLENGE - FASE 1
## FIAP - Pós-Graduação em Inteligência Artificial

---

### INFORMAÇÕES DO PROJETO

**Título do Projeto:**  
Sistema de Diagnóstico de Hipertensão com Machine Learning

**Descrição:**  
Sistema inteligente de suporte ao diagnóstico de hipertensão utilizando técnicas de Machine Learning para classificação binária de pacientes baseado em dados clínicos e demográficos.


### LINK DO REPOSITÓRIO GITHUB

**URL:**  
```
https://github.com/LucsOlv/Tech-FIAP-01/tree/main
```

**Branch Principal:** `main`

---

### CONTEÚDO DO REPOSITÓRIO

O repositório contém os seguintes arquivos e diretórios:

```
projeto-hipertensao/
├── data/
│   └── hypertension_dataset.csv       # Dataset original
├── index.ipynb                        # Notebook principal com análise completa
├── Dockerfile                         # Container Docker para execução
├── requirements.txt                   # Dependências Python
├── README.md                          # Documentação do projeto
├── RELATORIO_TECNICO.md              # Relatório técnico detalhado
└── .gitignore                         # Arquivos ignorados pelo Git
```

---

### LINKS ADICIONAIS

**Vídeo de Demonstração (YouTube/Vimeo):**  
```
https://www.youtube.com/watch?v=Criw6iUJc-Y
```
---

### INSTRUÇÕES DE EXECUÇÃO

#### Opção 1: Execução Local
```bash
# Clonar repositório
git clone https://github.com/LucsOlv/Tech-FIAP-01/tree/main
cd Tech-FIAP-01

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OU .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar Jupyter Notebook
jupyter notebook index.ipynb
```

#### Opção 2: Execução com Docker
```bash
# Clonar repositório
git clone https://github.com/[seu-usuario]/[nome-do-repositorio]
cd [nome-do-repositorio]

# Construir imagem
docker build -t hipertensao-ml .

# Executar container
docker run -p 8888:8888 hipertensao-ml

# Acessar: copiar link com token do terminal
```

---

### RESUMO DO PROJETO

**Problema:** Triagem automatizada de hipertensão em ambiente hospitalar

**Dataset:** 1.985 registros de pacientes com 10 variáveis preditoras

**Modelos Treinados:**
- Regressão Logística
- Árvore de Decisão

**Métricas Avaliadas:**
- Acurácia
- Precisão
- Recall (Sensibilidade)
- F1-Score

**Principais Resultados:**
- Identificação dos principais fatores de risco (idade, IMC, histórico familiar, BP_History)
- Análise crítica de limitações e viabilidade prática
- Discussão de aspectos éticos e regulatórios para uso em saúde

---

### TECNOLOGIAS UTILIZADAS

- **Linguagem:** Python 3.9
- **Machine Learning:** Scikit-learn 1.2+
- **Manipulação de Dados:** Pandas 2.0+, NumPy 1.24+
- **Visualização:** Matplotlib 3.7+, Seaborn 0.12+
- **Ambiente:** Jupyter Notebook, Docker
- **Controle de Versão:** Git, GitHub

---

**Data de Entrega:** 17/01/2026

---

**Para avaliação completa, consultar:**
1. Código-fonte no repositório GitHub (link acima)
2. Relatório técnico (`RELATORIO_TECNICO.md`)
3. Vídeo de demonstração (link acima)


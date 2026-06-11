# 🩺 Classificação de Condições da Coluna Vertebral

Este projeto implementa modelos de machine learning para classificar pacientes com condições ortopédicas da coluna vertebral (Normal vs. Anormal) com base em 6 medidas biomecânicas.

## Dataset

O dataset **Vertebral Column** (UCI Machine Learning Repository) contém 310 amostras, cada uma com 6 atributos numéricos contínuos:
- `pelvic_incidence`
- `pelvic_tilt`
- `lumbar_lordosis_angle`
- `sacral_slope`
- `pelvic_radius`
- `degree_spondylolisthesis`

**Rótulos** (2 classes):
- `Normal` (100 amostras)
- `Abnormal` (210 amostras)

## Objetivo

Desenvolver um modelo de classificação capaz de distinguir entre pacientes normais e anormais, auxiliando no diagnóstico de problemas ortopédicos como hérnia de disco e espondilolistese.

## Metodologia

1. **Análise Exploratória**: distribuição das classes, boxplots, matriz de correlação.
2. **Pré-processamento**:
   - Divisão treino/teste (70/30, estratificada)
   - Normalização com `StandardScaler` (dados numéricos)
3. **Otimização de hiperparâmetros** com `GridSearchCV` (validação cruzada 5-fold) para três modelos:
   - K‑Nearest Neighbors (KNN)
   - Árvore de Decisão
   - Naive Bayes (GaussianNB)
4. **Avaliação**:
   - Acurácia, relatório de classificação, matriz de confusão
   - Comparação gráfica dos modelos
5. **Função de predição** com suavização de Laplace para probabilidades mais realistas.

## Resultados

| Modelo             | Acurácia (teste) |
|--------------------|------------------|
| Naive Bayes        | ~78%             |
| Árvore de Decisão  | ~83%             |
| KNN                | ~80%             |

A árvore de decisão apresentou a melhor acurácia e foi escolhida para a função de predição.

## Tecnologias

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Estrutura do Projeto

```
Projeto-Final-IA-Vertebral-Column/
├── Projeto_Final_IA_Vertebral_Column.ipynb			# Notebook com todo o código
├── column_2C.csv									# Dataset Vertebral Column
└── README.md										# Este arquivo
```

## ▶️ Como executar

1. **Abra o notebook Jupyter:**
   ```bash
   jupyter notebook Projeto_Final_IA_Vertebral_Column.ipynb
   ```

2. Execute o notebook `Projeto_Final_IA_Vertebral_Column.ipynb` (as células estão organizadas sequencialmente).

3. Para testar novos pacientes, utilize a função `predizer_paciente()`.

```python
predizer_paciente(dados_paciente, modelo='dt', alpha=0.1)
```

## Autores

Esse projeto foi desenvolvido como parte da disciplina de Inteligência Artificial na FATEC Mogi Mirim, primeiro semestre de 2026. Autores: Lucca Bellini Pena e Danielle Rachel Rios Moreira.

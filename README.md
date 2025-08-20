# Challenge Telecom X - parte 2

## Notebook Colab

El archivo a ejecutar es: `Challenge_Telemax_parte2.ipynb`

## 🎯 La Misión

Tu nueva misión es desarrollar modelos predictivos capaces de __prever qué clientes tienen mayor probabilidad de cancelar sus servicios__ (churn).

La empresa quiere anticiparse al problema de la cancelación, y te corresponde a ti construir un pipeline robusto para esta etapa inicial de modelado.

## Contenido del Notebook ipynb

1. Pasos Previos

2. 🎯 La Misión

3. Procesamiento del archivo csv para Challenge Parte 2

    - 3.1 Lectura el archivo
    - 3.2 🛠️ Preparación de los Datos
    - 3.3 Verificación de los valores nulos
    - 3.4 Normalización/Estandarización 
    - 3.5 Valida correlación entre las variables
    - 3.6 Análisis de Multicolinealidad
    - 3.7 Conclusión

4. Modelos Predictivos

    - 4.1 Preparación de los sets de datos (features y target)
    - 4.2 Preparación de los sets de datos (train y test)
    - 4.3 Normalización
    - 4.4 Balancear (inicial)
    - 4.5 Prueba de predicción
    - 4.6 Análisis de la Matriz de Confusión
    - 4.7 Balancear (avanzado)

5. Entrenando Modelos

    - 5.1 Calentando motores: Probando modelo RandomForest
    - 5.2 Primer acercamiento a Grid Search
    - 5.3 Evaluación y búsqueda del mejor modelo
    - 5.4 ¿Cómo se evalúan overfitting y underfitting?

6. Implementación, Evaluación y Obtención del mejor modelo

    - 6.1 Implementación de funciones de apoyo
    - 6.2 Implementación funciones para búsqueda del modelo óptimo
    - 6.3 Gráficos
    - 6.4 Conclusión

## Librerías utilizadas

```python
## Básicas
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Análisis de Multicolinealidad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

## Preparación
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

## Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import ParameterGrid

# Validación y métricas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from itertools import product
```

## Resultado final

Dado el análisis realizado y los resultados obtenidos, se concluye que el mejor modelo es:

- 🏆 Mejor modelo: RandomForest
- 🔹 Balanceo usado: SMOTETomek

Este modelo ha obtenido el más alto ICM, indicador que consideramos sería el que defina el mejor modelo implementado.

# 📊 Credit Scoring – Give Me Some Credit

Proyecto completo de **Credit Scoring** orientado a evaluar la probabilidad de incumplimiento de un cliente, utilizando análisis de datos, selección de variables y modelos de **Machine Learning**.

El objetivo es **simular un flujo real de trabajo** de un Data Analyst / ML Junior, desde el análisis exploratorio hasta la inferencia del modelo.

---

## 🧠 Objetivo del proyecto

Construir un modelo capaz de **predecir riesgo crediticio** a partir de variables financieras y demográficas, respondiendo a la pregunta:

> ¿Cuál es la probabilidad de que un cliente incurra en un default?

Este tipo de modelos es ampliamente utilizado en **banca, fintechs y scoring crediticio**.

---

## 🛠️ Tecnologías utilizadas

- Python  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  
- Jupyter Notebook  
- Git & GitHub  

---

## 📂 Estructura del proyecto

├── notebooks/
│ ├── EDA.ipynb # Análisis Exploratorio de Datos
│ ├── ETL.ipynb # Limpieza y transformación de datos
│ ├── Feature_Selection.ipynb # Selección de variables relevantes
│ ├── Modelo_ML.ipynb # Entrenamiento y evaluación de modelos
│ └── Inference_Test.ipynb # Inferencia sobre nuevos registros
├── app.py # Script de inferencia / demo
└── README.md

> ⚠️ **Nota:**  
> Los datasets y modelos entrenados no se incluyen en el repositorio para mantenerlo liviano.  
> El flujo completo puede reproducirse ejecutando los notebooks en orden.

---

## 📊 Modelos implementados

- Regresión Logística  
- Random Forest Classifier  

### Métrica principal
- **ROC-AUC**, elegida por ser adecuada para datasets desbalanceados típicos de problemas de crédito y fraude.

---

## 📈 Flujo de trabajo

1. **EDA**
   - Distribución de variables
   - Detección de outliers
   - Análisis de correlación

2. **ETL**
   - Limpieza de valores faltantes
   - Transformaciones
   - Preparación de features

3. **Feature Selection**
   - Selección de variables relevantes

4. **Modelado**
   - Entrenamiento
   - Evaluación con ROC-AUC
   - Comparación de modelos

5. **Inferencia**
   - Predicción sobre nuevos registros

---

## 🚀 Cómo ejecutar el proyecto

1. Clonar el repositorio  
2. (Opcional) Crear un entorno virtual  
3. Instalar dependencias:
   pip install pandas numpy matplotlib seaborn scikit-learn

## 👤 Autor

Pablo Lerner
Data Analyst Jr.
Interesado en análisis de datos, credit scoring y prevención de fraude.

# Proyecto Final Data Science II

**Comision:** 77790  
**Curso:** Data Science II  
**Estudiante:** David Alvarez

Este proyecto analiza un dataset de uso de vehiculos electricos en India. La idea principal es predecir si un viaje puede tener alto riesgo de ansiedad por autonomia, usando modelos de Machine Learning.

## Presentacion rapida

La ansiedad por autonomia aparece cuando un usuario de vehiculo electrico puede sentir que no tiene suficiente bateria o autonomia para completar un viaje. Detectar estos casos puede ayudar a mejorar la experiencia del usuario y planificar mejor la infraestructura de carga.

## Dataset

El dataset utilizado es `EV_Usage_Dataset_v2.csv` y esta disponible en GitHub:

```text
https://raw.githubusercontent.com/rdalvarez/CoderHouse-DataScienceII/refs/heads/main/EV_Usage_Dataset_v2.csv
```

Contiene informacion sobre viajes de vehiculos electricos, ciudades, tipo de vehiculo, bateria, distancia, tipo de carga, clima, trafico y riesgo de ansiedad por autonomia.

## Problema de Machine Learning

El problema se trabaja como una **clasificacion binaria**.

Variable objetivo:

```text
range_anxiety_risk
```

Valores posibles:

- `0`: bajo riesgo
- `1`: alto riesgo

## Contenido del proyecto

- Analisis exploratorio de datos
- Revision de valores faltantes
- Feature Engineering
- Feature Selection
- Preparacion de datos con pipelines
- Entrenamiento de modelos
- Validacion cruzada
- Optimizacion de hiperparametros con GridSearchCV
- Evaluacion final del mejor modelo
- App simple en Streamlit

## Modelos utilizados

Se entrenaron dos modelos principales:

- Regresion Logistica
- Random Forest

El mejor resultado se obtuvo con Random Forest.

## Resultados principales

Metricas finales del modelo optimizado:

| Metrica | Resultado |
|---|---:|
| Accuracy | 0.971 |
| Precision | 0.971 |
| Recall | 0.909 |
| F1-score | 0.939 |
| ROC-AUC | 0.990 |

Matriz de confusion:

```text
[[1590, 14],
 [  47, 472]]
```

## Archivos principales

| Archivo | Descripcion |
|---|---|
| `Proyecto_Final_DataSciece_David_Alvarez.ipynb` | Notebook principal del trabajo final |
| `EV_Usage_Dataset_v2.csv` | Dataset utilizado |
| `app.py` | Aplicacion simple en Streamlit |
| `requirements.txt` | Librerias necesarias para ejecutar el proyecto |

## Como ejecutar el notebook

1. Abrir el archivo `Proyecto_Final_DataSciece_David_Alvarez.ipynb` en Google Colab o Jupyter.
2. Ejecutar las celdas en orden.
3. El dataset se carga directamente desde GitHub, por lo que no hace falta subirlo manualmente.

## Como ejecutar la app Streamlit

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la app:

```bash
streamlit run app.py
```

## Nota

El dataset es sintetico, por lo que los resultados pueden ser mas altos que los que saldrian usando datos reales.

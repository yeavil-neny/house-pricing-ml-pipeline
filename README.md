# House Pricing ML Pipeline (Model Factory): house-pricing-ml-pipeline

Este repositorio contiene el código fuente, los datasets y el pipeline de experimentación para el entrenamiento y exportación de un modelo predictivo de regresión enfocado en estimar el precio de viviendas.

El propósito fundamental de este proyecto es actuar como la **Fábrica de Modelos (Model Factory)**, aislando la etapa de ciencia de datos y entrenamiento del pipeline de despliegue automático (CD).

## Características del Proyecto
* **Dataset Tabular:** Datos históricos de viviendas que incluyen características clave como metros cuadrados, número de habitaciones, baños y antigüedad.
* **Algoritmo de Regresión:** Entrenamiento basado en Scikit-Learn (`RandomForestRegressor`).
* **Interoperabilidad (ONNX):** Conversión automática del modelo entrenado al formato estándar e intercambiable **ONNX** (`.onnx`) para asegurar un despliegue ligero y eficiente.
* **Separación de Entornos:** Este repositorio genera los artefactos (modelo y datos de prueba) que posteriormente serán almacenados en la nube y consumidos de forma desacoplada por el pipeline de MLOps.

## Estructura del Repositorio
* `data/`: Contiene el dataset completo de entrenamiento (`housing_data.csv`) y la muestra para pruebas unitarias de la API (`test_data.csv`).
* `train.py`: Script principal en Python encargado de cargar los datos, entrenar el modelo, evaluar métricas de validación y exportar el resultado final a `model_house_pricing.onnx`.
* `requirements.txt`: Dependencias necesarias para reproducir el entorno de desarrollo local.

## Stack Tecnológico
* Python 3
* Pandas & NumPy
* Scikit-Learn
* ONNX Runtime & Sklob / Skl2onnx

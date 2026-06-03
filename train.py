import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType # <- Esta es la forma más limpia para skl2onnx

# 1. Cargar los datos
print("Cargando datos...")
df = pd.read_csv("data/housing_data.csv")

X = df[['metros_cuadrados', 'habitaciones', 'banos', 'antiguedad']]
y = df['precio']

# 2. Dividir en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo
print("Entrenando el modelo...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar métricas básicas en el laboratorio
score = model.score(X_test, y_test)
print(f"R² Score del modelo en validación: {score:.4f}")

# 4. Convertir el modelo a formato ONNX
print("Convirtiendo modelo a formato ONNX...")
# Es vital indicarle a ONNX el tipo de datos y la forma (shape) que va a recibir.
# [None, 4] significa que puede recibir N filas, y cada fila tiene exactamente 4 columnas/features.
initial_type = [('float_input', FloatTensorType([None, 4]))]
    
# Generamos la estructura ONNX
onnx_model = to_onnx(model, initial_types=initial_type)

# 5. Guardar el archivo .onnx
onnx_path = "model_house_pricing.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"¡Éxito! Modelo guardado como '{onnx_path}' listo para MLOps.")
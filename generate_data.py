import pandas as pd
import numpy as np
import os

# Asegurar que la carpeta 'data' exista
os.makedirs("data", exist_ok=True)

# Fijar una semilla para que los datos sean reproducibles
np.random.seed(42)

# 1. Definir la cantidad de registros (viviendas)
num_viviendas = 500

# 2. Generar características (Features) realistas de forma aleatoria
metros_cuadrados = np.random.randint(40, 200, size=num_viviendas)  # Entre 40 y 200 m²
habitaciones = np.random.randint(1, 5, size=num_viviendas)        # Entre 1 y 4 alcobas
banos = np.random.randint(1, 4, size=num_viviendas)               # Entre 1 y 3 baños
antiguedad = np.random.randint(0, 20, size=num_viviendas)         # Entre 0 y 20 años de construida

# 3. Crear una regla lógica para calcular el precio (el "Target")
# Añadimos algo de ruido aleatorio para simular la vida real
ruido = np.random.normal(0, 15_000_000, size=num_viviendas) 

# Fórmula base: cada m² vale 2.5 millones, cada cuarto 15 millones, cada baño 20 millones
# y por cada año de antigüedad el precio baja 3 millones. Base de 40 millones.
precio = (
    40_000_000 + 
    (metros_cuadrados * 2_500_000) + 
    (habitaciones * 15_000_000) + 
    (banos * 20_000_000) - 
    (antiguedad * 3_000_000) + 
    ruido
)
precio = np.round(precio, -5) # Redondear para que se vea como precio real en pesos

# 4. Construir el DataFrame completo
df_completo = pd.DataFrame({
    'metros_cuadrados': metros_cuadrados,
    'habitaciones': habitaciones,
    'banos': banos,
    'antiguedad': antiguedad,
    'precio': precio
})

# 5. Guardar el dataset de entrenamiento principal
housing_path = "data/housing_data.csv"
df_completo.to_csv(housing_path, index=False)
print(f"¡Éxito! Creado '{housing_path}' con {num_viviendas} registros.")

# 6. Generar el 'test_data.csv' para el pipeline de MLOps
# Tomamos una muestra pequeña (ej. 5 filas) de las últimas filas del set
df_muestra_test = df_completo.tail(5).copy()

# CRUCIAL PARA MLOPS: Eliminamos la columna 'precio' 
# porque la API en producción recibirá datos crudos y generará la predicción.
df_muestra_test = df_muestra_test.drop(columns=['precio'])

test_path = "data/test_data.csv"
df_muestra_test.to_csv(test_path, index=False)
print(f"¡Éxito! Creado '{test_path}' con {len(df_muestra_test)} filas (sin la columna de precio).")
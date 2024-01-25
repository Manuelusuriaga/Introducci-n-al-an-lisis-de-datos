import numpy as np
import pandas as pd
from datasets import load_dataset

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Obtener la lista de edades
edades = data["age"]

# Convertir la lista de edades a un arreglo de NumPy

edades_np = np.array(edades)

# Calcular el promedio de edad
promedio_edad = np.mean(edades_np)

# Imprimir el resultado
print("El promedio de edad de las personas participantes en el estudio es:", promedio_edad)

# Parte 2: Carga de datos

# Continuando con la anterior sección del proyecto integrador, ahora debes realizar lo siguiente:

# 1.Convertir la estructura Dataset en un DataFrame de Pandas usando pd.DataFrame.
df = pd.DataFrame(data)

# 2.Separar el dataframe en dos diferentes, uno conteniendo las filas con personas que perecieron (is_dead=1) y otro con el complemento.

df_perecieron = df[df['is_dead'] == 1]
df_no_perecieron = df[df['is_dead'] == 0]

# 3.Calcular los promedios de las edades de cada dataset e imprimir.
promedio_edades_perecieron = np.mean(df_perecieron['age'])
promedio_edades_no_perecieron = np.mean(df_no_perecieron['age'])

# Imprimir los resultados
print(f"Promedio de edad de personas que perecieron: {promedio_edades_perecieron:.2f} años")
print(f"Promedio de edad de personas que no perecieron: {promedio_edades_no_perecieron:.2f} años")

# Parte 3: Calculando analíticas simples

# 1.Verificar que los tipos de datos son correctos en cada colúmna (por ejemplo que no existan colúmnas numéricas en formato de cadena).
# Verificar tipos de datos actuales
print("Tipos de datos actuales:")
print(df.dtypes)

# 2.Calcular la cantidad de hombres fumadores vs mujeres fumadoras (usando agregaciones en Pandas).
# Calcular la cantidad de hombres fumadores vs mujeres fumadoras
conteo_fumadores_por_genero = df.groupby(['is_male', 'is_smoker']).size().unstack().fillna(0)

# Imprimir el resultado
print("\nCantidad de hombres fumadores vs mujeres fumadoras:")
print(conteo_fumadores_por_genero)
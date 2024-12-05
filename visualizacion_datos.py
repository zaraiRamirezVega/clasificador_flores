# data_visualization.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def guardar_imagen(nombre_archivo):
    """Guardar la imagen generada en la carpeta 'datos'."""
    if not os.path.exists('clasificador_flores/datos'):  # Verificar si la carpeta 'datos' existe
        os.makedirs('clasificador_flores/datos')  # Crear la carpeta si no existe
    plt.savefig(f'clasificador_flores/datos/{nombre_archivo}.png')  # Guardar la imagen en la carpeta 'datos'

def visualize_data():
    # Cargar el dataset Iris
    iris = load_iris()
    X = iris.data  # Características
    y = iris.target  # Etiquetas (especies)

    # Convertir el dataset en un DataFrame de pandas para facilitar la manipulación
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = iris.target_names[y]  # Agregar la especie de la flor como columna

    # 1. Ver las primeras filas del DataFrame
    print(df.head())

    # 2. Gráfico de dispersión (scatterplot) para ver la relación entre las características
    sns.pairplot(df, hue="species", palette="Set1")
    plt.suptitle("Pairplot de las características del Iris", y=1.02)
    guardar_imagen("pairplot_caracteristicas_iris")

    # 3. Mapa de calor para ver la correlación entre las características
    correlation_matrix = df.drop('species', axis=1).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Calor de la Correlación entre Características")
    guardar_imagen("mapa_calor_correlacion")

    # 4. Diagramas de caja (boxplot) para comparar las distribuciones de las características por especie
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Distribución de la Longitud del Sépalo por Especie')
    guardar_imagen("boxplot_longitud_sepalo")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='species', y='petal length (cm)', data=df)
    plt.title('Distribución de la Longitud del Pétalo por Especie')
    guardar_imagen("boxplot_longitud_petalo")

    # 5. Gráfico de barras para ver la media de cada característica por especie
    mean_df = df.groupby('species').mean()
    mean_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Promedio de Características por Especie')
    plt.ylabel('Valor Promedio')
    plt.xticks(rotation=45)
    guardar_imagen("grafico_barras_promedio_especies")

if __name__ == "__main__":
    visualize_data()

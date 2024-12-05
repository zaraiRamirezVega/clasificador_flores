# train_model.py
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_and_save_model():
    # Cargar el dataset Iris
    iris = load_iris()
    X = iris.data  # Características
    y = iris.target  # Etiquetas

    # Dividir el dataset en entrenamiento y prueba (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el clasificador
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(clf, 'Clasificador_flores/modelo_flor_iris.pkl')

    print("Modelo entrenado y guardado correctamente.")

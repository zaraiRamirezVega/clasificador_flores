# predict.py
import joblib
import numpy as np
from sklearn.datasets import load_iris

def load_model():
    # Cargar el modelo guardado
    clf = joblib.load('Clasificador_flores/modelo_flor_iris.pkl')
    return clf

def predict_flower(clf):
    iris = load_iris()

    # Pedir al usuario que ingrese las medidas de la flor
    ancho_sepalo = float(input("Por favor, ingresa el ancho del sépalo (cm): "))
    longitud_sepalo = float(input("Por favor, ingresa la longitud del sépalo (cm): "))
    ancho_petalo = float(input("Por favor, ingresa el ancho del pétalo (cm): "))
    longitud_petalo = float(input("Por favor, ingresa la longitud del pétalo (cm): "))
    
    # Organizar las medidas en un array (como se espera en el modelo)
    nuevas_medidas = np.array([[ancho_sepalo, longitud_sepalo, ancho_petalo, longitud_petalo]])
    
    # Realizar la predicción
    prediccion = clf.predict(nuevas_medidas)
    
    # Mostrar la predicción
    print(f"La flor es de la especie: {iris.target_names[prediccion][0]}")

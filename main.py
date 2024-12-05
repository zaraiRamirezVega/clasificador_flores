# main.py
import os
from train_model import train_and_save_model
from predict import load_model, predict_flower

def main():
    # Verificar si el modelo ya está guardado, si no, entrenarlo
    if not os.path.exists('Clasificador_flores/modelo_flor_iris.pkl'):
        print("Modelo no encontrado. Entrenando el modelo...")
        train_and_save_model()

    # Cargar el modelo
    clf = load_model()

    while True:
        # Preguntar al usuario si desea predecir una flor nueva
        respuesta = input("¿Quieres saber qué tipo de flor es una flor nueva? (si/no): ").strip().lower()
        if respuesta == 'si':
            predict_flower(clf)
        elif respuesta == 'no':
            print("¡Hasta luego!")
            break
        else:
            print("Por favor, responde con 'sí' o 'no'.")

if __name__ == "__main__":
    main()

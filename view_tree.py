import matplotlib.pyplot as plt
from sklearn import tree
import joblib
from sklearn.datasets import load_iris

def plot_decision_tree():
    # Cargar el modelo entrenado
    clf = joblib.load('Clasificador_flores/modelo_flor_iris.pkl')
    
    # Cargar el dataset Iris para obtener los nombres de las características y las clases
    iris = load_iris()

    # Traducir los nombres de las características y las clases a español
    feature_names_es = [
        'Longitud del sépalo', 'Ancho del sépalo', 
        'Longitud del pétalo', 'Ancho del pétalo'
    ]
    
    class_names_es = ['Setosa', 'Versicolor', 'Virginica']

    # Crear la figura y visualizar el árbol de decisión
    plt.figure(figsize=(12, 8))  # Ajusta el tamaño si es necesario
    tree.plot_tree(clf, filled=True, feature_names=feature_names_es, class_names=class_names_es, rounded=True)

    # Guardar el árbol como imagen antes de mostrarlo
    plt.savefig('Clasificador_flores/arbol_decision.png')
    print("El árbol de decisión se ha guardado como 'arbol_decision.png' en la carpeta 'Clasificador_flores'.")
    
    # Mostrar el árbol en pantalla después de guardar
    plt.show()

if __name__ == "__main__":
    plot_decision_tree()

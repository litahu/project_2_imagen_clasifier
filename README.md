# Clasificador de imagenes

## 🏆 Descripción del proyecto
Este proyecto capacita a un clasificador de imágenes para reconocer diferentes especies de flores utilizando pytorch.

>[!Nota]
> Puedes obtener el conjunto de datos que se empleo(son 102 categorías de flores) [Ver](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

El modelo se basa en una red VGG16 preformada, seguida de un clasificador con 5 capas de capa lineal y estructura de capa de activación Relu. 

## 🛠 Herramientas 
```
**Soportes:** GPU, Notebooks y terminal de Jupyter
```
```
**Python 3:** Os, Json, Numpy, Matplotlib, Pytorch, PIL, Argparse
```
## 🗂 Files 
#### Parte 1: Desarrollo de un clasificador de imágenes con aprendizaje profundo
- `Image_Clasifier_Project` Implemento un clasificador de imágenes con PyTorch

#### Parte 2: Creación de la aplicación de línea de comandos
- `train.py` Es un Script que se encarga de entrenar una nueva red en un conjunto de datos y guarda el modelo como punto de control
- `predict.py` Este archivo se encarga de utilizar una red entrenada para predecir la clase de una imagen de entrada
- `model.py` Crea un archivo para funciones y clases relacionadas con el modelo(define, carga y guarda el modelo) 
- `utils.py` Crea un archivo para funciones de utilidad(carga datos, preprocesa imágenes y contiene funciones auxiliares para entrenar y predecir)
- `cat_to_name.json` Archivo que contiene los nombres de las categorías de las flores

## 👩‍🔧 Resultados
- Use Pytorch y el modelo preentrenado vgg16 para construir un preclasificador de imágenes de flores
- El modelo entrenado se guarda en el archivo 'checkpoint.pth'
- El clasificador puede identificar 102 tipos de flores con una precisión del 83,5% (en conjuntos de datos de prueba).






## 💖 Agradecimiento
Este proyecto ha sido parte del programa Udacity AI Programming with Python Nanodegree, como parte de mi graduación de Udacity Intro to Machine Learning using PyTorch, donde he entrenado una red de imágenes para clasificar flores

Si gustas puedes apuntar tu smartphone a una flor y la aplicación del teléfono te dirá qué flor es. Sin embargo, el modelo puede ser reentrenado en cualquier conjunto de datos de su elección. Usted puede aprender a reconocer coches,...






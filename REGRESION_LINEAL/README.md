# Modelos de Regresión y Optimización de Parámetros

Este repositorio contiene tres enfoques diferentes para realizar predicciones sobre datos de ventas minoristas. Cada modelo emplea distintos métodos para entrenar y optimizar el rendimiento de las predicciones.

## Modelo 1: Regresión Lineal por Descenso por Gradiente
Archivo: **REGRESION_LINEAL.py**

Este modelo implementa una regresión lineal personalizada utilizando el método de descenso por gradiente. Se enfoca en minimizar el error residual entre las predicciones del modelo y los valores reales de las etiquetas. El descenso por gradiente ajusta los coeficientes de la función lineal de manera iterativa, buscando el mínimo de la función de costo.

### Características:
- Ajuste manual de los coeficientes.
- Control de la tasa de aprendizaje.
- Visualización del proceso iterativo de ajuste de los pesos del modelo.
- Adecuado para escenarios donde se requiere un control más granular sobre el proceso de optimización.

## Modelo 2: Regresión Lineal utilizando Framework Sklearn
Archivo: **REGRESION_LINEAL_FRAMEWORK.py**

Este segundo modelo utiliza la implementación de **scikit-learn** para entrenar una regresión lineal. En este caso, se delega el proceso de ajuste de los coeficientes al algoritmo preconstruido en el framework, lo que simplifica el proceso de desarrollo.

### Características:
- Fácil implementación utilizando la clase `LinearRegression` de Sklearn.
- Optimización rápida de los coeficientes.
- Menor control sobre los pasos internos de optimización, pero adecuado para prototipado rápido.

## Modelo 3: RandomForest utilizando Método de Optimización de Parámetros Bayesiano
Archivo: **RANDOM_FOREST_REGRESION_LINEAR_BAYESIAN.py**

Este modelo implementa un **RandomForestRegressor** en conjunto con la búsqueda de hiperparámetros mediante optimización bayesiana. El enfoque bayesiano permite realizar un ajuste más preciso de los parámetros del modelo, mejorando su rendimiento en comparación con una búsqueda tradicional de tipo Grid o Random Search.

### Características:
- Optimización de hiperparámetros mediante el método **BayesSearchCV**.
- Mayor precisión en los resultados debido a la búsqueda eficiente de parámetros.
- Ideal para escenarios donde el modelo Random Forest necesita ser ajustado con precisión para maximizar el rendimiento.

## Archivo Dataset
- **dataset1_split.pkl**: Contiene los conjuntos de datos preprocesados, incluyendo `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, y `y_test`. Este archivo se genera tras el procesamiento de los datos originales y es utilizado para entrenar y evaluar los modelos.

¡Con estos tres enfoques, puedes experimentar diferentes formas de entrenar modelos predictivos y comparar su rendimiento!


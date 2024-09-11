# Proceso ETL de Datos de Ventas Minoristas

Este proyecto se enfoca en el procesamiento y análisis de datos de ventas minoristas utilizando un flujo ETL (Extracción, Transformación y Carga). Los datos originales provienen de un archivo CSV llamado **warehouse_and_Retail_Sales.csv**, que contiene información relevante sobre ventas e ingresos.

## Archivos del Proyecto

1. **DATASET_SALES1.py**:  
   Este script ejecutable es responsable de llevar a cabo el proceso ETL. Toma los datos originales del archivo CSV y los transforma en conjuntos de datos entrenables, dividiéndolos en subconjuntos específicos para entrenamiento, validación y prueba. Al ejecutar este archivo, se genera un archivo de salida denominado **dataset1_split**.

2. **RETAIL_SALES1.ipynb**:  
   Un cuaderno de Jupyter donde se documentan y exploran visualmente los resultados del proceso ETL. Este archivo incluye el análisis y las visualizaciones generadas a partir de los conjuntos de datos transformados.

3. **warehouse_and_Retail_Sales.csv**:  
   Es el archivo CSV que contiene los datos originales de ventas minoristas. Este archivo sirve como fuente de entrada para el proceso ETL.

4. **dataset1_split**:  
   Tras ejecutar el archivo **DATASET_SALES1.py**, se genera este archivo, que contiene los siguientes subconjuntos de datos:
   - **x_train**: Características para el entrenamiento del modelo.
   - **y_train**: Etiquetas correspondientes al entrenamiento.
   - **x_val**: Características para la validación del modelo.
   - **y_val**: Etiquetas correspondientes a la validación.
   - **x_test**: Características para la prueba final del modelo.
   - **y_test**: Etiquetas para la evaluación del modelo en el conjunto de prueba.

## Descripción General

Este flujo ETL es clave para preparar los datos antes de entrenar los modelos de machine learning. Con **DATASET_SALES1.py**, transformamos los datos crudos en subconjuntos útiles para garantizar que los modelos se entrenen, validen y prueben de manera eficiente. Además, **RETAIL_SALES1.ipynb** permite una exploración visual de estos datos, ofreciendo una visión más clara del comportamiento de las ventas y los márgenes de beneficio neto.

¡Todo listo para entrenar y obtener predicciones con precisión!


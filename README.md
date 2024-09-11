# ❚█══MACHINE LEARNING══█❚
🥤- PROJECT - ML - BASE - TEC -🥤
# Predicción de Ventas Minoristas (RETAIL SALES)

**Autor:** Jorge Martínez López  
**Institución:** Tecnológico de Monterrey, Campus Querétaro  
**Email:** A01704518@tec.mx  
**Email Alterno:** jorgemartinez2555@hotmail.com  

## Resumen

El avance de tecnologías como la inteligencia artificial ha mejorado los sectores académico y comercial, aunque pocos entienden su funcionamiento. En México, los pequeños negocios enfrentan dificultades económicas, y muchos cierran. La implementación de modelos de machine learning, como la regresión lineal y logística, podría optimizar su rendimiento y mejorar su competitividad.

## Introducción

El surgimiento de tecnologías como la inteligencia artificial ha permitido optimizar y tomar decisiones en diversos sectores. Sin embargo, muchas pequeñas empresas no implementan estos avances, lo que contribuye a su quiebra. Este trabajo se centra en aplicar algoritmos supervisados como la regresión lineal y logística para mejorar las ventas minoristas.

## Problema Actual

El mercado minorista en México tiene un crecimiento proyectado del 33.3% entre 2024 y 2027. Sin embargo, muchas tiendas pequeñas están cerrando debido a la baja capacidad adquisitiva de sus clientes. El uso de modelos de machine learning podría ayudar a estas tiendas a enfrentar las adversidades.

## Modelos Utilizados

1. **Regresión Lineal:** Utiliza una variable dependiente (label) y una o más variables independientes (features) para predecir resultados. La ecuación general es:  
   `Y = X*m + B`


## Dataset

El dataset utilizado proviene de Montgomery County y contiene información sobre ventas y movimientos de productos. Incluye las siguientes características:

- YEAR (Año)
- MONTH (Mes)
- SUPPLIER (Proveedor)
- ITEM CODE (Código del producto)
- ITEM DESCRIPTION (Descripción del producto)
- ITEM TYPE (Tipo de producto: licores)
- RETAIL SALES (Ventas en dólares)
- RETAIL TRANSFERS (Transferencias en dólares)
- WAREHOUSE SALES (Ventas a licenciatarios)

El dataset tiene alrededor de 300,000 registros.

## Implementación del Modelo

El modelo elegido es la regresión lineal, ya que las variables son continuas. La estructura de un modelo de machine learning incluye las siguientes fases:

1. **ETL (Extract, Transform, Load):**
   - Conocimiento del dataset.
   - Limpieza de datos: imputación y eliminación de redundancias.
   - Definición de variables dependientes e independientes.
   - División en set de entrenamiento, validación y prueba (75%, 10%, 15%).

2. **Algoritmo de Machine Learning:**
   - Optimización de pesos e hiperparámetros.

3. **Validación y Evaluación:**
   - Uso de R², MSE y RMSE para evaluar el rendimiento del modelo.

## Resultados

1. **Modelo 1 (Regresión Lineal desde cero):**
   - MSE de 626 en entrenamiento, 670 en validación, y 564 en prueba.
   - Un poder predictivo del 32%, con indicios de underfitting.

2. **Modelo 2 (Framework con Scikit-learn):**
   - MSE de 36 y un R² de 0.95, mostrando un poder predictivo del 95%.  
   - Tiempo de entrenamiento: 0.07 segundos, con inicios que el modelo está fitting.

3. **Modelo 3 (RandomForest):**
   - MSE de 28 y un poder predictivo del 96%.
   - Tiempo de entrenamiento: 7.5 hrs, con inicios que el modelo está fititng.

## Conclusiones

La implementación de modelos de machine learning puede ayudar a las tiendas minoristas a mejorar su rendimiento. Sin embargo, es esencial contar con datasets bien estructurados para obtener predicciones precisas. Aunque el primer modelo fue útil para entender el proceso, los modelos con frameworks ocomo Scikit-learn ofrecieron mejores resultados en menor tiempo. El modelo RandomForest mostró el mejor rendimiento, con un MSE más bajo y mayor precisión predictiva.

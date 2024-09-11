import numpy as np  # Biblioteca para cálculos numéricos y manejo de arrays.
import matplotlib.pyplot as plt  # Biblioteca para generar gráficos y visualizaciones.
from sklearn.ensemble import RandomForestRegressor  # Modelo de bosque aleatorio para regresión.
from sklearn.metrics import mean_squared_error, r2_score  # Métricas para evaluar rendimiento de regresión.
from sklearn.preprocessing import StandardScaler  # Escalador estándar para normalizar características.
from skopt import BayesSearchCV  # Herramienta para la búsqueda bayesiana de hiperparámetros.
from skopt.space import Real, Integer  # Definir rangos de búsqueda de hiperparámetros (reales e enteros).
import joblib  # Para guardar y cargar modelos y datos.
import time  # Para medir el tiempo transcurrido.


# Cargar los datos del conjunto de entrenamiento, validación y prueba desde un archivo .pkl
X_train, y_train, X_val, y_val, X_test, y_test = joblib.load('K:\\😈😈😈\\AI\\CIENCIA_DATOS\\PRACTICA\\PROJECT\\DATASET\\DATASET\\RETAIL_SALES\\dataset.pkl')

# Convertir los datos a arrays de NumPy, y asegurar que las características X tengan dos dimensiones
X_train = np.array(X_train).reshape(-1, 1) if X_train.ndim == 1 else np.array(X_train)
y_train = np.array(y_train).ravel()  # Aplanar el array de etiquetas
X_val = np.array(X_val).reshape(-1, 1) if X_val.ndim == 1 else np.array(X_val)
y_val = np.array(y_val).ravel()  # Aplanar el array de etiquetas
X_test = np.array(X_test).reshape(-1, 1) if X_test.ndim == 1 else np.array(X_test)
y_test = np.array(y_test).ravel()  # Aplanar el array de etiquetas

# Normalizar las características para asegurar que todas estén en la misma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Definir el modelo de RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# Definir el espacio de búsqueda para la optimización bayesiana de hiperparámetros
param_space = {
    'n_estimators': Integer(100, 1000),  # Número de árboles en el bosque
    'max_depth': Integer(3, 20),  # Profundidad máxima del árbol
    'min_samples_split': Integer(2, 10),  # Mínimo número de muestras para dividir un nodo
    'min_samples_leaf': Integer(1, 5),  # Mínimo número de muestras en una hoja
    'max_features': Real(0.1, 1.0, prior='uniform')  # Máximo número de características consideradas para dividir
}

# Configurar la búsqueda bayesiana para optimizar los hiperparámetros
opt = BayesSearchCV(
    model,
    param_space,
    n_iter=32,  # Número de iteraciones de la optimización bayesiana
    cv=3,  # Validación cruzada de 3 pliegues
    n_jobs=-1,  # Usar todos los núcleos de CPU para acelerar
    verbose=0,  # Nivel de verbosidad
    random_state=42
)

# Entrenar el modelo utilizando la optimización bayesiana para encontrar los mejores hiperparámetros
start_time = time.time()
opt.fit(X_train, y_train)

# Imprimir los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados: ", opt.best_params_)

# Extraer el mejor modelo entrenado con los mejores hiperparámetros
best_model = opt.best_estimator_
end_time = time.time()
training_time = end_time - start_time  # Calcular el tiempo de entrenamiento

# Realizar predicciones en el conjunto de entrenamiento, validación y prueba
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Calcular métricas de rendimiento (MSE y R^2) para cada conjunto de datos
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Mostrar los resultados del MSE, R^2 y tiempo de entrenamiento
print('MODELO 3')
print('----MSE----')
print("Train MSE:", train_mse)
print("Validation MSE:", val_mse)
print("Test MSE:", test_mse)
print('----R^2----')
print("Train R^2:", train_r2)
print("Validation R^2:", val_r2)
print("Test R^2:", test_r2)
print('----Time----')
print(f"Seg: {training_time:.2f}")
print('----Sesgo o Bias----')
print("Train:", np.mean(y_train_pred - y_train))
print("Validation:", np.mean(y_val_pred - y_val))
print("Test:", np.mean(y_test_pred - y_test))

# Visualización de las métricas de MSE para el conjunto de entrenamiento, validación y prueba
plt.tight_layout()
metrics = ['MSE (Train)', 'MSE (Validation)', 'MSE (Test)']
values = [train_mse, val_mse, test_mse]
colors = ['orange', 'purple', 'cyan']
plt.bar(metrics, values, color=colors)
plt.axhline(y=train_mse, color=colors[0], linestyle='--')
plt.axhline(y=val_mse, color=colors[1], linestyle='--')
plt.axhline(y=test_mse, color=colors[2], linestyle='--')
plt.ylabel('Valor')
plt.title('Métricas de MSE')
plt.show()

# Visualización de las métricas de R^2 para el conjunto de entrenamiento, validación y prueba
plt.tight_layout()
plt.figure(figsize=(8, 5))
metrics2 = [ 'R^2 (Train)', 'R^2 (Validation)', 'R^2 (Test)']
values2 = [ train_r2, val_r2, test_r2]
plt.bar(metrics2, values2, color=colors)
plt.axhline(y=train_r2, color=colors[0], linestyle='--')
plt.axhline(y=val_r2, color=colors[1], linestyle='--')
plt.axhline(y=test_r2, color=colors[2], linestyle='--')
plt.ylabel('Valor')
plt.title('Métricas de  R^2')
plt.show()

# Comparación de valores reales y predicciones para el conjunto de entrenamiento
plt.figure(figsize=(10, 6)) 
plt.plot(y_train, label='Valores Reales (y_train)', color=colors[0], linestyle='-', marker='o', markersize=4)  
plt.plot(y_train_pred, label='Predicciones (y_train_pred)', color=colors[1], linestyle='--', marker='x', markersize=4) 
plt.title('Comparación de Valores Reales vs Predicciones TRAIN', fontsize=16, fontweight='bold')
plt.xlabel('Índice de Datos', fontsize=12)
plt.ylabel('Ventas al por Menor (Retail Sales)', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Comparación de valores reales y predicciones para el conjunto de validación
plt.figure(figsize=(10, 6)) 
plt.plot(y_val, label='Valores Reales (y_val)', color=colors[0], linestyle='-', marker='o', markersize=4)  
plt.plot(y_val_pred, label='Predicciones (y_val_pred)', color=colors[1], linestyle='--', marker='x', markersize=4) 
plt.title('Comparación de Valores Reales vs Predicciones VALIDATION', fontsize=16, fontweight='bold')
plt.xlabel('Índice de Datos', fontsize=12)
plt.ylabel('Ventas al por Menor (Retail Sales)', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Comparación de valores reales y predicciones para el conjunto de prueba
plt.figure(figsize=(10, 6)) 
plt.plot(y_test, label='Valores Reales (y_test)', color=colors[0], linestyle='-', marker='o', markersize=4)  
plt.plot(y_test_pred, label='Predicciones (y_test_pred)', color=colors[1], linestyle='--', marker='x', markersize=4) 
plt.title('Comparación de Valores Reales vs Predicciones TEST', fontsize=16, fontweight='bold')
plt.xlabel('Índice de Datos', fontsize=12)
plt.ylabel('Ventas al por Menor (Retail Sales)', fontsize=12)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

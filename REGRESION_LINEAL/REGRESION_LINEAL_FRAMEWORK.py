import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar datos
X_train, y_train, X_val, y_val, X_test, y_test = joblib.load('K:\\😈😈😈\\AI\\CIENCIA_DATOS\\PRACTICA\\PROJECT\\DATASET\\DATASET\\RETAIL_SALES\\dataset.pkl')

# Asegúrate de que los datos sean arrays de NumPy
X_train = np.array(X_train)

y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalizar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calcular MSE y R^2
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Mostrar resultados
print("Train MSE:", train_mse)
print("Validation MSE:", val_mse)
print("Test MSE:", test_mse)

print("Train R^2:", train_r2)
print("Validation R^2:", val_r2)
print("Test R^2:", test_r2)

plt.figure(figsize=(10, 6)) 
plt.plot(y_val, label='Valores Reales (y_val)', color='blue', linestyle='-', marker='o', markersize=4)  
plt.plot(y_val_pred, label='Predicciones (y_val_pred)', color='orange', linestyle='--', marker='x', markersize=4) 

plt.title('Comparación de Valores Reales vs Predicciones', fontsize=16, fontweight='bold')
plt.xlabel('Índice de Datos', fontsize=12)
plt.ylabel('Ventas al por Menor (Retail Sales)', fontsize=12)

plt.legend(loc='best', fontsize=12)  # 'best' ajusta la leyenda en la mejor posición


plt.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout()
plt.show()
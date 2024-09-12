# Importar librerías necesarias
import numpy as np  # Para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Para la visualización de gráficos
import joblib  # Para cargar y guardar modelos y datos
import time  # Para medir el tiempo de entrenamiento

class LinearRegressionModel:
    # Inicialización de la clase LinearRegressionModel
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        # Configuración de los parámetros del modelo
        self.learning_rate = learning_rate  # Tasa de aprendizaje para el descenso de gradiente
        self.max_epochs = max_epochs  # Número máximo de épocas para el entrenamiento
        self.params = None  # Parámetros del modelo (pesos y bias)
        self.train_errors = []  # Lista para almacenar el error en el conjunto de entrenamiento
        self.val_error = 0  # Error en el conjunto de validación
        self.test_error = 0  # Error en el conjunto de prueba
        self.train_r2 = 0  # Coeficiente R^2 en entrenamiento
        self.val_r2 = 0  # Coeficiente R^2 en validación
        self.test_r2 = 0  # Coeficiente R^2 en prueba
        self.training_time = 0  # Tiempo de entrenamiento

    # Función para calcular la hipótesis (h(x) = a + bx1 + cx2 + ...)
    def h(self, sample):
        """Evaluates the hypothesis h(x) = a + bx1 + cx2 + ... for a given sample."""
        return np.dot(self.params, sample)

    # Función para calcular el error cuadrático medio (MSE)
    def compute_error(self, samples, y):
        """Calculates the mean squared error."""
        total_error = 0
        for i in range(len(samples)):
            error = self.h(samples[i]) - y[i]  # Error entre la predicción y el valor real
            total_error += error ** 2  # Sumar el cuadrado del error
        mean_error = total_error / len(samples)  # Promediar los errores
        return mean_error

    # Función para calcular el coeficiente de determinación R^2
    def compute_r2(self, samples, y):
        """Calcula el coeficiente de determinación R^2."""
        y_mean = np.mean(y)  # Calcular la media de los valores reales
        ss_total = sum((y_i - y_mean) ** 2 for y_i in y)  # Suma total de cuadrados
        ss_residual = sum((y_i - self.h(samples[i])) ** 2 for i, y_i in enumerate(y))  # Suma de los residuos
        return 1 - (ss_residual / ss_total)  # Fórmula de R^2

    # Función para realizar el descenso de gradiente
    def gradient_descent(self, samples, y):
        # Crear una copia temporal de los parámetros para actualizar
        temp_params = np.copy(self.params)
        for j in range(len(self.params)):  # Para cada parámetro
            gradient = 0
            for i in range(len(samples)):  # Calcular el gradiente
                error = self.h(samples[i]) - y[i]  # Error actual
                gradient += error * samples[i][j]  # Acumular el gradiente
            temp_params[j] -= self.learning_rate * gradient / len(samples)  # Actualizar los parámetros
        self.params = temp_params  # Asignar los nuevos parámetros

    # Función para escalar las muestras (normalización)
    def scale_samples(self, samples):
        # Convertir las muestras en un array transpuesto (features por columnas)
        samples = np.array(samples).T
        # Normalizar cada columna (excepto el bias)
        for i in range(1, len(samples)):
            avg = np.mean(samples[i])  # Calcular la media
            max_val = np.max(samples[i])  # Calcular el valor máximo
            samples[i] = (samples[i] - avg) / max_val  # Normalizar los valores
        return np.array(samples).T.tolist()

    # Función para hacer predicciones usando los parámetros entrenados
    def predict(self, X):
        samples = X.tolist()  # Convertir los datos a lista
        # Agregar el valor de bias (1) a cada muestra
        for i in range(len(samples)):
            samples[i] = [1] + samples[i]
        samples = self.scale_samples(samples)  # Escalar las muestras
        predictions = np.dot(samples, self.params)  # Calcular las predicciones
        return predictions

    # Función para entrenar el modelo usando los conjuntos de entrenamiento, validación y prueba
    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.params = np.zeros(len(X_train[0]) + 1)  # Inicializar los parámetros a cero (incluyendo bias)
        y = y_train.tolist()  # Convertir a lista
        samples = X_train.tolist()
        # Agregar el bias (1) a cada muestra
        for i in range(len(samples)):
            samples[i] = [1] + samples[i]
        samples = self.scale_samples(samples)  # Escalar las muestras de entrenamiento

        # Preparar muestras de validación y prueba
        val_samples = X_val.tolist()
        for i in range(len(val_samples)):
            val_samples[i] = [1] + val_samples[i]
        val_samples = self.scale_samples(val_samples)

        test_samples = X_test.tolist()
        for i in range(len(test_samples)):
            test_samples[i] = [1] + test_samples[i]
        test_samples = self.scale_samples(test_samples)

        start_time = time.time()  # Iniciar el cronómetro para medir el tiempo de entrenamiento
        epochs = 0
        # Bucle de entrenamiento (descenso de gradiente)
        while True:
            old_params = np.copy(self.params)  # Guardar una copia de los parámetros anteriores
            self.gradient_descent(samples, y)  # Actualizar los parámetros con descenso de gradiente
            train_error = self.compute_error(samples, y)  # Calcular el error de entrenamiento
            self.train_errors.append(train_error)  # Guardar el error en la lista
            epochs += 1  # Incrementar el número de épocas
            if np.array_equal(old_params, self.params) or epochs >= self.max_epochs:  # Condiciones de parada
                break
        end_time = time.time()  # Detener el cronómetro
        self.training_time = end_time - start_time  # Calcular el tiempo total de entrenamiento

        # Calcular errores y R^2 para validación y prueba
        self.val_error = self.compute_error(val_samples, y_val.tolist())
        self.test_error = self.compute_error(test_samples, y_test.tolist())
        self.train_r2 = self.compute_r2(samples, y)
        self.val_r2 = self.compute_r2(val_samples, y_val.tolist())
        self.test_r2 = self.compute_r2(test_samples, y_test.tolist())

        self.plot_errors()  # Graficar los errores

    # Función para visualizar los resultados (predicciones vs valores reales)
    def plot_results(self, X_train, y_train, X_val, y_val, X_test, y_test):
        lista_real = (y_train, y_val, y_test)
        y_train_p = self.predict(X_train)
        y_val_p = self.predict(X_val)
        y_test_p = self.predict(X_test)

        # Mostrar resultados en consola
        print('MODELO 1')
        print('----MSE----')
        print("Train MSE:", self.train_errors[-1])
        print("Validation MSE:", self.val_error)
        print("Test MSE:", self.test_error)
        print('----R^2----')
        print("Train R^2:", self.train_r2)
        print("Validation R^2:", self.val_r2)
        print("Test R^2:", self.test_r2)
        print('----Time----')
        print(f"Seg: {self.training_time:.2f}")
        print('----Sesgo o Bias----')
        print("Train:", np.mean(y_train_p - y_train))
        print("Validation:", np.mean(y_val_p - y_val))
        print("Test:", np.mean(y_test_p - y_test))

        # Graficar las predicciones vs los valores reales
        lista_pred = (y_train_p, y_val_p, y_test_p)
        lista_tittle = ('TRAIN', 'VALIDATION', 'TEST')
        for i in range(len(lista_real)):
            plt.figure(figsize=(10, 6))
            plt.plot(lista_real[i], label='Valores Reales', color='blue', linestyle='-', marker='o', markersize=4)
            plt.plot(lista_pred[i], label='Predicciones', color='orange', linestyle='--', marker='x', markersize=4)
            plt.title(f'Comparación de Valores Reales vs Predicciones {lista_tittle[i]}', fontsize=16, fontweight='bold')
            plt.xlabel('Índice de Datos', fontsize=12)
            plt.ylabel('Ventas al por Menor (Retail Sales)', fontsize=12)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    # Función para graficar los errores (MSE) y coeficientes R^2
    def plot_errors(self):
        plt.tight_layout()
        plt.figure(figsize=(8, 5))
        metrics = ['MSE (Train)', 'MSE (Validation)', 'MSE (Test)']
        values = [self.train_errors[-1], self.val_error, self.test_error]
        colors = ['blue', 'green', 'orange']
        plt.bar(metrics, values, color=colors)
        plt.axhline(y=self.train_errors[-1], color='b', linestyle='--')
        plt.axhline(y=self.val_error, color='g', linestyle='--')
        plt.axhline(y=self.test_error, color='r', linestyle='--')
        plt.ylabel('Valor')
        plt.title('Métricas de MSE ')
        plt.show()

        plt.tight_layout()
        plt.figure(figsize=(8, 5))
        metrics2 = ['R^2 (Train)', 'R^2 (Validation)', 'R^2 (Test)']
        values2 = [self.train_r2, self.val_r2, self.test_r2]
        colors2 = ['blue', 'green', 'orange']
        plt.bar(metrics2, values2, color=colors2)
        plt.axhline(y=self.train_r2, color='b', linestyle='--')
        plt.axhline(y=self.val_r2, color='g', linestyle='--')
        plt.axhline(y=self.test_r2, color='r', linestyle='--')
        plt.ylabel('Valor')
        plt.title('Métricas de  R^2')
        plt.show()

# Cargar datos y entrenar el modelo
X_train, y_train, X_val, y_val, X_test, y_test = joblib.load('dataset1_split.pkl')

# Instanciar el modelo con la tasa de aprendizaje y el número de épocas deseado
model = LinearRegressionModel(learning_rate=0.5, max_epochs=1000)

# Entrenar el modelo y visualizar los resultados
model.fit(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy())
model.plot_results(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy())


import numpy as np
import matplotlib.pyplot as plt
import joblib
import time  # Importar la librería time

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.params = None
        self.train_errors = []
        self.val_error = 0
        self.test_error = 0
        self.train_r2 = 0
        self.val_r2 = 0
        self.test_r2 = 0

    def h(self, sample):
        """Evaluates the hypothesis h(x) = a + bx1 + cx2 + ... for a given sample."""
        return np.dot(self.params, sample)

    def compute_error(self, samples, y):
        """Calculates the mean squared error."""
        total_error = 0
        for i in range(len(samples)):
            error = self.h(samples[i]) - y[i]
            total_error += error ** 2
        mean_error = total_error / len(samples)
        return mean_error

    def compute_r2(self, samples, y):
        """Calcula el coeficiente de determinación R^2."""
        y_mean = np.mean(y)
        ss_total = sum((y_i - y_mean) ** 2 for y_i in y)
        ss_residual = sum((y_i - self.h(samples[i])) ** 2 for i, y_i in enumerate(y))
        return 1 - (ss_residual / ss_total)

    def gradient_descent(self, samples, y):
        temp_params = np.copy(self.params)
        for j in range(len(self.params)):
            gradient = 0
            for i in range(len(samples)):
                error = self.h(samples[i]) - y[i]
                gradient += error * samples[i][j]
            temp_params[j] -= self.learning_rate * gradient / len(samples)
        self.params = temp_params

    def scale_samples(self, samples):
        samples = np.array(samples).T
        for i in range(1, len(samples)):
            avg = np.mean(samples[i])
            max_val = np.max(samples[i])
            samples[i] = (samples[i] - avg) / max_val
        return np.array(samples).T.tolist()

    def predict(self, X):
        samples = X.tolist()
        for i in range(len(samples)):
            samples[i] = [1] + samples[i]
        samples = self.scale_samples(samples)
        predictions = np.dot(samples, self.params)
        return predictions

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.params = np.zeros(len(X_train[0]) + 1)
        y = y_train.tolist()
        samples = X_train.tolist()
        for i in range(len(samples)):
            samples[i] = [1] + samples[i]
        samples = self.scale_samples(samples)
        val_samples = X_val.tolist()
        for i in range(len(val_samples)):
            val_samples[i] = [1] + val_samples[i]
        val_samples = self.scale_samples(val_samples)
        test_samples = X_test.tolist()
        for i in range(len(test_samples)):
            test_samples[i] = [1] + test_samples[i]
        test_samples = self.scale_samples(test_samples)

        start_time = time.time()  # Iniciar el cronómetro
        epochs = 0
        while True:
            old_params = np.copy(self.params)
            self.gradient_descent(samples, y)
            train_error = self.compute_error(samples, y)
            self.train_errors.append(train_error)
            epochs += 1
            if np.array_equal(old_params, self.params) or epochs >= self.max_epochs:
                break
        end_time = time.time()  # Detener el cronómetro
        training_time = end_time - start_time

        self.val_error = self.compute_error(val_samples, y_val.tolist())
        self.test_error = self.compute_error(test_samples, y_test.tolist())

        self.train_r2 = self.compute_r2(samples, y)
        self.val_r2 = self.compute_r2(val_samples, y_val.tolist())
        self.test_r2 = self.compute_r2(test_samples, y_test.tolist())

        # Mostrar resultados
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
        print(f"Seg: {training_time:.2f}")

        self.plot_errors()

    def plot_results(self, X_train, y_train, X_val, y_val, X_test, y_test):
        lista_real = (y_train, y_val, y_test)
        y_train_p = self.predict(X_train)
        y_val_p = self.predict(X_val)
        y_test_p = self.predict(X_test)
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
        metrics2 = [ 'R^2 (Train)', 'R^2 (Validation)', 'R^2 (Test)']
        values2 = [ self.train_r2, self.val_r2, self.test_r2]
        colors2 = [ 'blue', 'green', 'orange']
        plt.bar(metrics2, values2, color=colors2)
        plt.axhline(y=self.train_r2, color='b', linestyle='--')
        plt.axhline(y=self.val_r2, color='g', linestyle='--')
        plt.axhline(y=self.test_r2, color='r', linestyle='--')
        plt.ylabel('Valor')
        plt.title('Métricas de  R^2')
        plt.show()

# Cargar datos y entrenar el modelo
X_train, y_train, X_val, y_val, X_test, y_test = joblib.load('K:\😈😈😈\AI\CIENCIA_DATOS\PRACTICA\PROJECT\DATASET\DATASET\RETAIL_SALES\dataset.pkl')

model = LinearRegressionModel(learning_rate=0.5, max_epochs=1000)
model.fit(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy())
model.plot_results(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), X_test.to_numpy(), y_test.to_numpy())


	

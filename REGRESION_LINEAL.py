
#----LIBRERIA----
import numpy as np 
import matplotlib.pyplot as plt 


#----CLASS----
class RegresionLineal:

	#----INICIALIZACIÓN----
	def __init__(self,Theta,Label,Samples,LR,Bias,Label_test,Samples_test):
		
		#----VARIABLES CLASS----
		self.Theta = np.array(Theta).astype(float)
		self.Label = np.array(Label).astype(float)
		self.Samples = np.array(Samples).astype(float)
		self.Label_test = np.array(Label_test).astype(float)
		self.Samples_test = np.array(Samples_test).astype(float)
		self.LR = LR
		self.epochs = 0
		self.Bias = Bias
		self.error_train = []
		self.error_test = []
		# Insertar BIAS en Theta y Samples
		self.Theta = np.append(self.Theta, 0)
		self.Samples = np.insert(self.Samples, 0, self.Bias, axis=1)
		self.Samples_test = np.insert(self.Samples_test, 0, self.Bias, axis=1)



	#----THETA ÓPTIMAS----
	def Betas_Opt(self,samples,label,tittle):
		# Convertir a matrices NumPy
		X = np.array(samples)
		y = np.array(label)

    	# Cálculo de theta usando la fórmula de la regresión lineal múltiple
		Theta_ = np.linalg.pinv(X.T @ X) @ X.T @ y
		print(tittle)
		for i in range(len(Theta_)):
			print(f'Tetha opt {i} :{Theta_[i]}')


	#----MUESTRA ERROR MSE DE TRAIN Y TEST----
	def RESULTS(self,tittle):
		print(tittle)
		for i in range(len(self.Theta)):
			print(f'Tetha org  {i} :{self.Theta[i]}')
		print(f'''
ERROR TRAIN {self.error_train[-1]}
ERROR TEST {self.error_test[-1]}
''')
	
	#----MUESTRA GRÁFICAMENTE EL ERROR MSE DE TRAIN Y TEST----
	def plot_error(self):
		# Gráfico de los errores de entrenamiento y prueba
		plt.figure(figsize=(10, 6))
		plt.plot(self.error_train, label='Error en entrenamiento')
		plt.plot(self.error_test, label='Error en prueba')
		plt.xlabel('Épocas')
		plt.ylabel('Error MSE')
		plt.title('Comparación del error en entrenamiento y prueba durante las épocas')
		plt.legend()
		plt.show()
 

	#----HIPÓTESIS----
	def hypothesis(self, theta, samples):
		return np.dot(theta, samples)

	#----ERROR MSE DE LA HIPÓTESIS----
	def ERROR(self, Theta, Samples, Label,show,tittle):
		# Mean Squared Error (MSE)
		total_error = 0  # Inicializamos el error acumulado
    
    	# Iteramos sobre cada muestra y su etiqueta correspondiente
		for i in range(len(Samples)):
			sample = Samples[i]
			actual_label = Label[i]
        
        	# Calculamos la hipótesis para la muestra actual
			predicted_label = self.hypothesis(Theta, sample)
			if show:
				print( f"{tittle} hyp  {predicted_label}  y {Label[i]}")   
        	# Calculamos el error absoluto (diferencia entre la predicción y la etiqueta real)
			error = (predicted_label - actual_label)**2
        	# Acumulamos el error
			total_error += error
    
    	# Calculamos el MAE dividiendo el error acumulado entre el número de muestras
		mean_absolute_error = total_error / len(Samples)
		return mean_absolute_error
	
	
	#----NORMALIZACIÓN LABELS----
	def Scaling_labels(self, labels):
		# ESTANDARIZACIÓN DESVIACIÓN ESTANDAR
		
		medias = labels.mean(axis=0)
		desviaciones = labels.std(axis=0)
		labels = (labels- medias) / desviaciones
		
		# NORMALIZACIÓN (0-1) MIN-MAX
		label_min = np.min(labels)
		label_max = np.max(labels)
		return (labels - label_min) / (label_max - label_min)


	#----NORMALIZACIÓN SAMPLES----
	def Scaling(self, samples):

		# ESTANDARIZACIÓN DESVIACIÓN ESTANDAR
		# Escalado estándar (0 media, 1 desviación estándar)
		medias = samples[:, 1:].mean(axis=0)
		desviaciones = samples[:, 1:].std(axis=0)
		samples[:, 1:] = (samples[:, 1:] - medias) / desviaciones
		

		# NORMALIZACIÓN (0-1) MIN-MAX
		mins = samples[:, 1:].min(axis=0)
		maxs = samples[:, 1:].max(axis=0)
		dif = maxs - mins
		dif[dif == 0] = 1  # Evitar división por cero
		samples[:, 1:] = (samples[:, 1:] - mins) / dif
		return samples

		
	
	
	#----DESCENSO POR GRADIENTE----
	def GD(self, theta, samples, label, lr):

		m = len(samples)  # Número de muestras
		n = len(theta)    # Número de parámetros

    	# Crear una copia de theta para actualizar los valores gradualmente
		temp = np.copy(theta)

    	# Iterar sobre cada parámetro theta[j]
		for j in range(n):
			acum = 0  # Acumulador del gradiente

        	# Calcular la suma de los gradientes para todas las muestras
			for i in range(m):
				error = self.hypothesis(theta, samples[i]) - label[i]
				acum += error * samples[i][j]

        	# Actualizar el parámetro theta[j]
			temp[j] = theta[j] - (lr * (1 / m) * acum)

   		 # Devolver los nuevos parámetros actualizados
		return temp
	

	#----MODELO DE REGRESIÓN LINEAL----
	def LivingML(self, limit,M):
		# MUESTRA LAS THETA ÓPTIMAS
		self.Betas_Opt(self.Samples,self.Label,'Train Betas Opt') # Muestra theta óptimizadas, el módelo debería alcanzar estos parámetro

		# ESCALAMIENTO DE LABELS Y SAMPLES
		self.Samples = self.Scaling(self.Samples)
		self.Label = self.Scaling_labels(self.Label)
		# ESCALAMIENTO DE LABELS Y SAMPLES
		#self.Samples_test = self.Scaling(self.Samples_test)
		# self.Label_test = self.Scaling_labels(self.Label_test)
		
		# CÁLCULA Y ACTUALIZACIÓN DE THETA 
		while self.epochs < limit:
			old =  np.copy(self.Theta)
			# Actualización de parámetros
			self.Theta = self.GD(self.Theta, self.Samples, self.Label, self.LR)
			
			# Cálculo de errores TRAIN / TEST 
			train_error = self.ERROR(self.Theta, self.Samples, self.Label,M,'TRAIN')
			test_error = self.ERROR(self.Theta, self.Samples_test, self.Label_test,M,'TEST')
			
			# AÑADIR A LA LISTA
			self.error_train.append(train_error)
			self.error_test.append(test_error)
			# SI NO HAY CAMBIOS
			if np.allclose(self.Theta, old):
				break
			# ÉPOCAS
			self.epochs += 1
				
	
			

#----CORRER EL PROGRAMA----
if __name__ == "__main__":

	#----DEFINICIÓN DE VARIABLES----
	Theta= [0 , 0]  # Inicialización de theta
	Bias = 1# Bias
	Label = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 20]  # Etiquetas de entrenamiento
	Samples = [[100, 100], [200, 200], [300, 300], [400, 400], [500, 500], [600, 600],
	           [700, 700], [900, 900], [1000, 1000], [1100, 1100], [1200, 1200],
	           [1400, 1400], [1500, 1500], [2000, 2000]]  # Muestras de entrenamiento
	Label_test = [8, 21]  # Etiquetas de prueba
	Samples_test = [[800, 800], [2100, 2100]]  # Muestras de prueba
	LR = 0.01  # Tasa de aprendizaje



	
	#----CREACIÓN DE MODELO REGRESIÓN LINEAL----
	regresion = RegresionLineal(Theta, Label, Samples, LR, Bias, Label_test, Samples_test)
	regresion.LivingML(10000,False)  # Entrenar por 100 épocas LivingML(Epoch, Mostrar hyp vs label)
	regresion.RESULTS('Train Theta Model') # Muestra resultados 
	regresion.plot_error()  # Graficar los errores
	

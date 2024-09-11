
#----LIBRERIAS----
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
if __name__ == "__main__":
    df = pd.read_csv('K:\😈😈😈\AI\CIENCIA_DATOS\PRACTICA\PROJECT\DATASET\DATASET\RETAIL_SALES\Warehouse_and_Retail_Sales.csv')

    print('----INFORMACION DE TIPO DE DATOS----')
    print(df.info())
    print('----DESCRIPCIÓN----')
    print(df.describe())
    print('----ELEMENTOS QUE NO TIENEN VALOR EN LAS COLUMNAS----')
    print(df.isna().sum())

    df=df.iloc[np.random.permutation(df.index)].reset_index(drop=True)

    print('----EVALUAR ITEM CODE----')
    print('ITEM CODE',df['ITEM CODE'].mode())
    df=df.drop('ITEM CODE', axis=1)
    print('----EVALUAR ITEM DESCRIPTION----')
    R=df['ITEM DESCRIPTION'].mode() # CONOCER LA MODA
    G=df['ITEM DESCRIPTION'].nunique() # CONOCER CUANTOS ELEMENTOS ÚNICOS HAY EN LA COLUMNA
    print(f''' 
        Moda {R}
        Total de elementos únicos {G} vs total {df['ITEM DESCRIPTION'].count()} 
    ''')
    d =df[df['ITEM TYPE'].isna()].index
    print(df.isna().sum())
    df = df.drop(d)
    print(df.isna().sum())
    print('----EVALUAR SUPPLIER----')
    d =df[df['SUPPLIER'].isna()].index
    for i in d.values:
        row = df.loc[i]
        # Contar el número de elementos vacíos en esa fila
        num_empty_elements = row.isna().sum()
        if num_empty_elements >1:
            print(f'Número de elementos vacíos en la fila {i}: {num_empty_elements}')  # Salida: 2
    print(df['SUPPLIER'].isna().sum())

    #ELIMINAR TODOS AQUELLOS QUE TIENEN DOS O MÁS NAN POR RENGLON
    for i in d.values:
        row = df.loc[i]
        # Contar el número de elementos vacíos en esa fila
        num_empty_elements = row.isna().sum()
        if num_empty_elements >1:
            print(f'Número de elementos vacíos en la fila {i}: {num_empty_elements}')  # Salida: 2
            df = df.drop([i])

    print(df.isna().sum())
    print('----ELMINAR ELEMENTOS CON NAN----')
    df=df.dropna(subset=['SUPPLIER'])
    print(df.isna().sum())
    print('----CODIFICACIÓN----')
    q=[]
    # Obtiene los valor que no se repitan 
    for i in df['ITEM DESCRIPTION'].values: 
        if i not in q:
            q.append(i)
    item_dict = {item: index + 1 for index, item in enumerate( q)}
    df['ITEM DESCRIPTION'] = df['ITEM DESCRIPTION'].map(item_dict)


    qp=[]
    # Obtiene los valor que no se repitan 
    for i in df['SUPPLIER'].values: 
        if i not in qp:
            qp.append(i)

    item_dict = {item: index + 1 for index, item in enumerate( qp)}
    df['SUPPLIER'] = df['SUPPLIER'].map(item_dict)


    e=[]
    # Obtiene los valor que no se repitan 
    for i in df['ITEM TYPE'].values: 
        if i not in e:
            e.append(i)
    item_dict = {item: index + 1 for index, item in enumerate(e)}
    df['ITEM TYPE'] = df['ITEM TYPE'].map(item_dict)

    print('----SHUFFLED 2----')
    # REVOLVER VALORES O SHUFFLED
    # Establecer la semilla para reproducibilidad
    np.random.seed(42)
    # Mezclar el DataFrame
    df= df.sample(frac=1, random_state=42).reset_index(drop=True)
   
    print('----CORRELACIONAR PARA POSIBLE ELIMINACIÓN----')
    correlacion_data=df.corr()
    print(correlacion_data)

    # 3. Mostrar la matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion_data, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matriz de Correlación")
    plt.show()
    #pd.plotting.scatter_matrix(df)
    #plt.show()
    print(correlacion_data['RETAIL SALES'].sort_values(ascending=False))
    print('----GENERAR UNA REGRESIÓN LINEAL TEÓRICA----')
    xt= df.drop(columns=['RETAIL SALES','SUPPLIER','YEAR','MONTH','ITEM DESCRIPTION'])
    x = sm.add_constant(xt)  # Añadir la constante
    print(x)
    y = df['RETAIL SALES']
    # Alinear los datos
    x, y = x.align(y, join='inner', axis=0)
    # Ajustar el modelo
    model = sm.OLS(y, x).fit()
    # Mostrar resumen del modelo
    print(model.summary())
    print('----EVALUAMOS SI HAY MULTICOLINALIDAD---')
    X = sm.add_constant(xt)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



    for num,i in enumerate(vif_data['VIF']):
        if i>10:
            print('Multiconealidad',num,i)
        elif i>5:
            print('Correlación moderada que podría ser preocupante',num,i)
        elif 1<i and i<5:
            print(' Cierta correlación entre la variable independiente,no es la suficiente para ser preocupante',num,i)
        else:
            print('Escenario idela',num,i)
    print(vif_data)

    print('----DIVIDIR LA DATASET----')
    # Calcular el tamaño de los conjuntos
    total_size = len(df)
    test_size = int(total_size * 0.15)
    validation_size = int(total_size * 0.10)
    train_size = total_size - test_size - validation_size

    # Dividir en entrenamiento, validación y prueba
    df_train = df[:train_size]
    df_val = df[train_size:train_size + validation_size]
    df_test = df[train_size + validation_size:]
    
    # ENTRENAMIENTO 75%
    # Separar características y variable objetivo
    X_train = df_train.drop(columns=['RETAIL SALES','SUPPLIER','YEAR','MONTH','ITEM DESCRIPTION'])
    y_train = df_train['RETAIL SALES']

    # VALIDACION 10% 
    X_val = df_val.drop(columns=['RETAIL SALES','SUPPLIER','YEAR','MONTH','ITEM DESCRIPTION'])
    y_val = df_val['RETAIL SALES']

    # TEST 15%
    X_test = df_test.drop(columns=['RETAIL SALES','SUPPLIER','YEAR','MONTH','ITEM DESCRIPTION'])
    y_test = df_test['RETAIL SALES']

    joblib.dump((X_train, y_train, X_val, y_val, X_test, y_test), 'dataset1_split.pkl')
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
    print(f"Tamaño del conjunto de validación: {X_val.shape[0]}")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")

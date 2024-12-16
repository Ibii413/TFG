from matplotlib import scale
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn import metrics,svm as svmm
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle


def gaussiannoise(rt_dest_mod,desviacion):
    ruido_gausiano = np.random.normal(0,desviacion,rt_dest_mod.shape)
    
    
    rmse = np.sqrt(mean_squared_error(rt_dest_mod.flatten(), ruido_gausiano.flatten()))
    psnr_value = cv2.PSNR(rt_dest_mod, rt_dest_mod + ruido_gausiano)

    #print(f"RMSE (Raiz del Error Cuadrático Medio): {rmse}")
    #print(f"PSNR (Relación Señal a Ruido en Pico): {psnr_value}")
    
    return np.clip(rt_dest_mod + ruido_gausiano + 1e-8,-0.30, 1.5),rmse,psnr_value


#SVM usando one bs one
def svm(X_train,  X_test, y_train, y_test):
    
    modelo = svmm.SVC(kernel='poly',decision_function_shape="ovo", C=10,coef0=10, degree=5, gamma='scale', probability=True) #Tmabien probar con ovr
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    
    
    
    """ param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [3, 4, 5],
        'coef0': [0, 1, 10],
    }

    svm = svmm.SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Mejores parámetros encontrados:", grid_search.best_params_)
    modelo_optimo = grid_search.best_estimator_ """
   

    return y_pred, modelo
    
    
def random_forest(X_train,  X_test, y_train, y_test):
    
    modelo = RandomForestClassifier(n_estimators = 200, criterion = 'gini', max_depth= 20,min_samples_split=2, min_samples_leaf=1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    
   
    
    """ param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Mejores parámetros encontrados:", grid_search.best_params_)
    modelo = grid_search.best_estimator_
 """
    return y_pred, modelo
    
def knn(X_train,  X_test, y_train, y_test):
    
    modelo = KNeighborsClassifier(n_neighbors=15, algorithm= 'ball_tree', leaf_size=20, metric='manhattan', weights='distance')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    
    
    """  param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],  # Número de vecinos
        'weights': ['uniform', 'distance'],  # Forma de ponderar los vecinos
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # Métrica de distancia
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algoritmo para búsqueda de vecinos
        'leaf_size': [20, 30, 40, 50],  # Solo relevante para ball_tree y kd_tree
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Mejores parámetros encontrados:", grid_search.best_params_) """
    
    return y_pred, modelo
        
        
    
def clasificacionUno(x,y):
    
    resultados = []
    tipo ="SVM"
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    if tipo == 'Random Forest':
        y_pred, modelo = random_forest(X_train, X_test, y_train, y_test)
    elif tipo == 'SVM':
            y_pred, modelo = svm(X_train, X_test, y_train, y_test)
    elif tipo == 'KNN':
            y_pred, modelo = knn(X_train, X_test, y_train, y_test)
    else:
        raise ValueError("Modelo no reconocido.")
        
    report = classification_report(y_test, y_pred,output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    resultados.append({
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"]
    })
    with open('resultados_metricas.txt', 'w') as file:
        file.write("Resultados:\n\n")
        for resultado in resultados:
            file.write(f"  Accuracy: {resultado['accuracy']:.4f}\n")
            file.write(f"  Precision: {resultado['precision']:.4f}\n")
            file.write(f"  Recall: {resultado['recall']:.4f}\n")
            file.write(f"  F1 Score: {resultado['f1_score']:.4f}\n")
            file.write("\n") 

      
        
    #torch.save(modelo,'modelo_'+tipo+'.pth')

def clasificacion(x,y):
    tipo ="Random Forest"
    x=np.array(x)
    y=np.array(y)-1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(x,y)):
        print(f"Fold {fold + 1}")
        print(train_index.shape)
        print(test_index.shape)
        
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        if tipo == 'Random Forest':
            y_pred, modelo = random_forest(X_train, X_test, y_train, y_test)
        elif tipo == 'SVM':
                y_pred, modelo = svm(X_train, X_test, y_train, y_test)
        elif tipo == 'KNN':
                y_pred, modelo = knn(X_train, X_test, y_train, y_test)
        else:
            raise ValueError("Modelo no reconocido.")
        
        report = classification_report(y_test, y_pred,output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        resultados.append({
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
        })
        with open('resultados_metricas.txt', 'w') as file:
            file.write("Resultados:\n\n")
            for resultado in resultados:
                file.write(f"  Accuracy: {resultado['accuracy']:.4f}\n")
                file.write(f"  Precision: {resultado['precision']:.4f}\n")
                file.write(f"  Recall: {resultado['recall']:.4f}\n")
                file.write(f"  F1 Score: {resultado['f1_score']:.4f}\n")
                file.write("\n") 

      
    
   
    torch.save(modelo,'modelo_'+tipo+'.pth')


def test(modelo,cubo,h,w,mapas_prob,id,valor):
    
    prediccion= modelo.predict(cubo)
    probabilidad = modelo.predict_proba(cubo)
    #print(probabilidad.shape)
    #print(prediccion)
    
    cubo_proba = probabilidad.reshape(h,w,-1)
    """  for clase in range(cubo_proba.shape[2]):        
        filename = f'{mapas_prob}_{id}_{valor}_{clase}.png'
        plt.imshow(cubo_proba[:, :, clase], cmap='viridis')
        plt.colorbar()
        plt.title(f'Probabilidad de clase {clase}')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close() """
    
    
    colores = np.array([
    [0, 1, 0],      # Tejido sano: Verde
    [1, 0, 0],      # Tumor: Rojo
    [0, 0, 1],      # Vena: Azul
    [1, 1, 0],      # Duramadre: Amarillo
    ], dtype=np.float64)
    
    colores = (colores * 255).astype(np.uint8)
    probability_map = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(4): 
        probability_map += cubo_proba[:, :, i:i+1] * colores[i]
        
    
    probability_map_normalized = np.clip(probability_map / 255.0, 0, 1)
    

    plt.imshow(probability_map_normalized)
    plt.title("Probability Map")
    plt.axis("off")
    #plt.show()
    plt.imsave(f'{mapas_prob}_{id}_{valor}_mapa_de_probabilidad.png',probability_map_normalized)
    return prediccion,probabilidad




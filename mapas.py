import sys
import json
import re
import numpy as np
import torch
from dataloader import load_img
import ruido
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def main():
    
    in_path = sys.argv[1]
    json_dict = json.load(open(in_path))
    IDs = json_dict["IDs"]
    geom_cubes_folder = json_dict["geom_cubes_folder"]
    modelos_folder = json_dict["modelos_clas"]
    modelos_planos_folder = json_dict["modelos_planos_clas"]
    mapas_prob_folder = json_dict["mapas_prob_folder"]
    GT_folder = json_dict["GT_folder"]
    plano_folder = json_dict["raw_cubes_folder"]
    
    tipo ="Random Forest"
    modelo = torch.load(f"{modelos_folder}modelo_{tipo}.pth")
    modelo_plano = torch.load(f"{modelos_planos_folder}modelo_{tipo}.pth")
    
    
    for idp in IDs:
        print(f"---> {idp}")
        nums_list = re.findall("\d+", idp)
        id_num, _ = int(nums_list[0]), int(nums_list[1])
        desviacion = [0,0.000001,0.000002,0.000003]
        #desviacion = [0,0.17,0.2,0.2,0.5]
        
        gt = load_img(GT_folder,id_num,valor = None)
        filas,columnas = np.where(gt >0)
        gt_valido = gt[filas,columnas]
        y = []
        for etiqueta in (gt_valido):
                y.append(etiqueta)
        y=np.array(y)-1  
        resultados = []
        
        for valor in desviacion:
            x = []
            
            cubo = load_img(geom_cubes_folder,id_num,valor)
            
            """ h,w, bandas = cubo.shape
            
            
            for fila in range(h):
                for columna in range(w):
                    x.append(cubo[fila, columna, :])
                    
            ruido.test(modelo,x,h,w,mapas_prob_folder,idp,valor)"""
            
            cubo_valido = cubo[filas,columnas, :] 
        
            
            for fila,etiqueta in zip(filas,gt_valido):
                x.append(cubo_valido[fila, :])
                
                
            x=np.array(x)
            
            
            prediccion = modelo.predict(x)
            
            print(f"Clases reales en gt_valido: {np.unique(y)}")
            print(f"Clases predichas en prediccion: {np.unique(prediccion)}")

            print(f"Cantidad de valores en x: {len(prediccion)}")
            print(f"Cantidad de valores en gt_valido: {len(y)}")
            
            accuracy = accuracy_score(y, prediccion)
            report = classification_report(y, prediccion, output_dict=True, zero_division=1)
        
            resultados.append({
            "desviacion": valor,
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"] })
            
           

        with open(f'./dataset/resultados/metrics_{idp}.txt', 'w') as f:
            for res in resultados:
                f.write(f"Desviación: {res['desviacion']}, "
                    f"Accuracy: {res['accuracy']:.6f}, "
                    f"Precision: {res['precision']:.6f}, "
                    f"Recall: {res['recall']:.6f}, "
                    f"F1-Score: {res['f1_score']:.6f}\n")
             
        resultados = [] 
        cubo_plano = load_img(plano_folder,id_num,valor = 0)
        x = []
        h,w, bandas = cubo_plano.shape
        
        cubo_plano = cubo_plano[filas,columnas, :]
        
        for fila,etiqueta in zip(filas,gt_valido):
                x.append(cubo_plano[fila, :])
        
        x=np.array(x)
        prediccion = modelo_plano.predict(x)
        
        accuracy = accuracy_score(y, prediccion)
        report = classification_report(y, prediccion, output_dict=True, zero_division=1)
        
        resultados.append({
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"] })
        
        with open(f'./dataset/resultados/metricas_planas_{idp}.txt', 'w') as f:
            f.write(f"IDP: {idp}\n")
            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write(f"Precision: {report['weighted avg']['precision']:.6f}\n")
            f.write(f"Recall: {report['weighted avg']['recall']:.6f}\n")
            f.write(f"F1-Score: {report['weighted avg']['f1-score']:.6f}\n")  
            
if __name__ == "__main__":
    main()
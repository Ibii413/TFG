from dataloader import load_img
import ruido
import os
import numpy as np
from scipy.io import loadmat
import sys
import json
import re


def main():
    
    in_path = sys.argv[1]
    json_dict = json.load(open(in_path))
    GT_folder = json_dict["GT_folder"]
    IDs = json_dict["IDs"]
    geom_cubes_folder = json_dict["raw_cubes_folder"]
    
    x = []
    y = []
    
    for idp in IDs:
        print(f"---> {idp}")

        nums_list = re.findall("\d+", idp)
        id_num, _ = int(nums_list[0]), int(nums_list[1])
        
        cubo = load_img(geom_cubes_folder,id_num,valor = 0)
        
        
        gt = load_img(GT_folder,id_num,valor = None)
        
        filas,columnas = np.where(gt >0)
        gt_valido = gt[filas,columnas]
        cubo_valido = cubo[filas,columnas, :]
        print(cubo_valido.shape)
        

        for fila,etiqueta in zip(filas,gt_valido):
            x.append(cubo_valido[fila, :])  # Las bandas del píxel
            y.append(etiqueta)  

    ruido.clasificacion(x,y)
    
    
if __name__ == "__main__":
    main()
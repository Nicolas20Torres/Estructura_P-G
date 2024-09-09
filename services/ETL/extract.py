# importacion de recursos
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
print(sys.path)

class Extract:
    def __init__(self,path=None):
        self.path = path
        self.df = None
    
    def read_file_csv(self,name_file,separador):
        self.name_file = name_file
        full_path = os.path.join(self.path,self.name_file)
        self.df = pd.read_csv(full_path,encoding='utf-8',sep=separador)
        return self.df

    def read_file_excel(self,name_file,sheet_name,header=0):
        self.name_file = name_file
        self.sheet_name = sheet_name
        full_path = os.path.join(self.path,self.name_file)
        self.df = pd.read_excel(full_path,sheet_name=self.sheet_name,header=header)
        return self.df

    def read_folder(self):
        """
        read_file: Lee los archivos en formato texto de una carpeta
        self.path (str): ruta de accedo a la carpeta
        """
        files = []
        # Recorre los archivos de la carpeta
        for Nombre_Carpeta, _, Nombre_Archivos in os.walk(self.path):
            for Nombre_Archivo in Nombre_Archivos:
                # Validar Archivo .txt
                if Nombre_Archivo.endswith('.txt'):
                    Archivo_Carpeta = os.path.join(Nombre_Carpeta, Nombre_Archivo)
                    with open(Archivo_Carpeta,'r', encoding='utf-8') as file:
                        df = pd.read_csv(file, sep='|')
                    # crear un DataFrame con pandas
                    df_ = pd.DataFrame(data=df)
                    # Agregar el df al arreglo vacio DataFrame 
                    files.append(df_)
        self.df = pd.concat(files, axis=0)
        return self.df

    def read_DataFrame(self,datos):
        self.df = pd.DataFrame(datos)

    def read_data_set(self,name_dataset):
        """
        Lee y carga el DataSet a un DataFrame. 
        Parameters:
         - name_dataset -> (str): Nombre del Dataframe
        Sintaxis: 
        >> load_data = LoadData(DataFrame,directorio_origen)
        >> load_data.read_data_set(nombre_del_dataset)
        """
        try:
            ruta = os.path.join(self.path,name_dataset)
            ruta_ = ruta + '.pkl'
            with open(ruta_,'rb') as file:
                self.df = pickle.load(file)
                print('Dataset cargado exitosamente')
                return self.df
        except Exception as e:
            print(f'Error al leer el DataSet: {e}')
            return None
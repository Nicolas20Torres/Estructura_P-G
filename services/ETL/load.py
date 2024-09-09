# Recursos
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

class LoadData:
    """
    Instancia para guadar guardar los datos en un directorio
    Parametros: datos -> (DataFrame): Dataframe de pandas con los datos listos para ser guardados
    Sintaxis:
    >>> load_data = Loaddata(DataFrame,Directorio_donde_se_guarda)
    """
    def __init__(self,datos,full_path):
        self.datos = datos
        self.full_path = full_path
        self.save = pd.DataFrame()
    
    def load_data(self,name_file,type_file,date_=True):
        """
        load_data: Guardar un archivos en un formato de texto con la opcion de poner o no 
        el sufilo de mes ano, esto con el fin de poner guardar la fecha en el nombre del archivo.
        Parameters:
         - name_file -> (str): Nombre del archivos
         - type_file -> (str) = 'plano' metodo en constuccion, solo tiene metodo de guadardo
         de archivo plano.
         - date_ -> (bool): True para poner el año mes en vigente en el nombre del archivo False ára no.
         Sintaxsis:
         >>> save = LoadData()
         >>> save.load_data('nombre del archivio','plano',date_=True)

        """
        current_date = datetime.now()
        date = current_date.strftime('%Y%m')

        if date_:
            if type_file == 'plano':
                full_name_file = f'{name_file}_{date}.txt'
                path_ = os.path.join(self.full_path,full_name_file)
                self.save = self.datos.to_csv(path_,encoding='utf-8',sep='|',index=False)
        else: 
            if type_file == 'plano':
                full_name_file = f'{name_file}.txt'
                path_ = os.path.join(self.full_path,full_name_file)
                self.save = self.datos.to_csv(path_,encoding='utf-8',sep='|',index=False)

    def load_data_set(self,name_dataset):
        """
        load_data_set: Alamcena los datos de un Dataframe
        Parametros: 
            - name_dataset -> (str): Nombre del dateset con la extencion
        """
        try:
            ruta = os.path.join(self.full_path,name_dataset)
            ruta_ = ruta + '.pkl'
            with open(ruta_, 'wb') as file:
                pickle.dump(self.datos,file)
                print(f'Dataset guardado exitosamente como {name_dataset}')
        except Exception as e:
            print(f'Error al guardar DataSet: {name_dataset} error: {e}')
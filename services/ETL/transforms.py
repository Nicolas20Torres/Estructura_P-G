# Recursos
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from services.ETL.extract import Extract

class Transfomrs(Extract):
    """
    Transfomrs: Clase que tiene como proposito generar las trasnformaciones derivadas del cargue de los
    datos usando la clase Extract
    """
    def __init__(self,path,name_file='',sheet_name=None,file_type='',separador=None,datos=None,header=0):
        """
        __init__: Instancia de la clase Transfomrs:
         - path (str): Ruta o directorio donde se desea leer el archivo
         - name_file (str): nombre del archivo con extencion  
         - sheet_name (str): nombre de la hoja excel si aplica
         - file_type (str): tipo de archivo a usar, disponible: 'csv','excel'.  
        """
        super().__init__(path)
        if file_type == 'csv':
            self.read_file_csv(name_file,separador)
        elif file_type == 'excel':
            self.read_file_excel(name_file,sheet_name,header)
        elif file_type == 'dataset':
            self.read_data_set(name_file)
        elif file_type == 'DataFrame':
            self.read_DataFrame(datos)
        elif file_type == 'folder':
            self.read_folder()
        else:
            raise ValueError(f'El tipo de archivo {file_type} NO definido en el metodo.')
        
  
    def get_method(self,metodo,*args,**kwargs):
        """
        get_method: invoca cualquier metodo de la clase Transform
         - *args (list): lista de argumento
         - *kwargs (dcit)_ diccionario con los elementos clave valor que diligencian
            los parametro del metodo en cuestion.
        """
        # Validacion para ejecutacion de metdo invocado
        try:
            metodo = self.__getattribute__(metodo)
            if callable(metodo):
                return metodo(*args,**kwargs)
            else:
                print(f'el metodo {metodo} no es un metodo valido')
                return None
        except AttributeError:
            print(f'no se encontro el metodo {metodo}.')
            return None
    
    def columns_selection(self,columns):
        """
        columns_selection: Selecciona las columnas de un DataFrame
         - columns (srt, list): los las columnas se tomaran para seleccion en el DataFrame
        """
        if self.df is None or self.df.empty:
            print('El DataFrame esta vacio')
            return None
        try:
            query = self.df[columns]
            self.df = query
            return query
        except KeyError as e:
            print(f'Error al seleccionar las columnas: {e}')

    def vlookup(self,value_left_table,value_right_table,left_table=None,right_table=None):
        """
        vlookup: Busca elementos coindidente en dos tablas
         - value_left_table (str): Es la columna de la tabla de la izquierda que se usara para
           hacer el cruce
         - value_rigth_table (str): Es la columna de la tabla de la derecha que se usara para
           hacer el cruce
         - left_table (DataFrame): 
        """
        # Variables tabla izquierda y derecha del cruce
        if left_table is None:
            left_table = self.df
        if right_table is None:
            right_table = self.df
        # Cruce de los datos usando en metodo join de pandas
        query = left_table.join(right_table.set_index(value_right_table),on=value_left_table,how='left',lsuffix='l',rsuffix='r')
        self.df = query
        return query
        
    def condicional_columns_vlookup(self, column, conditional, result_true, result_false):
        """
        condicional_columns_vlookup: 
        Evalua una expresion en una columna de un DataFrame y como resultado hace dos busquedas una
        cuando se cumple con la condicion (true) y la otra cuando se cumple (false),
        es principalemente util para homologaciones de campos de DataFrame.
         - columns (str): Se pasa como parametro del nombre de una columna preteneciente al Dataframe
         - conditional (list): Evalua la expresion como igual (=) a columns
         - result_true (dict): parametros del metodo vlookup
         - result_false (dict): parametros del metodo vlookup 
            * Ejemplo: 
                dict = {
                    'metodo':'vlookup',
                    'left_table':self.df,
                    'right_table':self.df,
                    'value_left_table':'columna de la tabla izquierda',
                    'value_right_table':'columna de la tabla derecha'
                    }
        """
        # Separacion de los DataFrame para trabajar las variables
        df_values_true = self.df[self.df[column].isin(conditional)]
        df_values_false = self.df[~self.df[column].isin(conditional)]
        
        list_rows_true = []
        list_rows_flase = []
        
        # Validación y ejecución para ejecucion verdadera
        result_true_mask = (result_true is not None) & isinstance(result_true, dict)
        if result_true_mask:
            result_true['left_table'] = df_values_true
            filter_true = self.get_method(**result_true)
            if filter_true is not None:
                list_rows_true.append(filter_true)
            else:
                print('Error al validar el DataFrame (Condicion True)')

        # Validación y ejecución para condicional falsa 
        result_false_mask = (result_false is not None) & isinstance(result_false, dict)
        if result_false_mask:
            result_false['left_table'] = df_values_false
            filtre_false = self.get_method(**result_false)
            if filtre_false is not None:
                list_rows_flase.append(filtre_false)
            else:
                print('Error al validar el DataFrame (Condicion False)')
        
        if list_rows_true is not None and list_rows_flase is not None:
            # Definicion de los DataFrame:
            df_true = pd.concat(list_rows_true,ignore_index=True)
            df_false = pd.concat(list_rows_flase,ignore_index=True)
            # Union de los DataFrame
            query = pd.concat([df_true,df_false],axis=0)
            self.df = query
            return query

        # Mensaje de error si no se cumplen las condiciones
        if not result_true_mask.any() and not result_false_mask.any():
            print('Error al ejecutar el metodo conditional_columns_vlooup')
    
    def conditional_empty_fields(self,name_column_1,name_column_2,name_new_colmun,delete_columns=False):
        """
        conditional_two_fields: Toma dos campos de un DataFrame y evalua cual de los esta vacio y devuele un nuevo
        campos con el valor de la columna no vacia
         - name_column_1 (str): Es el nombre de una columna existente del DataFrame
         - name_column_2 (str): Es el nombre de una columna existente del DataFrame
         - new_name_column (str): nombre de la columna que llevara el resultado de condicion   
        """
        mask_1 = self.df[name_column_1].isna()
        mask_2 = self.df[name_column_2].isna()
        # np.select hace la validacion a fin de dejar la columna NO vacia
        self.df[name_new_colmun] = np.select([~mask_1,~mask_2],[self.df[name_column_1],self.df[name_column_2]])
        if delete_columns:
            self.df.drop([name_column_1,name_column_2],axis=1,inplace=True)
    
    def correct_date(self,name_columns,characters):
        """
        correct_date: Toma como parametros una lista de caracteres y los reemplaza por / 
        a fin de ajustar la fecha en formato DD/MM/YYYY
         - name_columns (str): Nombre de la columna para trabajar como fecha
         - charateres (list): Lista con los caracteres a reemplazar 
        """
        for columns in name_columns:
            if  isinstance(characters, list):
                for char in characters:
                    self.df[columns] = self.df[columns].astype(str)
                    self.df[columns] = self.df[columns].str.replace(char,'/')
                    self.df[columns] = pd.to_datetime(self.df[columns],errors='coerce')
            else:
                raise ValueError('El parametro no es una lista') 
        return self.df

    def date_format_V2(self,field_dates,columns_details=False):
        """
        Convierte una columna en formato de fecha y opcionalmente extrae año, mes y día en columnas separadas.

        Args:
        - field_dates (str): Nombre de la columna que se convertirá en formato fecha.
        - columns_details (bool): Si es True, extrae año, mes y día en columnas separadas. Default es False.

        Returns:
        - pandas.DataFrame: DataFrame con la columna de fecha convertida y opcionalmente las columnas de año, mes y día.

        Raises:
        - ValueError: Si la columna especificada no contiene valores válidos de fecha.

        Example:
        >>> import pandas as pd
        >>> data = {'fecha': ['2021-01-01 12:00:00', '2021-02-01 15:30:00', '03/15/2021', '2021/04/01']}
        >>> df = pd.DataFrame(data)
        >>> processor = DataFrameProcessor(df)
        >>> processed_df = processor.date_format_V2('fecha', columns_details=True)
        >>> print(processed_df)
                       fecha  Anio       Mes  Dia
        0 2021-01-01      2021    January    1
        1 2021-02-01      2021   February    1
        2 2021-03-15      2021      March   15
        3 2021-04-01      2021      April    1
        """
        def fecha(fecha_str):
            fecha_solo = fecha_str.split(' ')[0]
            partes = fecha_solo.replace('-','/').split('/')
            if len(partes[0]) == 4:  # Si la longitud es 2, entonces es el día
                anio = partes[0][:4]
                mes = partes[1][:2]
                dia = partes[2][:2]
            else:  # Si la longitud es mayor o igual a 4, entonces es el año
                dia = partes[0][:2]
                mes = partes[1][:2]
                anio = partes[2][:4]
        
            fecha_formateada = f'{dia}/{mes}/{anio}'
            return fecha_formateada
        try:
            self.df[field_dates] = self.df[field_dates].astype(str)
            self.df[field_dates] = self.df[field_dates].apply(lambda x: fecha(x)) 
            self.df[field_dates] = pd.to_datetime(self.df[field_dates],dayfirst=True)

            if self.df[field_dates].isnull().all():
                raise ValueError(f'La columna {field_dates} no contiene valores correctos para trabajar con fechas')
            if columns_details:
                # crear columna año tratando el valor como numero entero
                self.df['Anio'] = self.df[field_dates].dt.year.fillna(0).astype(int)
                # crear columna con el nombre del mes
                self.df['Mes'] = self.df[field_dates].dt.strftime('%B')
                # Crear columna con el numero del dia de mes
                self.df['Dia'] = self.df[field_dates].dt.day.fillna(0).astype(int)
        except Exception as e:
            print(f'Error al convertir la columna en formato fecha: {e}')
        self.df[field_dates] = pd.to_datetime(self.df[field_dates]).dt.date

        return self.df
        
    def fill_empty_column(self, column_name, new_column_name, value_if_empty='NO', value_if_not_empty='SI'):
        """
        fill_empty_column: Verifica si una columna está vacía y asigna valores según la condición.
         - column_name (str): Nombre de la columna a verificar.
         - new_column_name (str): Nombre de la nueva columna resultante.
         - value_if_empty (any): Valor a asignar si la columna está vacía.
         - value_if_not_empty (any): Valor a asignar si la columna no está vacía.
        """
        self.df[new_column_name] = np.where(self.df[column_name].isna(), value_if_empty, value_if_not_empty)
        return self.df
    
    def values_condicional_column(self,column_name,new_name_column,value_true,value_false,result_true,result_false,value_default):
        """
            values_condicional_column: Evalúa las condiciones en la columna column_name y asigna valores a new_name_column
            basado en las condiciones.
            
            - column_name (str): Nombre de la columna existente en el DataFrame que se evaluará.
            - new_name_column (str): Nombre de la nueva columna que contendrá el resultado de la evaluación.
            - value_true (str): Valor que se considera verdadero en la columna column_name.
            - value_false (str): Valor que se considera falso en la columna column_name.
            - result_true (str): Resultado a asignar cuando la condición es verdadera.
            - result_false (str): Resultado a asignar cuando la condición es falsa.
            - value_default (str): Valor por defecto a asignar cuando ninguna condición se cumple.
            
            Returns:
            - self.df (DataFrame): DataFrame con la nueva columna añadida.
            
            Ejemplo de uso:
            
            data = {
                'nombre': ['Juan', 'Ana', 'Pedro', 'Luis', 'Sofia'],
                'estatus': ['Aprobado', 'Reprobado', 'Aprobado', 'Pendiente', 'Aprobado']
            }
            df = pd.DataFrame(data)
            transformer = DataFrameTransformer(df)
            transformer.values_condicional_column(
                column_name='estatus',
                new_name_column='resultado',
                value_true='Aprobado',
                value_false='Reprobado',
                result_true='Pasa',
                result_false='No pasa',
                value_default='En espera'
            )
        """ 
        conditions = [self.df[column_name]==value_true,self.df[column_name]==value_false]
        result = [result_true,result_false]
        self.df[new_name_column] = np.select(conditions,result,default=value_default)
        return self.df
        
    def int_format(self,name_columns):
        """
        int_format: Convierte a numeros la columnas selecciona
        name_columns (str): nombre de la columnas de DataFrame existente
        """
        self.df[name_columns] = self.df[name_columns].fillna(0).astype(int)
        return self.df
    

    def tranform_values_int(self, columnas):
        """
        Transforms the specified columns in the DataFrame to integers.
        
        Args:
        columnas (list): List of column names in the DataFrame to be transformed.
        """
        self.df[columnas] = self.df[columnas].fillna(0)

        def find_decimal_separator(value):
            if '.' in value and ',' in value:
                return '.' if value.index('.') > value.index(',') else ','
            elif '.' in value:
                return '.'
            elif ',' in value:
                return ','
            return None

        for columna in columnas:
            self.df[columna] = self.df[columna].astype(str)
            self.df['evaluator'] = self.df[columna].apply(find_decimal_separator)
            
            if self.df['evaluator'].eq(',').any():  
                self.df[columna] = self.df[columna].str.replace('.', '')
                self.df[columna] = self.df[columna].str.replace('(', '-')
                self.df[columna] = self.df[columna].str.replace(',', '.')
            else:
                self.df[columna] = self.df[columna].str.replace(',', '')
                self.df[columna] = self.df[columna].str.replace('(', '-')

            self.df[columna] = self.df[columna].astype(float).astype(int)

        self.df.drop(columns=['evaluator'], inplace=True)
        return self.df

    def filtrar_tabla(self, conditions_list):
        """
        Filtra el DataFrame basado en una lista de condiciones.

        Args:
            conditions_list (list): Lista de diccionarios con las condiciones a evaluar. Cada diccionario debe 
                                    tener las claves 'column_name', 'operator' y 'value'.
                                    Ejemplo: [{'column_name': 'col1', 'operator': '==', 'value': 10}, ...]

        Returns:
            pd.DataFrame: El DataFrame filtrado basado en las condiciones proporcionadas.
        """
        datos = self.df.copy()

        # Generar las condiciones lógicas
        combined_condition = pd.Series([True] * len(datos))
        
        for cond in conditions_list:
            if cond['operator'] == '==':
                condition = datos[cond['column_name']] == cond['value']
            elif cond['operator'] == '!=':
                condition = datos[cond['column_name']] != cond['value']
            elif cond['operator'] == '>':
                condition = datos[cond['column_name']] > cond['value']
            elif cond['operator'] == '>=':
                condition = datos[cond['column_name']] >= cond['value']
            elif cond['operator'] == '<':
                condition = datos[cond['column_name']] < cond['value']
            elif cond['operator'] == '<=':
                condition = datos[cond['column_name']] <= cond['value']
            elif cond['operator'] == 'vacio':
                condition = datos[cond['column_name']].isna()
            elif cond['operator'] == 'notna':
                condition = datos[cond['column_name']].notna()
            elif cond['operator'] == 'in':
                condition = datos[cond['column_name']].isin(cond['value'])
            elif cond['operator'] == 'not in':
                condition = ~datos[cond['column_name']].isin(cond['value'])
            else:
                raise ValueError(f"Operador desconocido: {cond['operator']}")

            combined_condition &= condition

        return datos[combined_condition]

    def agregar_etiqueta(self, conditions_list, nombre_nueva_columna, valor_nueva_columna, nombre_nivel0_columna=None, valor_nivel0_columna=None):
        """
        Agrega una nueva columna con un valor específico basado en condiciones.

        Args:
            conditions_list (list): Lista de diccionarios con las condiciones a evaluar.
            nombre_nueva_columna (str): Nombre de la nueva columna.
            valor_nueva_columna (any): Valor a asignar en la nueva columna si se cumplen las condiciones.
            nombre_nivel0_columna (str, optional): Nombre de la columna de nivel 0.
            valor_nivel0_columna (any, optional): Valor a asignar en la columna de nivel 0.
        """
        filtered_df = self.filtrar_tabla(conditions_list)
        self.df.loc[filtered_df.index, nombre_nueva_columna] = valor_nueva_columna
        
        if nombre_nivel0_columna and valor_nivel0_columna:
            self.df.loc[filtered_df.index, nombre_nivel0_columna] = valor_nivel0_columna

    def aplicar_etiquetas(self, etiquetas):
        """
        Aplica múltiples etiquetas al DataFrame.

        Args:
            etiquetas (list): Lista de tuplas, donde cada tupla contiene:
                              (conditions_list, nombre_nueva_columna, valor_nueva_columna, [nombre_nivel0_columna, valor_nivel0_columna])
        """
        for etiqueta in etiquetas:
            if len(etiqueta) == 3:
                conditions_list, nombre_nueva_columna, valor_nueva_columna = etiqueta
                self.agregar_etiqueta(conditions_list, nombre_nueva_columna, valor_nueva_columna)
            elif len(etiqueta) == 5:
                conditions_list, nombre_nueva_columna, valor_nueva_columna, nombre_nivel0_columna, valor_nivel0_columna = etiqueta
                self.agregar_etiqueta(conditions_list, nombre_nueva_columna, valor_nueva_columna, nombre_nivel0_columna, valor_nivel0_columna)
            else:
                raise ValueError("Las tuplas en 'etiquetas' deben tener 3 o 5 elementos.")
            

    def check_dynamic_conditionsV2(self, conditions_list, new_name_column, results,resulf_default=None):
        """
        Evalúa condiciones dinámicas en columnas específicas y retorna el resultado en una nueva columna.

        Este método evalúa una lista de condiciones en diferentes columnas del DataFrame y asigna valores 
        específicos según el resultado de la evaluación de esas condiciones.

        Args:
            conditions_list (list): Lista de diccionarios con las condiciones a evaluar. Cada diccionario debe 
                                    tener las claves 'column_name', 'operator' y 'value'.
                                    Ejemplo: [{'column_name': 'col1', 'operator': '==', 'value': 10}, ...]
            new_name_column (str): Nombre de la nueva columna que contendrá los resultados de la evaluación.
            result_true (any): Valor que se asignará en la nueva columna si la condición es verdadera.
            result_false (any): Valor que se asignará en la nueva columna si la condición es falsa.

        Returns:
            pd.DataFrame: El DataFrame con la nueva columna que contiene los resultados de la evaluación.
        
        Example:
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, np.nan, 4, 5],
            ...     'B': [10, 20, 30, 40, 50]
            ... })
            >>> mi_clase = MiClase(df)
            >>> conditions = [
            ...     {'column_name': 'A', 'operator': '==', 'value': 2},
            ...     {'column_name': 'B', 'operator': '>', 'value': 20},
            ...     {'column_name': 'A', 'operator': 'isna'}
            ... ]
            >>> result_df = mi_clase.check_dynamic_conditionsV2(conditions, 'Result', 'True', 'False')
            >>> print(result_df)
               A   B Result
            0  1  10  False
            1  2  20   True
            2 NaN  30   True
            3  4  40   True
            4  5  50   True
        """
        if len(conditions_list) != len(results):
            raise ValueError("La lista de condiciones y la lista de resultados deben tener la misma longitud.")
        
        conditions = [
            (self.df[cond['column_name']] == cond['value']) if cond['operator'] == '==' else
            (self.df[cond['column_name']] != cond['value']) if cond['operator'] == '!=' else
            (self.df[cond['column_name']] > cond['value']) if cond['operator'] == '>' else
            (self.df[cond['column_name']] >= cond['value']) if cond['operator'] == '>=' else
            (self.df[cond['column_name']] < cond['value']) if cond['operator'] == '<' else
            (self.df[cond['column_name']] <= cond['value']) if cond['operator'] == '<=' else
            (self.df[cond['column_name']].isna()) if cond['operator'] == 'vacio' else
            (self.df[cond['column_name']].notna()) if cond['operator'] == 'notna' else
            (self.df[cond['column_name']].isin(cond['value'])) if cond['operator'] == 'in' else
            (~self.df[cond['column_name']].isin(cond['value'])) if cond['operator'] == 'not in' else
            False
            for cond in conditions_list
        ]
        
        self.df[new_name_column] = np.select(conditions, results, default=resulf_default)
        return self.df
    
    def check_dymanic_conditions(self,conditions,new_name_column,result_true,result_false,value_default):
        """
        check_dynamic_conditions: Evalúa condiciones dinámicamente en column_name y retorna
        el resultado en una nueva columna.
         - new_name_column (str): Nombre de la columna que llevara el resultado de la expresion.
         - result_true (any): Resultado verdadero de la evaluación.
         - result_false (any): Resultado falso de la evaluación.
         - conditions (tuple-dict): Tuplas de condiciones en formato (valor, operador).
        """
        if isinstance(conditions, dict):
            conditions = [self.df[key] == value for key, value in conditions.items()]
        if isinstance(conditions, list):
            conditions = list(conditions)
        else:
            raise ValueError('Las condiciones deben ser o un diccionario o una tupla')
        self.df[new_name_column] = np.select(conditions,[result_true,result_false],default=value_default)
        return self.df
    
    def delete_characters(self,name_column,new_name_column,characters):
        """
        delete_characters: toma una columna como str y eliminar los caracteres dados en los parametros
         - name_column (str) DateFrame[name_column]: nombre de la columna existente en el DataFrame
         - name_column (str) DateFrame[name_column]: nombre de la nueva columna del DataFrame
         - characters (list): Lista con los caracteres que se eliminaran 
        """
        self.df[new_name_column] = self.df[name_column]
        self.df[new_name_column] = self.df[new_name_column].astype(str)
        self.df[new_name_column] = self.df[new_name_column].str.replace(characters,'')
        return self.df
    
    def month_current(self,field_date_time,new_name_column):
        """
        month_current: Valida la fecha actual del sistema y la compara con una columna del DataFrame en formato datetime64
        y determina en una nueva columna el estado del mes en funcion de: 
        * ACTIVO = mes vigente
        * INCATIVO = mes no vigente.
        Parametres:
         - new_name_column (str): nombre de la nueva columna con la validacion [ACTIVO-INACTIVO]
        """
        try:
            date = datetime.now()
            month_current = date.month
            # Verificar si la columna fecha existe y es de tipo fecha
            self.df[field_date_time] = pd.to_datetime(self.df[field_date_time],dayfirst=True)
            if field_date_time not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df[field_date_time]):
                raise ValueError(f'La columna {field_date_time} no existe en el dataFrame o no es de tipo fecha')
            # Crea una columna validando.. a para generar la comparacion
            self.df['validando...'] = self.df[field_date_time].dt.month
            #  Crea la nueva columna con los valores ACTIVO INACTIVO
            self.df[new_name_column] = np.where(self.df[field_date_time].dt.month==month_current,'ACTIVO','INACTIVO')
            self.df.drop('validando...',axis=1,inplace=True)
            self.df[field_date_time] = self.df[field_date_time].dt.strftime('%d/%m/%Y')
            return self.df
        except Exception as e:
            print(f'Error en el metodo month_current: {e}')
      
    def delete_columns(self,name_columns):
        """
        delete_columns: Eliminar las columnas pasadas en los parametros
         - name_columns (list): Nombre de las columnas a eliminar
        """
        if self.df is None or self.df.empty:
            print('El DataFrame esta vacio o es None')
            return None
        try:
            self.df.drop(name_columns,axis=1,inplace=True)
            return self.df  
        except KeyError as e:
            print(f'Error al eliminar las columnas: {e}')
            return None

    
    def rename_columns(self,column_name_mapping):
        """
        rename_columns: Renombra una columna en el DataFrame.
        Parameters:
        - name_column (str): Nombre de la columna existente.
        - new_name_column (str): Nuevo nombre para la columna.
        """
        try:
            self.df.rename(columns=column_name_mapping,inplace=True)
            return self.df
        except Exception as e:
            print(f'Error en el metodo rename_columns: {e}')
            
    def create_columns_date(self):
        self.df['FECHA_REGISTRO'] = datetime.now() 
        return self.df
    
    def replace_values(self, columns_name, replace_values):
        """
        replace_values: Reemplaza los valores en column_name según el diccionario del parámetro replace_values.
        
        Parámetros:
         - columns_name (str): Nombre de la columna del DataFrame
         - replace_values (dict): Diccionario clave valor con los datos a reemplazar:
            dict{'valor1':'nuevo_valor1', 'valorn':'nuevo_valorn'}
        """
        # Asegurarse de que la columna es de tipo str
        self.df[columns_name] = self.df[columns_name].astype(str)
        
        # Reemplazar valores completos usando mapeo de diccionario
        self.df[columns_name] = self.df[columns_name].map(replace_values).fillna(self.df[columns_name])
        
        return self.df
    
    def order_data(self,columns,ascending):
        """
        order_data: Ordena las columnas pasadas en el parametros columns.
        Parameters: 
         - columns -> (list): Lista con los nombre de las columnas existentes en un DataFrame
         - ascending -> (boleano list): True para ordenas acsendentemente y False para descendente         
        """
        try: 
            self.df.sort_values(by=columns,ascending=ascending,inplace=True)
            return self.df
        except Exception as e:
            print(f'Error al ordenar el DataFrame: {e}')
            
    def delete_condition_column(self,col_chasis,col_tipo_documento,valores):
        """
        delete_condition_column: Eliminar los registros coincidentes en la evaludacion, la cual
        consiste en filtar si los datos de cada VIN en su primer registro son notas credito a fin de
        excluirlos del resto de la tabla, dado que no suma en la facturacion. 
        parametres: 
         - chasis -> (str): nombre de la columna existente en el DataFrame que contiene el chasis
         por la cual se quiere encontrar el primero registro existente. 
         - tipo_documentos -> (str): Nombre de la columna que contiene el tipo de documento
         es decir es si suma o resta. 
         - valores -> (list): Lista con los valores contenidos en la columna tipo_documento
         los cuales se buscan filtrar en la condicion.
         Sintaxis: 
         >>> transformacion = Transform()
         >>> list_values = ['CRE','CRE_1','CRE_N']
         >>> transformacion.delete_condition_column('ITM_CHASIS','TRA_TIPO',list_values)
        """
        df_filtro = self.df.copy()
        try:
            eliminar_indices = []
            for chasis in df_filtro[col_chasis].unique():
                primer_chasis = df_filtro.loc[df_filtro[col_chasis]==chasis].iloc[0]
                ultimo_registro = df_filtro.loc[df_filtro[col_chasis]==chasis].tail(1).iloc[0]
                # Validar si el primero registro resta o suma
                if primer_chasis[col_tipo_documento] in valores or ultimo_registro[col_tipo_documento] in valores:
                    eliminar_indices.extend(df_filtro.index[df_filtro[col_chasis]==chasis])
            self.df = df_filtro.drop(eliminar_indices)
        except Exception as e:
            print(f'Error al filtrar el DataFrame: {e}')
    
    def delete_duplicates(self,name_columns,last_record='last'):
        """
        delete_dupicates: eliminar los resgitro duplicados en la columna name_columns dejando
        los ultimos registros 'last' o 'frist' para los primero:
        Parameters: 
         - name_colmuns -> (list): lista con las columnas existentes en el DataFrame.
         - last_record -> (option: 'last' or 'frist'): opcion de eliminacion: manteniendo los primero o 
            ultimos registros. 
        """
        try:
            self.df.drop_duplicates(subset=name_columns,keep=last_record,inplace=True)
        except Exception as e:
            print(f'Error en el momento de eliminar los duplicados en DataFrame: {e}')

    def filter_dataframe_columns(self, column_name, **kwargs):
        """
        Filtra un DataFrame dada una columna con la condicion (=), 
        donde DataFrame[column_name] == condition.

        Parameters: 
            - column_name (str): Nombre de la columna existente en el DataFrame que se desea filtrar.
            - kwargs (dict): Expresiones con los valores filtrados en forma de {operador: valor}. 
            Por ejemplo, {==: 123, !=: 'abc'}.

        Example:
            filter_dataframe_columns_equal(column_name="Nombre Columna", **{==: 123, !=: 'abc'})  
        """
        try:
            dataframe_copy = self.df.copy()
            mask = None
            for comparison, condition in kwargs.items():
                if mask is None:
                    mask = dataframe_copy[column_name] == condition
                else:
                    if comparison == '==':
                        mask &= dataframe_copy[column_name] == condition
                    elif comparison == '!=':
                        mask &= dataframe_copy[column_name] != condition
                    elif comparison == '>':
                        mask &= dataframe_copy[column_name] > condition
                    elif comparison == '<':
                        mask &= dataframe_copy[column_name] < condition
                    # Add more conditions as needed...
            if mask is not None:
                self.df = dataframe_copy[mask]
        except Exception as e:
            print(f'Error al filtrar el DataFrame: {e}')

    def new_column(self,name_column,value):
        """
        new_columns: crear una nueva columna en el DataFrame y le asigna un nuevo valor 
        Parameters:
         - name_columns -> (str): Nombre de la nueva columna
         - value -> (str - int): Valor que la nueva columna va a tomar
        Sintaxis: 
        >>> transformacion = Transform()
        >>> transformacion.new_column(name_column=ejemplo,value=123)
        """
        try:
            if name_column in self.df.columns:
                print(f'La columna {name_column} ya existe en el DataFrame')
            else:
                self.df[name_column] = value
                print(f'Columna {name_column} se agrego correctamente')
        except Exception as e:
            print(f'Error al generar la nueva columna: {e}')   
        
    # def imagen64(self,origen,destino,tamano):
    #     from services.imagenbase64 import ajustar_imagen
    #     df = ajustar_imagen(origen=origen,destino=destino,tamaño=tamano)
    #     self.df = self.df.join(df.set_index('Familia'),on='Familia',how='left')

    def Facturados_no_entregados(self):
        """

        """
        # Filtrar valores que no tiene fecha de entrega
        datos_validacion = self.df.copy()
        
        # Funcion para mostrar el ultimo registro
        def ultimo_registro(serie):
            return serie.iloc[-1]
        # Seleccion de columnas para la tabla
        columnas = [
            'id_registro',
            'Numero de Registro',
            'Fecha',
            'Subtotal',
            'Descuentos',
            'Costos',
            'Cantidad',
            'Costo Traslado',
            'Marca',
            'Gamas',
            'Segmento',
            'Familia',
            'Vitrina',
            'Grupo'
        ]
        # Diccionario de funciones
        funciones = {
            'id_registro':ultimo_registro,
            'Numero de Registro':ultimo_registro,
            'Fecha':ultimo_registro,
            'Subtotal':ultimo_registro,
            'Descuentos':ultimo_registro,
            'Costos':ultimo_registro,
            'Cantidad':ultimo_registro,
            'Costo Traslado':ultimo_registro,
            'Marca':ultimo_registro,
            'Gamas':ultimo_registro,
            'Segmento':ultimo_registro,
            'Familia':ultimo_registro,
            'Vitrina':ultimo_registro,
            'Grupo':ultimo_registro
        }
        # Generacion de tabla dinamica
        f_no_entre = pd.pivot_table(
            data=datos_validacion,
            index='Chasis',
            values=columnas,
            aggfunc=funciones
        ).reset_index()
        self.df = f_no_entre 
        return self.df
    
    def filtrar_valores_vacios(self,nombre_columna,no_vacio=False):
        if nombre_columna not in self.df.columns:
            raise ValueError(f'La columna {nombre_columna} no existe en el DataFrame')
        if no_vacio:
            resultado = self.df[self.df[nombre_columna].isna()]
            self.df = resultado
            return self.df
        else:
            resultado = self.df[~self.df[nombre_columna].isna()]
            self.df = resultado
            return self.df
        
    
    def values_mayor_cero(self,columna):
        if columna not in self.df.columns:
            raise ValueError(f'la columna {columna}, no esta en el DataFrame')
        resultado = self.df[self.df[columna]>0]
        self.df = resultado
        return self.df 
    
    def tabla_dinamica(self,filas=None,columnas=None,valores=None,funciones=dict,campo_cantidad=None):
        if self.df is None or self.df.empty:
            print('El DataFrame esta vacio o es None')
            return None
        try:
            if not isinstance(funciones,dict):
                raise ValueError('El parametro funciones debe ser un diccionario')
            datos = self.df.copy()
            # Crear la tabla dinamica
            dinamica = pd.pivot_table(
                data=datos,
                index=filas,
                columns=columnas,
                values=valores,
                aggfunc=funciones
            ).reset_index()
            if dinamica.empty:
                return None
            if campo_cantidad is not None:
                dinamica = dinamica[dinamica[campo_cantidad]!=0]
            self.df = dinamica
            return self.df
        except KeyError as e:
            print(f'Error al generar la tabla dinamica: {e}')
            return None
    
    def extraer_valor(self,columna_df,nueva_columna,str_extraer):
        self.df[nueva_columna] = self.df[ columna_df].str.extract(str_extraer)
        return self.df
    
    def validacion_cruzada(self,nombre_columna,lista_validacion=None,presentes=True,filtrar_vacios=False):
        """
        Realiza una validación cruzada en el DataFrame basado en una lista de valores, una columna específica,
        y/o campos vacíos.

        Este método filtra el DataFrame según los valores especificados en `lista_validacion` 
        para la columna `nombre_columna`. Dependiendo del parámetro `presentes`, se puede 
        filtrar para mantener solo las filas con valores presentes en `lista_validacion` 
        o eliminar dichas filas. También puede filtrar campos vacíos.

        Args:
            nombre_columna (str): El nombre de la columna en el DataFrame para realizar la validación cruzada.
            lista_validacion (list, optional): Una lista de valores a validar en la columna especificada. 
                                            Por defecto es None.
            presentes (bool, optional): Si es True, mantiene las filas con valores presentes en 
                                        `lista_validacion`. Si es False, elimina las filas con 
                                        valores presentes en `lista_validacion`. El valor 
                                        predeterminado es True.
            filtrar_vacios (bool, optional): Si es True, filtra los campos vacíos (NaN). El valor 
                                            predeterminado es False.

        Returns:
            pd.DataFrame: El DataFrame filtrado según los criterios especificados.
        
        Example:
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, 3, 4, 5],
            ...     'B': ['a', 'b', 'c', 'd', 'e']
            ... })
            >>> mi_clase = MiClase(df)
            >>> resultado = mi_clase.validacion_cruzada('A', [2, 4], presentes=True)
            >>> print(resultado)
            A  B
            1  2  b
            3  4  d
        """
        datos = self.df.copy()
        
        if filtrar_vacios:
            if presentes:
                filtro = datos[nombre_columna].notna()
            else:
                filtro = datos[nombre_columna].isna()
        else:
            if lista_validacion is None:
                raise ValueError("Debe proporcionar una lista de validación o habilitar el filtrado de vacíos.")
            
            if presentes:
                filtro = datos[nombre_columna].isin(lista_validacion)
            else:
                filtro = ~datos[nombre_columna].isin(lista_validacion)
        
        resultado = datos[filtro]
        self.df = resultado
        return self.df

    def filtro_entre_fechas(self,columna_fecha,desde=None,hasta=None,hasta_fecha_max=True):

        """
        filtro_entre_fechas: Filtra el DataFrame entre dos fechas dadas en una columna específica.
        
        Parámetros:
         - columna_fecha (str): Nombre de la columna con fechas a filtrar.
         - desde (str o datetime, opcional): Fecha de inicio para el filtro. Si es None, no se aplica filtro de inicio.
         - hasta (str o datetime, opcional): Fecha de fin para el filtro. Si es None y hasta_fecha_max es False, no se aplica filtro de fin.
         - hasta_fecha_max (bool, opcional): Si es True y hasta es None, usa la fecha máxima de la columna como límite superior.
        
        Retorna:
         - DataFrame filtrado.
        """
        # Validar si la columna está en formato datetime
        try:
            self.df[columna_fecha] = pd.to_datetime(self.df[columna_fecha])
        except Exception as e:
            raise ValueError(f"Error al convertir {columna_fecha} a formato datetime: {e}")

        # Asegurarse de que el índice sea único
        if not self.df.index.is_unique:
            self.df = self.df.reset_index(drop=True)
            
        datos = self.df.copy()
        # Crear mascara boleanda con valor True para toda la serie
        mask = pd.Series([True]*len(datos))

        # Validacion del parametro desde
        if desde:
            desde = pd.to_datetime(desde)
            mask &= (datos[columna_fecha]>=desde)

        # Validacion del parametro hasta_fecha_max
        if hasta_fecha_max and hasta is None:
            hasta = datos[columna_fecha].max()
            
        # Validacion del parametros hasta
        if hasta:
            hasta = pd.to_datetime(hasta)
            mask &= (datos[columna_fecha]<=hasta)
        
        # aplicacion del filtro
        resultado = datos[mask]
        self.df = resultado
        return self.df
    
    def stock_terceros(self):
        """
        stock_terceros: Metodo especilizados para generar el inventario de terceros. 
        filtra el reporte comercial sacando unicamente la facturacion 
        """
        datos = self.df.copy()

        # Filtrar Matriculas
        mask_matriculas = datos['Tipo_Registro']=='Matriculas'
        tabla_matriculas = datos[mask_matriculas]
        lista_vines_matriculos = tabla_matriculas['Chasis'].unique()

        # Filtrar Facturacion
        mask = (datos['Tipo_Registro']=='Facturacion') & (datos['Fecha']>'2021/1/1') 
        tabla = datos[mask]
        tabla.sort_values(by=['Fecha'],ascending=True)

        # Funcion que retorna el ultimo registro
        def ultimo_registro(serie):
            return serie.iloc[-1]
        
        valores = {
            'Color':ultimo_registro,
            'Grupo':ultimo_registro,
            'Familia':ultimo_registro,
            'Modelo':ultimo_registro,
            'Marca':ultimo_registro,
            'Gamas':ultimo_registro,
            'Segmento':ultimo_registro,
            'Fecha':ultimo_registro,
            'Vitrina':ultimo_registro,
            'Cantidad':'sum',
            'Subtotal':ultimo_registro
        }
        lista_columnas = [
            'Color',
            'Grupo',
            'Familia',
            'Modelo',
            'Marca',
            'Gamas',
            'Segmento',
            'Fecha',
            'Vitrina',
            'Cantidad',
            'Subtotal'
        ]   
        # Generacion de Tabla Dinamica
        pivote = pd.pivot_table(
            data=tabla,
            index='Chasis',
            values=lista_columnas,
            aggfunc=valores
        ).reset_index()
        mask_grupo = (pivote['Vitrina']!='VENTAS CORPORATIVAS') & (pivote['Grupo']=='WHOLE TERCEROS') & (pivote['Cantidad']!=0) & (~pivote['Chasis'].isin(lista_vines_matriculos) & (pivote['Modelo']>=2015))
        tabla_terceros = pivote[mask_grupo]
        self.df = tabla_terceros
        return self.df   

    def limpiar_campo(self,nombre_columna):
        def depurar_campos(valor):
            resultado = valor.split('.')[0]
            return resultado
        datos = self.df.copy() 
        datos[nombre_columna] = datos[nombre_columna].astype(str)
        datos[nombre_columna] = datos[nombre_columna].apply(lambda x: depurar_campos(x))
        self.df = datos
        return self.df
    
    def combinar_columnas(self, nuevas_columnas, nombre_nueva_columna, separador=''):
        self.df[nombre_nueva_columna] = self.df[nuevas_columnas].apply(
            lambda row: separador.join(
                [str(val) for val in row.values if pd.notnull(val) and val != '']
            ),
            axis=1
        )
        return self.df

from services.ETL.extract import Extract
import openpyxl

ruta = r'C:\Users\bi\Autocom S.A\Financiero KIA - Contenedor - Financiero KIA\Q13'
prueba = Extract(path=ruta)
prueba.read_file_excel(sheet_name='Q13_6.2 Ecuador',header=0,name_file='Estructura BI Ecuador (Original).xlsx')
print(prueba.df)


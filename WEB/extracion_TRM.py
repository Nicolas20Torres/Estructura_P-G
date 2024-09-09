"""
------------------------------------------------------------------------------------------
* WEB SCRAPING PAGINA BANCO DE LA REPUBLICA
-------------------------------------------- 
-- Creado por: @Nicolas Torres
-- Cargo: Business Inteligence Enginer
-- Contacto: 3213107305
--------------------------------------------
"""
import time
import os
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import shutil
# Otros paquetes locales
from config.parametrizaciones import Driver

class DownLoadTRM:
    """
    Inicializa la clase DownLoadTRM
    Parametros: 
     - url (str): cadena con el acceso al enlace de la pagina de TRM
     - download_path (str): cadena con la ruta donde se guadar los reportes descargados 
    """
    def __init__(self, url, download_path):
        self.url = url
        self.download_path = download_path
        self.options = Options()
        self.options.add_argument('--start-maximized')
        self.options.binary_location = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
        self.options.add_experimental_option("prefs", {"download.default_directory": download_path, "download.prompt_for_download": False})
        self.service_path = Driver.web_driver
        self.driver = self.create_driver()

    def create_driver(self):
        service = ChromeService(executable_path=self.service_path)
        return webdriver.Chrome(service=service, options=self.options)
    
    def open_window(self):
        self.driver.get(self.url)
        time.sleep(5)
    
    def descargar_TRM(self):
        # No aceptar las Cokies en el momento de ingresar a la pagina
        Element_Button = "//button[text()='No aceptar y continuar']"
        Button = WebDriverWait(self.driver,10).until(
            EC.element_to_be_clickable((By.XPATH,Element_Button))
        )
        Button.click()
        time.sleep(2)
        # Dar click en el elemento que direcciona descagar del archivo
        url = "//a[text()='Serie histórica mensual promedio y fin de mes (desde 27/11/1991)']"
        TRM = WebDriverWait(self.driver,10).until(
            EC.element_to_be_clickable((By.XPATH,url))
        )
        TRM.click()
        time.sleep(10)
        
    def generar_tabla(self):
        tabla = WebDriverWait(self.driver,20).until(
            EC.presence_of_element_located((By.CLASS_NAME,"PTChildPivotTable"))
        )
        rows = tabla.find_elements(By.TAG_NAME,'tr')
        data = []
        for row in rows[4:]:
            cols = row.find_elements(By.TAG_NAME,'td')
            cols = [col.text.strip() for col in cols if col.text.strip() !='']
            print(cols)
            if len(cols) == 3:
                data.append(cols)

        column_names = ['Año - Mes (aaaa -mm)', 'Promedio mensual', 'Fin de mes']
        if data:
            df = pd.DataFrame(data,columns=column_names)
            df['TipoTRM'] = 'Real'
            df.to_csv(r'C:\Users\bi\Autocom S.A\Financiero KIA - Contenedor - Financiero KIA\TRM\TRM_Banco_Republica.txt',encoding='utf-8',sep='|',index=False)
        
        self.driver.quit()

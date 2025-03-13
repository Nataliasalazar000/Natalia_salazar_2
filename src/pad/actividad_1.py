import json
import requests
import sys
from pathlib import Path
import os


class Actividad_1():
    def __init__(self):
        self.ruta_static="src/pad/static/"
        self.ruta_actual = str(Path.cwd())
        self.ruta_static="{}/src/pad/static/".format(self.ruta_actual)
        self.ruta_json="{}/src/pad/static/json/".format(self.ruta_actual)
        directoario = os.path.dirname(self.ruta_json)
        if not os.path.exists(directoario):
            os.makedirs(directoario,exist_ok=True)

        sys.stdout.reconfigure(encoding='utf-8')

#Dos funciones,leer y escribr
    def leer_api(self, url):
            response = requests.get(url)
            return response.json()

    def escribir_json(self,nombre_archivo="",datos=None): # "" '' """ """
        if nombre_archivo=="":
            nombre_archivo="datos.json"
        if datos is None:
            datos = "No hay datos"
        ruta_json = "{}json/{}".format(self.ruta_static,nombre_archivo)
        try:
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(datos, f, ensure_ascii=False, indent=4)
            
        except Exception as e:
            print("Error:",e)

        return True # booleano True (1) False (0)

        #Creamos instancia de la clase
ingestion = Actividad_1()

datos_json = ingestion.leer_api("https://share.osf.io/api/v2/normalizeddata/")
print("datos json:",datos_json)
       


if ingestion.escribir_json(nombre_archivo="entrega_actividad_1.json",datos=datos_json):
    print("Se crea el archivo json.")

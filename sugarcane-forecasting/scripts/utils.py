import os
import json

def ruta_completa():
    # Ruta donde quieres que empiece la "acotación" (la raíz del proyecto)
    ruta_base = '/Users/adharacavazos/codes/pythonProyects/sugarcane-productions/sugarcane-forecasting/scripts/'

    # Ruta completa al archivo o carpeta
    ruta_completa = '/Users/adharacavazos/codes/pythonProyects/sugarcane-productions/sugarcane-forecasting/scripts/data/Mills No Weather'

    # Usar os.path.relpath para obtener la ruta relativa desde la carpeta base
    ruta_relativa = os.path.relpath(ruta_completa, ruta_base)
    return ruta_relativa

# Ruta al JSON y a la carpeta que contiene los archivos actuales
ruta_json = 'mills_index.json'
csv_path = ruta_completa()

# Cargar datos del JSON
with open(ruta_json, 'r', encoding='utf-8') as f:
    datos = json.load(f)

archivos_actuales = os.listdir(csv_path) # Listar todos los archivos CSV en la carpeta

# Recorrer cada entrada del JSON
for sm_id, info in datos.items():
    nombre_humano = f"{info['name']}.csv"
    nombre_humano_normalizado = nombre_humano.strip().lower()
    nombre_nuevo = f"{sm_id}.csv"

    # Buscar el archivo real haciendo comparación insensible a mayúsculas
    archivo_real = None
    for archivo in archivos_actuales:
        if archivo.strip().lower() == nombre_humano_normalizado:
            archivo_real = archivo
            break

    if archivo_real:
        ruta_actual = os.path.join(csv_path, archivo_real)
        ruta_nueva = os.path.join(csv_path, nombre_nuevo)

        os.rename(ruta_actual, ruta_nueva)
        print(f"✅ Renombrado: {archivo_real} → {nombre_nuevo}")
    else:
        print(f"❌ No encontrado en carpeta: {nombre_humano}")

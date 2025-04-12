import os

# Ruta de la carpeta (puede ser relativa o absoluta)
ruta_carpeta = 'data/Mills No Weather'

ds_store_path = os.path.join(ruta_carpeta, '.DS_Store')

if os.path.exists(ds_store_path):
    os.remove(ds_store_path)
    print(f"Se eliminó {ds_store_path}")
else:
    print(f"No se encontró {ds_store_path}")

# Obtener lista de archivos y directorios
contenido = os.listdir(ruta_carpeta)

# Filtrar solo archivos (opcional)
archivos = [nombre for nombre in contenido if os.path.isfile(os.path.join(ruta_carpeta, nombre))]

print(archivos)

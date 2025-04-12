import os
import json

folder_path = 'data/Mills With Weather' # Mills No Weather
json_path = 'scripts/mills_index.json'

"""  Delete files
ds_store_path = os.path.join(folder_path, '.DS_Store')

if os.path.exists(ds_store_path):
    os.remove(ds_store_path)
    print(f"Se eliminó {ds_store_path}")
else:
    print(f"No se encontró {ds_store_path}")
"""
info = os.listdir(folder_path)

current_files = [name_file for name_file in info if os.path.isfile(os.path.join(folder_path, name_file))]

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for ID, info in data.items():
    complete_mill_name = f"{info['name']}.csv"
    id_mill_name = complete_mill_name.strip().lower()
    new_mill_name = f"{ID}.csv"

    real_file = None
    for files in current_files:
        if files.strip().lower() == id_mill_name:
            real_file = files
            break

    if real_file:
        current_path = os.path.join(folder_path, real_file)
        new_path = os.path.join(folder_path, new_mill_name)

        os.rename(current_path, new_path)
        print(f"✅ Renamed File : {real_file} → {new_mill_name}")
    else:
        print(f"❌ File Not Found: {complete_mill_name}")
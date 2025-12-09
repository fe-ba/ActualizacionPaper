# kaggle_download.py
import os
import sys
import shutil
import zipfile # Necesario para la descompresión manual
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_lfw(dataset_name='jessicali9530/lfw-dataset', path='lfw_dataset'):
    """
    Descarga y extrae el dataset LFW de Kaggle, manejando la descompresión
    manualmente para evitar fallos de la API.
    """
    print(f"Iniciando descarga del dataset: {dataset_name}...")
    
    # 1. Limpiar directorio existente
    if os.path.exists(path):
        print(f"Limpiando directorio existente: {path}...")
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error al intentar limpiar el directorio: {e}")
            sys.exit(1)
            
    if not os.path.exists(path):
        os.makedirs(path)
    
    zip_filename = "lfw-dataset.zip" # Nombre de archivo esperado del dataset
    zip_filepath = os.path.join(path, zip_filename)

    try:
        api = KaggleApi() 
        api.authenticate()

        print("Autenticación exitosa. Descargando archivos...")
        
        # 2. Descargar solo el ZIP (unzip=False)
        api.dataset_download_files(dataset_name, path=path, unzip=False)
        
        print("Descarga completada. Iniciando descompresión manual...")
        
        # 3. Descomprimir el archivo manualmente
        if not os.path.exists(zip_filepath):
            # A veces el nombre del archivo cambia. Verificamos el contenido del path.
            downloaded_files = os.listdir(path)
            zip_files = [f for f in downloaded_files if f.endswith('.zip')]
            
            if not zip_files:
                raise FileNotFoundError(f"No se encontró ningún archivo ZIP en {path}.")
            
            zip_filepath = os.path.join(path, zip_files[0]) # Usamos el primer ZIP encontrado
        
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            # Extraer todo en la misma carpeta 'lfw_dataset'
            zip_ref.extractall(path)

        # 4. Limpiar el archivo ZIP después de la extracción exitosa
        os.remove(zip_filepath)

        # La estructura final debería ser lfw_dataset/lfw
        if os.path.exists(os.path.join(path, "lfw")):
             print(f"¡ÉXITO! Dataset LFW extraído correctamente en: {path}/lfw")
        else:
             print("ADVERTENCIA: Descompresión completada, pero la estructura de carpetas es inusual. Revise el directorio 'lfw_dataset'.")

    except Exception as e:
        print("\n==================================================================================")
        print("ERROR: La descarga o descompresión falló.")
        print(f"Detalles del error: {e}")
        print("==================================================================================")
        sys.exit(1) 

if __name__ == "__main__":
    download_and_extract_lfw()

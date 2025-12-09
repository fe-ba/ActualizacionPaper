# Sistema de Control de Acceso Basado en Reconocimiento Facial (ArcFace)

Este repositorio contiene la implementación del sistema de control de acceso facial utilizando técnicas de Aprendizaje Profundo.Emplea **RetinaFace** para la detección de rostros y **ArcFace** para la extracción de características, basado en el framework **InsightFace**.

El sistema proporciona una solución robusta y precisa, alcanzando una precisión reportada del **94.90%** y un tiempo de procesamiento de **0.1029 segundos por par de imágenes**.

## Configuración e Instalación

Sigue estos pasos para configurar el entorno de desarrollo y obtener los datos necesarios.

### 1. Entorno Virtual y Dependencias

Es crucial aislar las dependencias del proyecto utilizando un entorno virtual.

| Sistema Operativo | Comando de Activación |
| :--- | :--- |
| **Linux/macOS** | `source .venv/bin/activate` |
| **Windows** | `.venv\Scripts\activate` |

**Pasos Comunes:**

1.  **Crear el entorno virtual:**
    ```bash
    python -m venv .venv
    ```
2.  **Activar el entorno virtual:** (Usar el comando de la tabla superior)
3.  **Instalar dependencias:** Instala todas las librerías necesarias utilizando el archivo `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### 2. Descarga del Dataset LFW (Requerimientos de Kaggle API)

El dataset **LFW (Labeled Faces in the Wild)**, necesario para el cálculo de métricas (Opción 3), no se incluye en el repositorio por razones de tamaño. Su descarga se automatiza mediante la API de Kaggle.

1.  **Generar `kaggle.json`**:
    * Ve a tu cuenta de Kaggle y en la sección **Account**, haz clic en **Create New API Token**.
    * Bajar y pulsar el Create Legacy API token, para generar un `kaggle.json`.

2.  **Colocar `kaggle.json`**:
    * Mueve el archivo `kaggle.json` dentro de la carpeta raíz de este proyecto (`ActualizacionPaper`).

3.  **Ejecutar el script de descarga**:
    * Ejecuta el script `kaggle_download.py` para descargar y configurar el dataset.

    ```bash
    python kaggle_download.py
    ```
    *(**Nota sobre Rutas**): La ruta de los archivos en `arcface_implementation.py` ya está configurada con la doble anidación (`lfw_dataset/lfw-deepfunneled/lfw-deepfunneled`) para funcionar correctamente en Linux y Windows, en caso de no funcionar en Windows, solo modifcar la ruta.*

## Uso del Sistema (`access_control_cli.py`)

El sistema se opera a través de la interfaz de línea de comandos.

```bash
python access_control_cli.py

# Wildlife Detection and Counting System

Este proyecto implementa un sistema de detección y conteo de mamíferos africanos en imágenes aéreas. Incluye un pipeline para entrenamiento, evaluación, minería de negativos, ajuste fino y despliegue mediante Gradio y Docker.

## Características principales
- Entrenamiento incremental de HerdNet con hard negative mining
- Entrenamiento de fine-tuning de HerdNet oficial
- Entrenamiento completo de YOLOv11m sobre ULiège-AIR
- Métricas: F1, precisión, recall, MAE y RMSE
- Interfaz con Gradio
- Despliegue portable con Docker
- Hosting en Hugging Face Spaces

## Dependencias
- Python 3.10+
- PyTorch 2.x
- Ultralytics YOLOv8/11
- Albumentations
- OpenCV
- NumPy
- Pandas
- Gradio
- tqdm
- Matplotlib
- Docker

## Instalación
```
pip install -r requirements.txt
```

## Entorno de ejecución probado
- Google Colab con GPU T4
- Hugging Face Spaces
- Windows 11 con WSL2
- Docker con imagen base de PyTorch

## Uso básico
Iniciar la aplicación local:
```
python app.py
```

## Despliegue con Docker
```
docker build -t wildlife-detector .
docker run -p 7860:7860 wildlife-detector
```

## Despliegue en Hugging Face
1. Subir Dockerfile y app.py al repositorio del Space
2. Seleccionar Docker como runtime
3. Activar GPU si se requiere
4. El Space ejecutará la aplicación de forma automática

## Enlaces relevantes
- Dataset ULiège-AIR: https://github.com/uliege-air/dataset
- HerdNet (Delplanque et al., 2023)
- Documentación YOLOv11: https://docs.ultralytics.com

## Archivos clave
- app.py: punto de entrada de la aplicación
- Dockerfile: define la imagen base y dependencias
- requirements.txt: especifica dependencias
- resources/models/: modelos HerdNet en formato .pth

## Requisitos de entorno
- Docker Desktop o Docker con NVIDIA Container Toolkit
- GPU con soporte CUDA (opcional)
- Acceso a internet para el primer build

## Construcción de imagen Docker
Ubicarse en la raíz del proyecto y ejecutar:
```
docker build -t herdnet-app .
```

## Ejecución de la aplicación
Con GPU:
```
docker run --gpus all -p 7860:7860 herdnet-app
```
En CPU:
```
docker run -p 7860:7860 herdnet-app
```
Abrir en el navegador:
http://localhost:7860

## Artefactos
Disponible en:
https://drive.google.com/open?id=1oD3-ZtvEfPJtfDrBbefJ2JLMIWksVBK6&usp=drive_fs

## Créditos
Universidad de los Andes - Maestría en Inteligencia Artificial  
Grupo Proyecto Guacamaya (CINFONIA)

# Wildlife Detection and Counting System

Este proyecto implementa un sistema de detección y conteo de mamíferos africanos en imágenes aéreas usando dos enfoques: detección con YOLOv11m y estimación por mapas de densidad con HerdNet. Incluye un pipeline reproducible para entrenamiento, evaluación, minería de negativos, ajuste fino y despliegue mediante Gradio y Docker.

## Características principales
- Entrenamiento incremental de HerdNet con hard negative mining
- Entrenamiento completo de YOLOv11m sobre ULiège-AIR
- Métricas: F1, precisión, recall, MAE y RMSE
- Inferencia con tiling, NMS global y Test-Time Augmentation
- Interfaz con Gradio
- Despliegue portable con Docker
- Hosting en Hugging Face Spaces

## Estructura del proyecto
```
project/
├── resources/
│   ├── configs/
│   ├── weights/
│   └── figures/
├── animaloc/
├── inference/
├── training/
├── app.py
├── Dockerfile
└── README.md
```

## Dependencias
- Python 3.10+
- PyTorch 2.x
- Ultralytics (YOLOv8/11)
- Albumentations
- OpenCV
- NumPy
- Pandas
- Gradio
- tqdm
- Matplotlib
- Docker (opcional)

### Instalación
```bash
pip install -r requirements.txt
```

## Entorno de ejecución
Probado en:
- Google Colab (GPU T4)
- Hugging Face Spaces
- Windows 11 + WSL2
- Docker + PyTorch base image

## Uso básico
Iniciar la aplicación local:
```bash
python app.py
```

## Despliegue con Docker
```bash
docker build -t wildlife-detector .
docker run -p 7860:7860 wildlife-detector
```

## Despliegue en Hugging Face
1. Subir Dockerfile y app.py al repositório del Space
2. Seleccionar “Docker” como runtime
3. Activar GPU si es necesario
4. El Space ejecutará automáticamente la app Gradio

## Enlaces relevantes
- Dataset ULiège-AIR: https://github.com/uliege-air/dataset
- HerdNet (Delplanque et al., 2023)
- Documentación YOLOv11: https://docs.ultralytics.com

# Wildlife Detection and Counting System

Sistema para detección y conteo de mamíferos africanos en imágenes aéreas. El proyecto incluye un pipeline reproducible para entrenamiento, evaluación, minería de negativos, ajuste fino de HerdNet y comparación con YOLOv11m. También integra una aplicación interactiva con Gradio y un despliegue portable mediante Docker y Hugging Face Spaces.

## Características principales
- Entrenamiento progresivo de HerdNet con fases base, hard negative mining y fine-tuning.
- Ajuste fino del modelo oficial de HerdNet publicado por Delplanque et al.
- Entrenamiento completo de YOLOv11m sobre el dataset ULiège-AIR.
- Métricas estandarizadas: F1, precisión, recall, MAE y RMSE.
- Pipeline de inferencia con sliding window, NMS global y centroid matching.
- Interfaz interactiva con Gradio.
- Contenedorización con Docker.
- Hosting en Hugging Face Spaces con opción GPU.

## Enlaces del proyecto
- **Repositorio HerdNet original:** https://github.com/Alexandre-Delplanque/HerdNet/tree/main
- **Aplicación desplegada:** https://huggingface.co/spaces/jaimevera1107/herdnet-app
- **Dataset ULiège-AIR:** https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5

## Ejemplo visual
![Ejemplo de imagen aérea](https://github.com/jaimevera1107/aerial-wildlife-count/blob/main/datos/image_2025-11-21_120159287.png?raw=true)

## Estructura del repositorio
```
datos/          # Muestras del dataset
inference/      # Scripts de inferencia y postprocesamiento
modelos/        # Modelos finales y checkpoints
notebooks/      # Entrenamiento y experimentos
resources/      # Utilidades y configuraciones
app.py          # App de inferencia con Gradio
Dockerfile      # Imagen de despliegue
requirements.txt# Dependencias
README.md       # Documentación
```

## Dependencias
- Python 3.10+
- PyTorch 2.x
- Ultralytics YOLOv8/11
- Albumentations
- OpenCV
- NumPy
- Pandas
- Gradio
- Matplotlib
- tqdm
- Docker

Instalación:
```
pip install -r requirements.txt
```

## Entorno probado
- Google Colab con GPU T4
- Hugging Face Spaces con Docker
- Windows 11 + WSL2
- Docker Desktop con NVIDIA Container Toolkit

## Ejecutar aplicación local
```
python app.py
```
Abrir:
```
http://localhost:7860
```

Salida:
- Imagen anotada
- Conteo global y por especie
- JSON con predicciones

## Docker
### Construcción
```
docker build -t wildlife-detector .
```

### Ejecución con GPU
```
docker run --gpus all -p 7860:7860 wildlife-detector
```

### CPU
```
docker run -p 7860:7860 wildlife-detector
```

## Hugging Face Spaces
1. Crear Space en modo Docker.
2. Subir `Dockerfile`, `app.py`, `requirements.txt`, `resources/`, `modelos/`.
3. Activar GPU si se necesita.
4. La app se ejecuta automáticamente.

## Artefactos
https://drive.google.com/drive/folders/1oD3-ZtvEfPJtfDrBbefJ2JLMIWksVBK6

## Créditos
Proyecto desarrollado en la  
**Maestría en Inteligencia Artificial – Universidad de los Andes**  
Grupo **Proyecto Guacamaya (CINFONIA)**

Contribuidores:
- Jaime A. Vera
- Julián F. Cujabante
- Rafael A. Ortega
- Uldy D. Paloma Rozo

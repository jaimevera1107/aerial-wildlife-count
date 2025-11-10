# HerdNet App - Detección y Conteo de Mamíferos Africanos

Aplicación basada en PyTorch y Gradio para la detección y conteo automatizado de mamíferos africanos a partir de imágenes aéreas.  
La imagen de Docker incluye todos los componentes necesarios: modelo, dependencias y configuración CUDA.

Archivos clave:
- app.py: punto de entrada de la aplicación (interfaz Gradio o API)
- Dockerfile: define la imagen base (PyTorch + CUDA + dependencias)
- requirements.txt: dependencias reproducibles
- resources/models/: modelo preentrenado HerdNet (.pth)

## Requisitos

- Docker Desktop (Windows/macOS) o Docker con NVIDIA Container Toolkit (Linux)
- GPU con soporte CUDA (opcional, la app también corre en CPU)
- Acceso a internet para el primer build

## Construir la imagen

Ubicarse en la raíz del proyecto (donde está el Dockerfile) y ejecutar:

docker build -t herdnet-app .

El primer build puede tardar varios minutos (descarga aproximada de 7 GB).  
Docker usará caché en builds posteriores.

## Ejecutar la aplicación

Con GPU (si disponible):

```
docker run --gpus all -p 7860:7860 herdnet-app
```

En CPU:
```
docker run -p 7860:7860 herdnet-app
```

Luego abrir en el navegador:  
http://localhost:7860

## Artefactos
Se puede encontrar en el enlace compartido de Google Drive:
https://drive.google.com/open?id=1oD3-ZtvEfPJtfDrBbefJ2JLMIWksVBK6&usp=drive_fs

## Créditos

Universidad de los Andes - Maestría en Inteligencia Artificial  
Grupo Proyecto Guacamaya (CINFONIA)

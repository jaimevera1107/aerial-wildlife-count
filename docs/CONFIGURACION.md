# 🔧 Guía de Configuración - Wildlife Vision

Esta guía detalla todas las opciones de configuración disponibles para el sistema Wildlife Vision.

## 📋 Tabla de Contenidos

- [Variables de Entorno](#variables-de-entorno)
- [Archivo de Configuración YAML](#archivo-de-configuración-yaml)
- [Configuración Docker](#configuración-docker)
- [Configuración de GPU](#configuración-de-gpu)
- [Configuración de DVC](#configuración-de-dvc)

---

## Variables de Entorno

Las siguientes variables de entorno pueden ser configuradas para personalizar el comportamiento del sistema:

### Servidor Gradio

| Variable | Descripción | Valor por Defecto | Ejemplo |
|----------|-------------|-------------------|---------|
| `GRADIO_SERVER_NAME` | Dirección IP del servidor | `0.0.0.0` | `127.0.0.1` |
| `GRADIO_SERVER_PORT` | Puerto del servidor | `7860` | `8080` |

### GPU y CUDA

| Variable | Descripción | Valor por Defecto | Ejemplo |
|----------|-------------|-------------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPUs a utilizar | `0` | `0,1` |
| `TORCH_CUDA_ARCH_LIST` | Arquitecturas CUDA | Auto-detectado | `7.5;8.0` |

### Matplotlib y Visualización

| Variable | Descripción | Valor por Defecto | Ejemplo |
|----------|-------------|-------------------|---------|
| `MPLCONFIGDIR` | Directorio de caché Matplotlib | `/tmp/matplotlib` | `/var/cache/mpl` |

### Logging

| Variable | Descripción | Valor por Defecto | Ejemplo |
|----------|-------------|-------------------|---------|
| `LOG_LEVEL` | Nivel de logging | `INFO` | `DEBUG` |

---

## Archivo de Configuración YAML

El archivo principal de configuración se encuentra en `resources/configs/default.yaml`.

### Estructura Completa

```yaml
# ===========================================
# Configuración del Modelo HerdNet
# ===========================================
model:
  # Nombre identificador del modelo
  name: "herdnet_fase1_best"
  
  # Ruta al archivo del modelo (.pth)
  path: "resources/models/herdnet_best.pth"
  
  # Dispositivo de inferencia: "cuda" o "cpu"
  device: "cuda"
  
  # Tamaño del parche en píxeles (debe ser múltiplo de 32)
  patch_size: 512
  
  # Solapamiento entre parches en píxeles
  overlap: 160
  
  # Ratio de reducción del mapa de densidad
  down_ratio: 2

# ===========================================
# Rutas de Directorios
# ===========================================
paths:
  # Directorio para imágenes subidas temporalmente
  uploads_dir: "resources/uploads"
  
  # Directorio para resultados de inferencia
  outputs_dir: "resources/outputs"
  
  # Directorio para archivos de log
  logs_dir: "resources/logs"

# ===========================================
# Opciones de Inferencia
# ===========================================
inference:
  # Guardar visualizaciones de detecciones
  save_plots: true
  
  # Guardar resultados en formato CSV
  save_csv: true
  
  # Guardar miniaturas de cada detección
  save_thumbnails: false
  
  # Mostrar logs detallados durante la inferencia
  verbose: true
```

### Parámetros Críticos

#### `patch_size`
- **Descripción**: Tamaño de los parches para sliding window
- **Valores recomendados**: 256, 512, 1024
- **Impacto**: Mayor tamaño = más memoria GPU, menos parches
- **Nota**: Debe ser múltiplo de 32

#### `overlap`
- **Descripción**: Solapamiento entre parches adyacentes
- **Valores recomendados**: 64-256 (25-50% del patch_size)
- **Impacto**: Mayor overlap = mejor detección en bordes, más tiempo de procesamiento

#### `down_ratio`
- **Descripción**: Factor de reducción del mapa de densidad
- **Valores válidos**: 1, 2, 4
- **Impacto**: Mayor ratio = menor resolución del mapa, más rápido

---

## Configuración Docker

### Dockerfile Principal (CUDA)

```dockerfile
# Variables de entorno predefinidas
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV MPLCONFIGDIR="/tmp/matplotlib"
```

### Personalización en Runtime

```bash
# Cambiar puerto
docker run -p 8080:7860 -e GRADIO_SERVER_PORT=7860 wildlife-detector

# Usar CPU en lugar de GPU
docker run -p 7860:7860 -e MODEL_DEVICE=cpu wildlife-detector

# Especificar GPU
docker run --gpus '"device=1"' -p 7860:7860 wildlife-detector
```

### Docker Compose con Variables

```yaml
version: '3.8'
services:
  wildlife-vision:
    build: .
    ports:
      - "${HOST_PORT:-7860}:7860"
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
      - CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Configuración de GPU

### Verificar Disponibilidad de GPU

```python
import torch

# Verificar CUDA
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión CUDA: {torch.version.cuda}")
print(f"Número de GPUs: {torch.cuda.device_count()}")

# Listar GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memoria: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
```

### Seleccionar GPU Específica

```bash
# Usar GPU 0
CUDA_VISIBLE_DEVICES=0 python app.py

# Usar GPU 1
CUDA_VISIBLE_DEVICES=1 python app.py

# Usar múltiples GPUs
CUDA_VISIBLE_DEVICES=0,1 python app.py

# Forzar CPU
CUDA_VISIBLE_DEVICES="" python app.py
```

### Configuración en el Archivo YAML

Para forzar el uso de CPU, modificar `resources/configs/default.yaml`:

```yaml
model:
  device: "cpu"  # Cambiar de "cuda" a "cpu"
```

---

## Ejemplos de Configuración

### Configuración para Desarrollo (Local)

```yaml
model:
  device: "cpu"
  patch_size: 256
  overlap: 64

inference:
  save_plots: true
  save_csv: true
  verbose: true
```

### Configuración para Producción (GPU)

```yaml
model:
  device: "cuda"
  patch_size: 512
  overlap: 160

inference:
  save_plots: false
  save_csv: true
  verbose: false
```

### Configuración para Alta Resolución

```yaml
model:
  device: "cuda"
  patch_size: 1024
  overlap: 256

inference:
  save_plots: true
  save_csv: true
  save_thumbnails: true
```

---

## Solución de Problemas

### Error: "CUDA out of memory"

**Solución**: Reducir `patch_size` en el archivo de configuración:

```yaml
model:
  patch_size: 256  # Reducir de 512 a 256
```

### Error: "No CUDA GPUs are available"

**Solución**: Cambiar a CPU:

```yaml
model:
  device: "cpu"
```

### Error: "Model file not found"

**Solución**: Verificar la ruta del modelo:

```bash
# Verificar que el modelo existe
ls -la resources/models/herdnet_best.pth

# Si no existe, descargar con DVC
dvc pull
```

---

---

## Configuración de DVC

Este proyecto utiliza **DVC (Data Version Control)** para gestionar archivos grandes como datos y modelos.

### Remote Configurado

```bash
# Ver configuración del remote
dvc remote list

# Resultado:
# storage    ssh://dvc@rinconseguro.com:33/share/DVC
```

### Archivos Versionados con DVC

| Archivo/Carpeta | Descripción | Tamaño Aprox. |
|-----------------|-------------|---------------|
| `data/` | Dataset completo | ~33 GB |
| `modelos/herdnet_best.pth` | Modelo entrenado | ~200 MB |
| `resources/models/herdnet_best.pth` | Copia del modelo | ~200 MB |

### Configuración de Credenciales SSH

```bash
# Opción 1: Autenticación por llave SSH (recomendado)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_dvc
ssh-copy-id -i ~/.ssh/id_rsa_dvc -p 33 dvc@rinconseguro.com

# Configurar SSH
cat >> ~/.ssh/config << EOF
Host rinconseguro.com
    IdentityFile ~/.ssh/id_rsa_dvc
    Port 33
    User dvc
EOF

# Opción 2: Contraseña (se pedirá interactivamente)
# Simplemente ejecutar dvc pull
```

### Comandos DVC Útiles

```bash
# Ver estado de archivos DVC
dvc status

# Descargar todos los archivos
dvc pull

# Descargar solo el modelo
dvc pull modelos/herdnet_best.pth.dvc

# Subir cambios al remote
dvc push

# Ver diferencias
dvc diff

# Reproducir pipeline (si está configurado)
dvc repro
```

### Variables de Entorno para DVC

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `DVC_REMOTE` | Remote por defecto | `storage` |
| `DVC_REMOTE_STORAGE_PASSWORD` | Contraseña SSH (no recomendado) | `***` |

---

## Contacto

Para preguntas sobre configuración, contactar al equipo de desarrollo:

- **Email**: proyecto-guacamaya@uniandes.edu.co
- **GitHub Issues**: [Abrir issue](https://github.com/jaimevera1107/aerial-wildlife-count/issues)


# 🚀 Manual de Despliegue - Wildlife Vision

Este documento proporciona instrucciones detalladas para desplegar la aplicación Wildlife Vision en diferentes entornos.

## 📋 Tabla de Contenidos

- [Requisitos Previos](#requisitos-previos)
- [Opción 1: Despliegue Local](#opción-1-despliegue-local)
- [Opción 2: Despliegue con Docker](#opción-2-despliegue-con-docker)
- [Opción 3: Despliegue en Hugging Face Spaces](#opción-3-despliegue-en-hugging-face-spaces)
- [Opción 4: Despliegue en Servidor Cloud](#opción-4-despliegue-en-servidor-cloud)
- [Verificación del Despliegue](#verificación-del-despliegue)
- [Monitoreo y Logs](#monitoreo-y-logs)
- [Solución de Problemas](#solución-de-problemas)

---

## Requisitos Previos

> ⚠️ **IMPORTANTE: Este proyecto usa DVC (Data Version Control)**
> 
> Los datos y modelos grandes están versionados con DVC y almacenados en un servidor remoto SSH.
> Antes de ejecutar la aplicación, necesitas descargar estos archivos con `dvc pull`.
> Consulta la sección [Configurar DVC](#paso-4-configurar-dvc-y-descargar-datosmodelos) para más detalles.

### Hardware Mínimo

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 8 GB | 16 GB |
| Almacenamiento | 10 GB | 20 GB |
| GPU | Opcional | NVIDIA con 4+ GB VRAM |

### Software Requerido

| Software | Versión | Notas |
|----------|---------|-------|
| Python | 3.10+ | Requerido para ejecución local |
| Docker | 20.10+ | Para despliegue containerizado |
| Git | 2.30+ | Para clonar el repositorio |
| NVIDIA Driver | 525+ | Solo si usa GPU |
| CUDA | 12.1+ | Solo si usa GPU |

---

## Opción 1: Despliegue Local

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/jaimevera1107/aerial-wildlife-count.git
cd aerial-wildlife-count
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/macOS:
source venv/bin/activate

# En Windows:
.\venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 4: Configurar DVC y Descargar Datos/Modelos

Este proyecto utiliza **DVC (Data Version Control)** para gestionar los datos y modelos grandes. Los archivos versionados con DVC incluyen:

- `data/` - Dataset completo (~33 GB, 1.3M archivos)
- `modelos/herdnet_best.pth` - Modelo entrenado
- `resources/models/herdnet_best.pth` - Copia del modelo

#### 4.1 Instalar DVC

```bash
pip install dvc
```

#### 4.2 Configurar Credenciales SSH

El remote de DVC está configurado en un servidor SSH. Necesitas configurar el acceso:

```bash
# Verificar configuración del remote
dvc remote list

# El remote está en:
# storage: ssh://dvc@rinconseguro.com:33/share/DVC
```

**Opciones de autenticación:**

```bash
# Opción 1: Usar contraseña (se pedirá interactivamente)
dvc pull

# Opción 2: Configurar llave SSH
ssh-keygen -t rsa -b 4096
ssh-copy-id -p 33 dvc@rinconseguro.com

# Opción 3: Configurar contraseña en DVC (no recomendado para producción)
dvc remote modify storage password TU_PASSWORD
```

#### 4.3 Descargar Datos y Modelos

```bash
# Descargar todos los archivos versionados con DVC
dvc pull

# Esto descargará:
# - data/ (dataset completo)
# - modelos/herdnet_best.pth
# - resources/models/herdnet_best.pth
```

#### 4.4 Verificar Descarga

```bash
# Verificar que los archivos se descargaron
ls -la data/
ls -la modelos/
ls -la resources/models/

# Verificar estado de DVC
dvc status
```

> ⚠️ **Nota**: La descarga del dataset completo puede tomar varios minutos dependiendo de la velocidad de conexión (~33 GB).

#### 4.5 Alternativa: Solo Descargar el Modelo

Si solo necesitas el modelo (sin el dataset completo):

```bash
# Descargar solo el modelo
dvc pull modelos/herdnet_best.pth.dvc
dvc pull resources/models/herdnet_best.pth.dvc
```

### Paso 5: Verificar Configuración

```bash
# Verificar que el modelo existe
ls -la resources/models/herdnet_best.pth

# Verificar configuración
cat resources/configs/default.yaml
```

### Paso 6: Ejecutar la Aplicación

```bash
# Ejecutar con configuración por defecto
python app.py

# O con variables de entorno personalizadas
GRADIO_SERVER_PORT=8080 python app.py
```

### Paso 7: Acceder a la Aplicación

Abrir navegador en: `http://localhost:7860`

---

## Opción 2: Despliegue con Docker

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/jaimevera1107/aerial-wildlife-count.git
cd aerial-wildlife-count
```

### Paso 2: Descargar Datos con DVC

Antes de construir la imagen Docker, asegúrate de descargar el modelo con DVC:

```bash
# Instalar DVC
pip install dvc

# Descargar el modelo (requerido para la imagen Docker)
dvc pull modelos/herdnet_best.pth.dvc
dvc pull resources/models/herdnet_best.pth.dvc

# Verificar que el modelo existe
ls -la resources/models/herdnet_best.pth
```

> 📝 **Nota**: El Dockerfile copia el modelo desde `resources/models/` al contenedor.
> Si el modelo no existe, la imagen se construirá pero la aplicación fallará al iniciar.

### Paso 3: Construir la Imagen Docker

#### Para sistemas x86_64 con GPU NVIDIA:

```bash
docker build -t wildlife-vision:latest .
```

#### Para Apple Silicon (ARM64):

```bash
docker build -f Dockerfile.arm64 -t wildlife-vision:latest .
```

### Paso 4: Ejecutar el Contenedor

#### Con GPU NVIDIA:

```bash
docker run -d \
  --name wildlife-vision \
  --gpus all \
  -p 7860:7860 \
  -v $(pwd)/resources/outputs:/app/resources/outputs \
  wildlife-vision:latest
```

#### Sin GPU (CPU):

```bash
docker run -d \
  --name wildlife-vision \
  -p 7860:7860 \
  -v $(pwd)/resources/outputs:/app/resources/outputs \
  wildlife-vision:latest
```

### Paso 5: Verificar que el Contenedor está Corriendo

```bash
# Ver logs
docker logs -f wildlife-vision

# Verificar estado
docker ps
```

### Paso 6: Acceder a la Aplicación

Abrir navegador en: `http://localhost:7860`

### Comandos Útiles de Docker

```bash
# Detener el contenedor
docker stop wildlife-vision

# Reiniciar el contenedor
docker restart wildlife-vision

# Eliminar el contenedor
docker rm -f wildlife-vision

# Ver logs en tiempo real
docker logs -f wildlife-vision

# Entrar al contenedor
docker exec -it wildlife-vision /bin/bash
```

---

## Opción 3: Despliegue en Hugging Face Spaces

### Paso 1: Crear Cuenta en Hugging Face

1. Ir a [https://huggingface.co/join](https://huggingface.co/join)
2. Crear una cuenta o iniciar sesión

### Paso 2: Crear un Nuevo Space

1. Ir a [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Configurar el Space:
   - **Owner**: Tu usuario u organización
   - **Space name**: `wildlife-vision` (o el nombre que prefieras)
   - **License**: CC BY-NC-SA 4.0
   - **SDK**: Docker
   - **Hardware**: CPU basic (o GPU si necesitas mayor velocidad)

### Paso 3: Clonar el Space Localmente

```bash
git clone https://huggingface.co/spaces/TU_USUARIO/wildlife-vision
cd wildlife-vision
```

### Paso 4: Copiar los Archivos del Proyecto

```bash
# Copiar archivos necesarios
cp -r /ruta/al/proyecto/aerial-wildlife-count/* .

# IMPORTANTE: Asegurarse de que el modelo está descargado con DVC
cd /ruta/al/proyecto/aerial-wildlife-count
dvc pull modelos/herdnet_best.pth.dvc
dvc pull resources/models/herdnet_best.pth.dvc

# Copiar el modelo al Space
cp modelos/herdnet_best.pth /ruta/al/space/modelos/
cp resources/models/herdnet_best.pth /ruta/al/space/resources/models/

# Asegurarse de que los siguientes archivos están presentes:
# - Dockerfile
# - app.py
# - requirements.txt
# - inference/
# - resources/ (incluyendo models/herdnet_best.pth)
# - modelos/ (incluyendo herdnet_best.pth)
```

> ⚠️ **Nota sobre Hugging Face y DVC**: Hugging Face Spaces no ejecuta `dvc pull` automáticamente.
> Debes subir el modelo directamente al Space o usar Hugging Face Hub para almacenar el modelo.

### Paso 5: Subir los Cambios

```bash
git add .
git commit -m "Initial deployment of Wildlife Vision"
git push
```

### Paso 6: Configurar Hardware (Opcional)

1. Ir a Settings del Space
2. En "Space Hardware", seleccionar:
   - **CPU basic**: Gratis, más lento
   - **CPU upgrade**: Mejor rendimiento
   - **T4 small/medium**: GPU para inferencia rápida

### Paso 7: Verificar el Despliegue

1. Ir a la URL del Space: `https://huggingface.co/spaces/TU_USUARIO/wildlife-vision`
2. Esperar a que el contenedor se construya (puede tomar 5-10 minutos)
3. Verificar que la aplicación carga correctamente

---

## Opción 4: Despliegue en Servidor Cloud

### AWS EC2

#### Paso 1: Crear Instancia EC2

1. Ir a AWS Console → EC2 → Launch Instance
2. Seleccionar:
   - **AMI**: Ubuntu 22.04 LTS
   - **Instance Type**: `t3.large` (CPU) o `g4dn.xlarge` (GPU)
   - **Storage**: 30 GB gp3
   - **Security Group**: Abrir puerto 7860

#### Paso 2: Conectar a la Instancia

```bash
ssh -i tu-key.pem ubuntu@IP_PUBLICA
```

#### Paso 3: Instalar Docker

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Reiniciar sesión
exit
# Reconectar SSH
```

#### Paso 4: Instalar NVIDIA Container Toolkit (solo GPU)

```bash
# Añadir repositorio NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Instalar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Paso 5: Desplegar la Aplicación

```bash
# Clonar repositorio
git clone https://github.com/jaimevera1107/aerial-wildlife-count.git
cd aerial-wildlife-count

# Construir y ejecutar
docker build -t wildlife-vision .
docker run -d --name wildlife-vision --gpus all -p 7860:7860 wildlife-vision
```

#### Paso 6: Configurar Dominio (Opcional)

```bash
# Instalar Nginx
sudo apt install nginx -y

# Configurar proxy reverso
sudo nano /etc/nginx/sites-available/wildlife-vision
```

Contenido del archivo:

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}
```

```bash
# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/wildlife-vision /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Instalar certificado SSL (opcional)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d tu-dominio.com
```

### Google Cloud Platform (GCP)

#### Paso 1: Crear VM en Compute Engine

```bash
gcloud compute instances create wildlife-vision \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server
```

#### Paso 2: Abrir Firewall

```bash
gcloud compute firewall-rules create allow-wildlife-vision \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:7860 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server
```

#### Paso 3: Conectar y Desplegar

```bash
gcloud compute ssh wildlife-vision --zone=us-central1-a
# Seguir pasos de Docker descritos anteriormente
```

---

## Verificación del Despliegue

### Verificación Básica

```bash
# Verificar que el servidor responde
curl -I http://localhost:7860

# Respuesta esperada:
# HTTP/1.1 200 OK
```

### Verificación de Salud

```bash
# Verificar estado de la aplicación
curl http://localhost:7860/api/health
```

### Verificación Funcional

1. Abrir la aplicación en el navegador
2. Verificar que la interfaz carga correctamente
3. Verificar que muestra "Modelo activo" ✅
4. Subir una imagen de prueba
5. Verificar que la detección funciona

### Lista de Verificación

| Verificación | Comando/Acción | Resultado Esperado |
|--------------|----------------|-------------------|
| Servidor responde | `curl -I localhost:7860` | HTTP 200 |
| Interfaz carga | Abrir navegador | UI visible |
| Modelo cargado | Ver indicador | "Modelo activo" |
| Detección funciona | Subir imagen | Detecciones visibles |
| CSV descargable | Click en botón | Archivo descarga |

---

## Monitoreo y Logs

### Logs de la Aplicación

```bash
# Docker: Ver logs en tiempo real
docker logs -f wildlife-vision

# Local: Los logs se guardan en
tail -f resources/logs/herdnet_infer_*.log
```

### Monitoreo de Recursos

```bash
# Uso de CPU y memoria del contenedor
docker stats wildlife-vision

# Uso de GPU (si aplica)
nvidia-smi -l 1
```

### Configurar Logging Persistente

```bash
# Docker con logging persistente
docker run -d \
  --name wildlife-vision \
  --log-driver json-file \
  --log-opt max-size=100m \
  --log-opt max-file=3 \
  -p 7860:7860 \
  wildlife-vision:latest
```

---

## Solución de Problemas

### Error: "Model file not found"

**Causa**: El archivo del modelo no existe en la ruta especificada.

**Solución**:
```bash
# Verificar que el modelo existe
ls -la resources/models/herdnet_best.pth

# Si usa DVC, descargar el modelo
dvc pull modelos/herdnet_best.pth.dvc
dvc pull resources/models/herdnet_best.pth.dvc

# Verificar estado de DVC
dvc status

# O descargar manualmente desde Google Drive
# Ver enlace en README.md
```

### Error: "DVC remote authentication failed"

**Causa**: No se pueden autenticar las credenciales SSH para el remote de DVC.

**Solución**:
```bash
# Verificar configuración del remote
dvc remote list

# Probar conexión SSH directamente
ssh -p 33 dvc@rinconseguro.com

# Si falla, configurar llave SSH
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_dvc
ssh-copy-id -i ~/.ssh/id_rsa_dvc -p 33 dvc@rinconseguro.com

# Configurar SSH para usar la llave
echo "Host rinconseguro.com
    IdentityFile ~/.ssh/id_rsa_dvc
    Port 33" >> ~/.ssh/config

# Reintentar
dvc pull
```

### Error: "DVC pull timeout" o descarga muy lenta

**Causa**: El dataset es grande (~33 GB) y la conexión es lenta.

**Solución**:
```bash
# Descargar solo el modelo (sin dataset completo)
dvc pull modelos/herdnet_best.pth.dvc
dvc pull resources/models/herdnet_best.pth.dvc

# O usar la carpeta datos/ incluida en el repo (muestras)
# Esta carpeta contiene muestras representativas para pruebas
```

### Error: "CUDA out of memory"

**Causa**: La GPU no tiene suficiente memoria.

**Solución**:
```bash
# Opción 1: Reducir tamaño de parche en configuración
# Editar resources/configs/default.yaml
model:
  patch_size: 256  # Reducir de 512 a 256

# Opción 2: Usar CPU
model:
  device: "cpu"
```

### Error: "Port 7860 already in use"

**Causa**: Otro proceso está usando el puerto.

**Solución**:
```bash
# Encontrar el proceso
lsof -i :7860

# Matar el proceso
kill -9 PID

# O usar otro puerto
GRADIO_SERVER_PORT=8080 python app.py
```

### Error: "Permission denied" en Docker

**Causa**: El usuario no tiene permisos para Docker.

**Solución**:
```bash
# Añadir usuario al grupo docker
sudo usermod -aG docker $USER

# Cerrar sesión y volver a entrar
exit
# Reconectar
```

### Error: "Container keeps restarting"

**Causa**: La aplicación falla al iniciar.

**Solución**:
```bash
# Ver logs del contenedor
docker logs wildlife-vision

# Ejecutar en modo interactivo para debug
docker run -it --rm wildlife-vision:latest /bin/bash

# Dentro del contenedor, probar manualmente
python app.py
```

### La aplicación es muy lenta

**Causa**: Está usando CPU en lugar de GPU.

**Solución**:
```bash
# Verificar que CUDA está disponible
python -c "import torch; print(torch.cuda.is_available())"

# Si es False, verificar drivers NVIDIA
nvidia-smi

# Verificar configuración
cat resources/configs/default.yaml | grep device
```

---

## Resumen de Comandos

### Despliegue Rápido (Docker)

```bash
# Clonar, construir y ejecutar
git clone https://github.com/jaimevera1107/aerial-wildlife-count.git
cd aerial-wildlife-count
docker build -t wildlife-vision .
docker run -d -p 7860:7860 --name wildlife-vision wildlife-vision
```

### Actualizar Despliegue

```bash
# Detener contenedor actual
docker stop wildlife-vision
docker rm wildlife-vision

# Obtener últimos cambios
git pull

# Reconstruir y ejecutar
docker build -t wildlife-vision .
docker run -d -p 7860:7860 --name wildlife-vision wildlife-vision
```

### Backup de Resultados

```bash
# Copiar resultados del contenedor al host
docker cp wildlife-vision:/app/resources/outputs ./backup_outputs
```

---

## Contacto y Soporte

Si encuentra problemas durante el despliegue:

- **GitHub Issues**: [Reportar problema](https://github.com/jaimevera1107/aerial-wildlife-count/issues)
- **Email**: proyecto-guacamaya@uniandes.edu.co

---

<div align="center">

**Wildlife Vision - Manual de Despliegue v1.0**

*Universidad de los Andes - Maestría en Inteligencia Artificial*

</div>


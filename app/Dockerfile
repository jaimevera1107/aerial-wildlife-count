# Imagen base con soporte CUDA + PyTorch
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Establecer directorio de trabajo
WORKDIR /app

# Instalar utilidades necesarias (git para clonar repos incluidos en requirements)
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Copiar requirements y dependencias
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar todo el proyecto
COPY . /app/

# Crear directorios necesarios en caso de que no existan
RUN mkdir -p resources/uploads resources/outputs resources/logs
# Dar permisos totales a las carpetas para evitar errores de escritura
RUN chmod -R 777 resources

# Exponer puerto para Gradio
EXPOSE 7860

# Variables obligatorias para Hugging Face Spaces y Matplotlib
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV MPLCONFIGDIR="/tmp/matplotlib"

# Comando de ejecución
CMD ["python", "app.py"]

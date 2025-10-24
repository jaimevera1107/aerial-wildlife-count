#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para generar notebooks de Google Colab completos y funcionales
"""

import json
from pathlib import Path

def create_notebook(cells):
    """Crea estructura de notebook Jupyter"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def markdown_cell(content):
    """Crea celda markdown"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.strip().split('\n')
    }

def code_cell(content):
    """Crea celda de código"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.strip().split('\n')
    }

# ==================== YOLOV8 NOTEBOOK ====================
def create_yolov8_notebook():
    cells = [
        markdown_cell("""# 🚀 Entrenamiento YOLOv8 en Google Colab

Este notebook entrena un modelo YOLOv8 con Ultralytics en Google Colab.

## 📋 Características
- Modelo: YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- Detección automática de GPU
- Conversión automática COCO a YOLO
- Visualización de resultados
- Exportación a ONNX"""),
        
        markdown_cell("## 🔧 Instalación de Dependencias"),
        
        code_cell("""# Instalar dependencias
%pip install -q ultralytics pyyaml opencv-python pillow tqdm matplotlib seaborn pandas"""),
        
        markdown_cell("## 📦 Importar Librerías"),
        
        code_cell("""import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import shutil
import json
from tqdm import tqdm

from ultralytics import YOLO
from google.colab import files, drive
from IPython.display import Image as IPImage, display

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")"""),
        
        markdown_cell("## 📁 Montar Google Drive y Configurar Rutas"),
        
        code_cell("""# Montar Google Drive
drive.mount('/content/drive')

# Configurar ruta a tus datos en Drive
DRIVE_DATA_PATH = '/content/drive/MyDrive/aerial-wildlife-count/data'

# Verificar si existe
if os.path.exists(DRIVE_DATA_PATH):
    print(f"✅ Datos encontrados en: {DRIVE_DATA_PATH}")
else:
    print(f"❌ No se encontraron datos en: {DRIVE_DATA_PATH}")
    print("Por favor, ajusta la ruta DRIVE_DATA_PATH")"""),
        
        markdown_cell("## ⚙️ Configuración del Entrenamiento"),
        
        code_cell("""# Configuración del entrenamiento
TRAINING_CONFIG = {
    'model': 'yolov8s.pt',  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    'image_size': 640,
    'epochs': 100,
    'batch_size': 16,  # Ajustar según memoria disponible
    'learning_rate': 0.01,
    'patience': 20,
    'device': 0,  # GPU 0
    'workers': 4,
    'project': '/content/runs_yolo',
    'name': 'yolo_aerial_wildlife',
    'fp16': True,  # Mixed precision
    'close_mosaic': 10,  # Cerrar mosaic en últimas 10 épocas
}

# Rutas de datos
DATA_PATHS = {
    'train_json': f'{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/train_joined.json',
    'train_images': f'{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/images',
    'val_json': f'{DRIVE_DATA_PATH}/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json',
    'val_images': f'{DRIVE_DATA_PATH}/val',
    'test_json': f'{DRIVE_DATA_PATH}/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json',
    'test_images': f'{DRIVE_DATA_PATH}/test',
}

# Clases del dataset
CLASSES = ['A', 'B', 'E', 'K', 'WH', 'WB']
NUM_CLASSES = len(CLASSES)

print("📋 Configuración de Entrenamiento:")
for key, value in TRAINING_CONFIG.items():
    print(f"  {key}: {value}")
print(f"\\n📊 Dataset:")
print(f"  Clases: {CLASSES}")
print(f"  Número de clases: {NUM_CLASSES}")"""),
        
        markdown_cell("## 🔄 Conversión de Datos COCO a YOLO"),
        
        code_cell("""def coco_to_yolo(coco_json_path, images_dir, output_dir, class_names):
    \"\"\"Convierte anotaciones COCO a formato YOLO\"\"\"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Leer archivo COCO
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Crear mapeo de categorías
    cat_id_to_class = {cat['id']: cat['name'] for cat in coco_data['categories']}
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Crear mapeo de imágenes
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Procesar anotaciones
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convertir cada imagen
    for img_id, img_info in tqdm(img_id_to_info.items(), desc="Convirtiendo a YOLO"):
        img_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Crear archivo de anotación YOLO
        txt_name = Path(img_name).stem + '.txt'
        txt_path = output_dir / txt_name
        
        with open(txt_path, 'w') as f:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    cat_name = cat_id_to_class[ann['category_id']]
                    if cat_name in class_to_id:
                        class_id = class_to_id[cat_name]
                        
                        # Convertir bbox [x, y, w, h] a [center_x, center_y, w, h] normalizado
                        x, y, w, h = ann['bbox']
                        center_x = (x + w/2) / img_width
                        center_y = (y + h/2) / img_height
                        norm_w = w / img_width
                        norm_h = h / img_height
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\\n")
    
    print(f"✅ Convertido {len(img_id_to_info)} imágenes a formato YOLO")
    return output_dir

# Convertir datos de entrenamiento
print("🔄 Convirtiendo datos de entrenamiento...")
train_yolo_dir = coco_to_yolo(
    DATA_PATHS['train_json'],
    DATA_PATHS['train_images'],
    '/content/yolo_data/train/labels',
    CLASSES
)

# Convertir datos de validación
print("🔄 Convirtiendo datos de validación...")
val_yolo_dir = coco_to_yolo(
    DATA_PATHS['val_json'],
    DATA_PATHS['val_images'],
    '/content/yolo_data/val/labels',
    CLASSES
)

print("✅ Conversión completada")"""),
        
        markdown_cell("## 📂 Copiar Imágenes a Estructura YOLO"),
        
        code_cell("""# Crear directorios de imágenes
os.makedirs('/content/yolo_data/train/images', exist_ok=True)
os.makedirs('/content/yolo_data/val/images', exist_ok=True)

# Copiar imágenes de entrenamiento
print("📁 Copiando imágenes de entrenamiento...")
for img_file in tqdm(os.listdir(DATA_PATHS['train_images'])):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        src = os.path.join(DATA_PATHS['train_images'], img_file)
        dst = os.path.join('/content/yolo_data/train/images', img_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

# Copiar imágenes de validación
print("📁 Copiando imágenes de validación...")
for img_file in tqdm(os.listdir(DATA_PATHS['val_images'])):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        src = os.path.join(DATA_PATHS['val_images'], img_file)
        dst = os.path.join('/content/yolo_data/val/images', img_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

print("✅ Estructura de datos YOLO creada")
print(f"📊 Imágenes de entrenamiento: {len(os.listdir('/content/yolo_data/train/images'))}")
print(f"📊 Imágenes de validación: {len(os.listdir('/content/yolo_data/val/images'))}")
print(f"📊 Anotaciones de entrenamiento: {len(os.listdir('/content/yolo_data/train/labels'))}")
print(f"📊 Anotaciones de validación: {len(os.listdir('/content/yolo_data/val/labels'))}")"""),
        
        markdown_cell("## 📝 Crear Archivo de Configuración YOLO"),
        
        code_cell("""# Crear archivo de configuración YOLO
yolo_config = f\"\"\"# Dataset configuration for YOLOv8
path: /content/yolo_data
train: train/images
val: val/images

# Classes
nc: {NUM_CLASSES}
names: {CLASSES}
\"\"\"

with open('/content/yolo_data/dataset.yaml', 'w') as f:
    f.write(yolo_config)

print("✅ Configuración YOLO creada en /content/yolo_data/dataset.yaml")
print("\\n📄 Contenido:")
print(yolo_config)"""),
        
        markdown_cell("## 🚀 Entrenamiento del Modelo"),
        
        code_cell("""# Inicializar modelo YOLOv8
model = YOLO(TRAINING_CONFIG['model'])

# Configurar parámetros de entrenamiento
train_args = {
    'data': '/content/yolo_data/dataset.yaml',
    'epochs': TRAINING_CONFIG['epochs'],
    'imgsz': TRAINING_CONFIG['image_size'],
    'batch': TRAINING_CONFIG['batch_size'],
    'device': TRAINING_CONFIG['device'],
    'workers': TRAINING_CONFIG['workers'],
    'project': TRAINING_CONFIG['project'],
    'name': TRAINING_CONFIG['name'],
    'patience': TRAINING_CONFIG['patience'],
    'lr0': TRAINING_CONFIG['learning_rate'],
    'amp': TRAINING_CONFIG['fp16'],
    'close_mosaic': TRAINING_CONFIG['close_mosaic'],
    'save': True,
    'save_period': 10,
    'val': True,
    'plots': True,
    'verbose': True,
}

print("🚀 Iniciando entrenamiento...")
print("📋 Parámetros de entrenamiento:")
for key, value in train_args.items():
    print(f"  {key}: {value}")

# Iniciar entrenamiento
results = model.train(**train_args)

print("✅ Entrenamiento completado!")
print(f"📁 Resultados guardados en: {TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}")"""),
        
        markdown_cell("## 📊 Visualización de Resultados"),
        
        code_cell("""# Cargar el mejor modelo entrenado
best_model_path = f"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}/weights/best.pt"
model = YOLO(best_model_path)

# Visualizar curvas de entrenamiento
results_dir = f"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}"
if os.path.exists(f"{results_dir}/results.png"):
    display(IPImage(f"{results_dir}/results.png"))

# Mostrar métricas finales
if os.path.exists(f"{results_dir}/results.csv"):
    results_df = pd.read_csv(f"{results_dir}/results.csv")
    print("📊 Métricas de entrenamiento:")
    print(results_df.tail(10))

# Mostrar matriz de confusión
if os.path.exists(f"{results_dir}/confusion_matrix.png"):
    print("\\n📊 Matriz de Confusión:")
    display(IPImage(f"{results_dir}/confusion_matrix.png"))"""),
        
        markdown_cell("## 🔍 Inferencia y Pruebas"),
        
        code_cell("""# Realizar inferencia en imágenes de prueba
test_images_dir = DATA_PATHS['test_images']
test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Seleccionar algunas imágenes para prueba
sample_images = test_images[:5]  # Primeras 5 imágenes

print(f"🔍 Realizando inferencia en {len(sample_images)} imágenes de prueba...")

for img_name in sample_images:
    img_path = os.path.join(test_images_dir, img_name)
    
    # Realizar predicción
    results = model(img_path, conf=0.5)
    
    # Mostrar resultado
    for r in results:
        # Guardar imagen con predicciones
        output_path = f"/content/test_results_{img_name}"
        r.save(output_path)
        
        # Mostrar imagen
        display(IPImage(output_path))
        
        # Mostrar estadísticas
        print(f"📊 {img_name}: {len(r.boxes)} objetos detectados")
        if len(r.boxes) > 0:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = CLASSES[class_id]
                print(f"  - {class_name}: {confidence:.2f}")
        print()"""),
        
        markdown_cell("## 💾 Guardar y Exportar Modelo"),
        
        code_cell("""# Exportar modelo a ONNX para deployment
print("🔄 Exportando modelo a ONNX...")
onnx_path = model.export(format='onnx', imgsz=TRAINING_CONFIG['image_size'])
print(f"✅ Modelo exportado a: {onnx_path}")

# Copiar resultados a Google Drive
drive_results_dir = f"/content/drive/MyDrive/aerial-wildlife-count/results/yolov8_{TRAINING_CONFIG['name']}"
os.makedirs(drive_results_dir, exist_ok=True)

# Copiar archivos importantes
files_to_copy = [
    f"{results_dir}/weights/best.pt",
    f"{results_dir}/weights/last.pt",
    f"{results_dir}/results.png",
    f"{results_dir}/confusion_matrix.png",
    f"{results_dir}/results.csv",
    onnx_path
]

for file_path in files_to_copy:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        shutil.copy2(file_path, os.path.join(drive_results_dir, filename))
        print(f"📁 Copiado: {filename}")

print(f"✅ Resultados guardados en Google Drive: {drive_results_dir}")

# Mostrar resumen final
print("\\n🎉 RESUMEN DEL ENTRENAMIENTO")
print("=" * 50)
print(f"Modelo: {TRAINING_CONFIG['model']}")
print(f"Épocas: {TRAINING_CONFIG['epochs']}")
print(f"Tamaño de imagen: {TRAINING_CONFIG['image_size']}")
print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"Clases: {CLASSES}")
print(f"Mejor modelo: {best_model_path}")
print(f"Modelo ONNX: {onnx_path}")
print(f"Resultados en Drive: {drive_results_dir}")"""),
    ]
    
    return create_notebook(cells)

# Generar notebook
notebook = create_yolov8_notebook()
output_path = Path("train_yolov8_colab.ipynb")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook creado: {output_path}")


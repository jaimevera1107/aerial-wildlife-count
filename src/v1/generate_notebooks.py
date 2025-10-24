#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script simple para generar notebooks de Google Colab
"""

import json
import os

def create_yolov8_notebook():
    """Crear notebook de YOLOv8"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Entrenamiento YOLOv8 en Google Colab\n",
                "\n",
                "Este notebook entrena un modelo YOLOv8 con Ultralytics en Google Colab.\n",
                "\n",
                "## 📋 Características\n",
                "- Modelo: YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x\n",
                "- Detección automática de GPU\n",
                "- Conversión automática COCO a YOLO\n",
                "- Visualización de resultados\n",
                "- Exportación a ONNX"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔧 Instalación de Dependencias"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Instalar dependencias\n",
                "%pip install -q ultralytics pyyaml opencv-python pillow tqdm matplotlib seaborn pandas"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📦 Importar Librerías"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import yaml\n",
                "import torch\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import cv2\n",
                "from PIL import Image\n",
                "import shutil\n",
                "import json\n",
                "from tqdm import tqdm\n",
                "\n",
                "from ultralytics import YOLO\n",
                "from google.colab import files, drive\n",
                "from IPython.display import Image as IPImage, display\n",
                "\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📁 Montar Google Drive y Configurar Rutas"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Montar Google Drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Configurar ruta a tus datos en Drive\n",
                "DRIVE_DATA_PATH = '/content/drive/MyDrive/aerial-wildlife-count/data'\n",
                "\n",
                "# Verificar si existe\n",
                "if os.path.exists(DRIVE_DATA_PATH):\n",
                "    print(f\"✅ Datos encontrados en: {DRIVE_DATA_PATH}\")\n",
                "else:\n",
                "    print(f\"❌ No se encontraron datos en: {DRIVE_DATA_PATH}\")\n",
                "    print(\"Por favor, ajusta la ruta DRIVE_DATA_PATH\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Configuración del Entrenamiento"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Configuración del entrenamiento\n",
                "TRAINING_CONFIG = {\n",
                "    'model': 'yolov8s.pt',  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt\n",
                "    'image_size': 640,\n",
                "    'epochs': 100,\n",
                "    'batch_size': 16,  # Ajustar según memoria disponible\n",
                "    'learning_rate': 0.01,\n",
                "    'patience': 20,\n",
                "    'device': 0,  # GPU 0\n",
                "    'workers': 4,\n",
                "    'project': '/content/runs_yolo',\n",
                "    'name': 'yolo_aerial_wildlife',\n",
                "    'fp16': True,  # Mixed precision\n",
                "    'close_mosaic': 10,  # Cerrar mosaic en últimas 10 épocas\n",
                "}\n",
                "\n",
                "# Rutas de datos\n",
                "DATA_PATHS = {\n",
                "    'train_json': f'{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/train_joined.json',\n",
                "    'train_images': f'{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/images',\n",
                "    'val_json': f'{DRIVE_DATA_PATH}/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json',\n",
                "    'val_images': f'{DRIVE_DATA_PATH}/val',\n",
                "    'test_json': f'{DRIVE_DATA_PATH}/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json',\n",
                "    'test_images': f'{DRIVE_DATA_PATH}/test',\n",
                "}\n",
                "\n",
                "# Clases del dataset\n",
                "CLASSES = ['A', 'B', 'E', 'K', 'WH', 'WB']\n",
                "NUM_CLASSES = len(CLASSES)\n",
                "\n",
                "print(\"📋 Configuración de Entrenamiento:\")\n",
                "for key, value in TRAINING_CONFIG.items():\n",
                "    print(f\"  {key}: {value}\")\n",
                "print(f\"\\n📊 Dataset:\")\n",
                "print(f\"  Clases: {CLASSES}\")\n",
                "print(f\"  Número de clases: {NUM_CLASSES}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔄 Conversión de Datos COCO a YOLO"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def coco_to_yolo(coco_json_path, images_dir, output_dir, class_names):\n",
                "    \"\"\"Convierte anotaciones COCO a formato YOLO\"\"\"\n",
                "    output_dir = Path(output_dir)\n",
                "    output_dir.mkdir(parents=True, exist_ok=True)\n",
                "    \n",
                "    # Leer archivo COCO\n",
                "    with open(coco_json_path, 'r') as f:\n",
                "        coco_data = json.load(f)\n",
                "    \n",
                "    # Crear mapeo de categorías\n",
                "    cat_id_to_class = {cat['id']: cat['name'] for cat in coco_data['categories']}\n",
                "    class_to_id = {name: idx for idx, name in enumerate(class_names)}\n",
                "    \n",
                "    # Crear mapeo de imágenes\n",
                "    img_id_to_info = {img['id']: img for img in coco_data['images']}\n",
                "    \n",
                "    # Procesar anotaciones\n",
                "    annotations_by_image = {}\n",
                "    for ann in coco_data['annotations']:\n",
                "        img_id = ann['image_id']\n",
                "        if img_id not in annotations_by_image:\n",
                "            annotations_by_image[img_id] = []\n",
                "        annotations_by_image[img_id].append(ann)\n",
                "    \n",
                "    # Convertir cada imagen\n",
                "    for img_id, img_info in tqdm(img_id_to_info.items(), desc=\"Convirtiendo a YOLO\"):\n",
                "        img_name = img_info['file_name']\n",
                "        img_width = img_info['width']\n",
                "        img_height = img_info['height']\n",
                "        \n",
                "        # Crear archivo de anotación YOLO\n",
                "        txt_name = Path(img_name).stem + '.txt'\n",
                "        txt_path = output_dir / txt_name\n",
                "        \n",
                "        with open(txt_path, 'w') as f:\n",
                "            if img_id in annotations_by_image:\n",
                "                for ann in annotations_by_image[img_id]:\n",
                "                    cat_name = cat_id_to_class[ann['category_id']]\n",
                "                    if cat_name in class_to_id:\n",
                "                        class_id = class_to_id[cat_name]\n",
                "                        \n",
                "                        # Convertir bbox [x, y, w, h] a [center_x, center_y, w, h] normalizado\n",
                "                        x, y, w, h = ann['bbox']\n",
                "                        center_x = (x + w/2) / img_width\n",
                "                        center_y = (y + h/2) / img_height\n",
                "                        norm_w = w / img_width\n",
                "                        norm_h = h / img_height\n",
                "                        \n",
                "                        f.write(f\"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\\n\")\n",
                "    \n",
                "    print(f\"✅ Convertido {len(img_id_to_info)} imágenes a formato YOLO\")\n",
                "    return output_dir\n",
                "\n",
                "# Convertir datos de entrenamiento\n",
                "print(\"🔄 Convirtiendo datos de entrenamiento...\")\n",
                "train_yolo_dir = coco_to_yolo(\n",
                "    DATA_PATHS['train_json'],\n",
                "    DATA_PATHS['train_images'],\n",
                "    '/content/yolo_data/train/labels',\n",
                "    CLASSES\n",
                ")\n",
                "\n",
                "# Convertir datos de validación\n",
                "print(\"🔄 Convirtiendo datos de validación...\")\n",
                "val_yolo_dir = coco_to_yolo(\n",
                "    DATA_PATHS['val_json'],\n",
                "    DATA_PATHS['val_images'],\n",
                "    '/content/yolo_data/val/labels',\n",
                "    CLASSES\n",
                ")\n",
                "\n",
                "print(\"✅ Conversión completada\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📂 Copiar Imágenes a Estructura YOLO"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Crear directorios de imágenes\n",
                "os.makedirs('/content/yolo_data/train/images', exist_ok=True)\n",
                "os.makedirs('/content/yolo_data/val/images', exist_ok=True)\n",
                "\n",
                "# Copiar imágenes de entrenamiento\n",
                "print(\"📁 Copiando imágenes de entrenamiento...\")\n",
                "for img_file in tqdm(os.listdir(DATA_PATHS['train_images'])):\n",
                "    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):\n",
                "        src = os.path.join(DATA_PATHS['train_images'], img_file)\n",
                "        dst = os.path.join('/content/yolo_data/train/images', img_file)\n",
                "        if os.path.exists(src):\n",
                "            shutil.copy2(src, dst)\n",
                "\n",
                "# Copiar imágenes de validación\n",
                "print(\"📁 Copiando imágenes de validación...\")\n",
                "for img_file in tqdm(os.listdir(DATA_PATHS['val_images'])):\n",
                "    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):\n",
                "        src = os.path.join(DATA_PATHS['val_images'], img_file)\n",
                "        dst = os.path.join('/content/yolo_data/val/images', img_file)\n",
                "        if os.path.exists(src):\n",
                "            shutil.copy2(src, dst)\n",
                "\n",
                "print(\"✅ Estructura de datos YOLO creada\")\n",
                "print(f\"📊 Imágenes de entrenamiento: {len(os.listdir('/content/yolo_data/train/images'))}\")\n",
                "print(f\"📊 Imágenes de validación: {len(os.listdir('/content/yolo_data/val/images'))}\")\n",
                "print(f\"📊 Anotaciones de entrenamiento: {len(os.listdir('/content/yolo_data/train/labels'))}\")\n",
                "print(f\"📊 Anotaciones de validación: {len(os.listdir('/content/yolo_data/val/labels'))}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📝 Crear Archivo de Configuración YOLO"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Crear archivo de configuración YOLO\n",
                "yolo_config = f\"\"\"# Dataset configuration for YOLOv8\n",
                "path: /content/yolo_data\n",
                "train: train/images\n",
                "val: val/images\n",
                "\n",
                "# Classes\n",
                "nc: {NUM_CLASSES}\n",
                "names: {CLASSES}\n",
                "\"\"\"\n",
                "\n",
                "with open('/content/yolo_data/dataset.yaml', 'w') as f:\n",
                "    f.write(yolo_config)\n",
                "\n",
                "print(\"✅ Configuración YOLO creada en /content/yolo_data/dataset.yaml\")\n",
                "print(\"\\n📄 Contenido:\")\n",
                "print(yolo_config)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 Entrenamiento del Modelo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inicializar modelo YOLOv8\n",
                "model = YOLO(TRAINING_CONFIG['model'])\n",
                "\n",
                "# Configurar parámetros de entrenamiento\n",
                "train_args = {\n",
                "    'data': '/content/yolo_data/dataset.yaml',\n",
                "    'epochs': TRAINING_CONFIG['epochs'],\n",
                "    'imgsz': TRAINING_CONFIG['image_size'],\n",
                "    'batch': TRAINING_CONFIG['batch_size'],\n",
                "    'device': TRAINING_CONFIG['device'],\n",
                "    'workers': TRAINING_CONFIG['workers'],\n",
                "    'project': TRAINING_CONFIG['project'],\n",
                "    'name': TRAINING_CONFIG['name'],\n",
                "    'patience': TRAINING_CONFIG['patience'],\n",
                "    'lr0': TRAINING_CONFIG['learning_rate'],\n",
                "    'amp': TRAINING_CONFIG['fp16'],\n",
                "    'close_mosaic': TRAINING_CONFIG['close_mosaic'],\n",
                "    'save': True,\n",
                "    'save_period': 10,\n",
                "    'val': True,\n",
                "    'plots': True,\n",
                "    'verbose': True,\n",
                "}\n",
                "\n",
                "print(\"🚀 Iniciando entrenamiento...\")\n",
                "print(\"📋 Parámetros de entrenamiento:\")\n",
                "for key, value in train_args.items():\n",
                "    print(f\"  {key}: {value}\")\n",
                "\n",
                "# Iniciar entrenamiento\n",
                "results = model.train(**train_args)\n",
                "\n",
                "print(\"✅ Entrenamiento completado!\")\n",
                "print(f\"📁 Resultados guardados en: {TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 Visualización de Resultados"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cargar el mejor modelo entrenado\n",
                "best_model_path = f\"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}/weights/best.pt\"\n",
                "model = YOLO(best_model_path)\n",
                "\n",
                "# Visualizar curvas de entrenamiento\n",
                "results_dir = f\"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}\"\n",
                "if os.path.exists(f\"{results_dir}/results.png\"):\n",
                "    display(IPImage(f\"{results_dir}/results.png\"))\n",
                "\n",
                "# Mostrar métricas finales\n",
                "if os.path.exists(f\"{results_dir}/results.csv\"):\n",
                "    results_df = pd.read_csv(f\"{results_dir}/results.csv\")\n",
                "    print(\"📊 Métricas de entrenamiento:\")\n",
                "    print(results_df.tail(10))\n",
                "\n",
                "# Mostrar matriz de confusión\n",
                "if os.path.exists(f\"{results_dir}/confusion_matrix.png\"):\n",
                "    print(\"\\n📊 Matriz de Confusión:\")\n",
                "    display(IPImage(f\"{results_dir}/confusion_matrix.png\"))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔍 Inferencia y Pruebas"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Realizar inferencia en imágenes de prueba\n",
                "test_images_dir = DATA_PATHS['test_images']\n",
                "test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n",
                "\n",
                "# Seleccionar algunas imágenes para prueba\n",
                "sample_images = test_images[:5]  # Primeras 5 imágenes\n",
                "\n",
                "print(f\"🔍 Realizando inferencia en {len(sample_images)} imágenes de prueba...\")\n",
                "\n",
                "for img_name in sample_images:\n",
                "    img_path = os.path.join(test_images_dir, img_name)\n",
                "    \n",
                "    # Realizar predicción\n",
                "    results = model(img_path, conf=0.5)\n",
                "    \n",
                "    # Mostrar resultado\n",
                "    for r in results:\n",
                "        # Guardar imagen con predicciones\n",
                "        output_path = f\"/content/test_results_{img_name}\"\n",
                "        r.save(output_path)\n",
                "        \n",
                "        # Mostrar imagen\n",
                "        display(IPImage(output_path))\n",
                "        \n",
                "        # Mostrar estadísticas\n",
                "        print(f\"📊 {img_name}: {len(r.boxes)} objetos detectados\")\n",
                "        if len(r.boxes) > 0:\n",
                "            for box in r.boxes:\n",
                "                class_id = int(box.cls[0])\n",
                "                confidence = float(box.conf[0])\n",
                "                class_name = CLASSES[class_id]\n",
                "                print(f\"  - {class_name}: {confidence:.2f}\")\n",
                "        print()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💾 Guardar y Exportar Modelo"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Exportar modelo a ONNX para deployment\n",
                "print(\"🔄 Exportando modelo a ONNX...\")\n",
                "onnx_path = model.export(format='onnx', imgsz=TRAINING_CONFIG['image_size'])\n",
                "print(f\"✅ Modelo exportado a: {onnx_path}\")\n",
                "\n",
                "# Copiar resultados a Google Drive\n",
                "drive_results_dir = f\"/content/drive/MyDrive/aerial-wildlife-count/results/yolov8_{TRAINING_CONFIG['name']}\"\n",
                "os.makedirs(drive_results_dir, exist_ok=True)\n",
                "\n",
                "# Copiar archivos importantes\n",
                "files_to_copy = [\n",
                "    f\"{results_dir}/weights/best.pt\",\n",
                "    f\"{results_dir}/weights/last.pt\",\n",
                "    f\"{results_dir}/results.png\",\n",
                "    f\"{results_dir}/confusion_matrix.png\",\n",
                "    f\"{results_dir}/results.csv\",\n",
                "    onnx_path\n",
                "]\n",
                "\n",
                "for file_path in files_to_copy:\n",
                "    if os.path.exists(file_path):\n",
                "        filename = os.path.basename(file_path)\n",
                "        shutil.copy2(file_path, os.path.join(drive_results_dir, filename))\n",
                "        print(f\"📁 Copiado: {filename}\")\n",
                "\n",
                "print(f\"✅ Resultados guardados en Google Drive: {drive_results_dir}\")\n",
                "\n",
                "# Mostrar resumen final\n",
                "print(\"\\n🎉 RESUMEN DEL ENTRENAMIENTO\")\n",
                "print(\"=\" * 50)\n",
                "print(f\"Modelo: {TRAINING_CONFIG['model']}\")\n",
                "print(f\"Épocas: {TRAINING_CONFIG['epochs']}\")\n",
                "print(f\"Tamaño de imagen: {TRAINING_CONFIG['image_size']}\")\n",
                "print(f\"Batch size: {TRAINING_CONFIG['batch_size']}\")\n",
                "print(f\"Clases: {CLASSES}\")\n",
                "print(f\"Mejor modelo: {best_model_path}\")\n",
                "print(f\"Modelo ONNX: {onnx_path}\")\n",
                "print(f\"Resultados en Drive: {drive_results_dir}\")"
            ]
        }
    ]
    
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

def main():
    # Crear notebook de YOLOv8
    notebook = create_yolov8_notebook()
    
    # Guardar notebook
    with open('train_yolov8_colab.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("Notebook YOLOv8 creado exitosamente!")

if __name__ == "__main__":
    main()

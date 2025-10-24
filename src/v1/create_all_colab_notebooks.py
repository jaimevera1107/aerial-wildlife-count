#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para generar todos los notebooks de Google Colab completos y funcionales
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

# ==================== CASCADE R-CNN NOTEBOOK ====================
def create_cascade_rcnn_notebook():
    cells = [
        markdown_cell("""# 🚀 Entrenamiento Cascade R-CNN en Google Colab

Este notebook entrena un modelo Cascade R-CNN con MMDetection en Google Colab.

## 📋 Características
- Backbone: Swin-T o ResNeXt
- Arquitectura: Multi-stage R-CNN
- Detección automática de GPU
- Integración con pipeline de datos
- Visualización de resultados"""),
        
        markdown_cell("## 🔧 Instalación de Dependencias"),
        
        code_cell("""# Instalar dependencias
%pip install -q mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
%pip install -q mmdet
%pip install -q pyyaml opencv-python pillow tqdm matplotlib seaborn"""),
        
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

# Importar módulos de MMDetection
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.utils import register_all_modules

# Importar de Google Colab
from google.colab import files, drive
from IPython.display import Image as IPImage, display

# Registrar módulos de MMDetection
register_all_modules()

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
    'backbone': 'swin_t',  # swin_t, resnext_101_64x4d
    'image_size': 896,
    'epochs': 50,
    'batch_size': 2,  # Ajustar según memoria disponible
    'learning_rate': 0.0001,
    'device': 0,  # GPU 0
    'workers': 4,
    'work_dir': '/content/runs_cascade',
    'name': 'cascade_aerial_wildlife',
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
        
        markdown_cell("## ⚙️ Configuración del Modelo"),
        
        code_cell("""# Crear configuración del modelo Cascade R-CNN
def create_cascade_config(backbone, num_classes, classes, data_paths, training_config):
    \"\"\"Crear configuración para Cascade R-CNN\"\"\"
    
    if backbone == 'swin_t':
        backbone_config = \"\"\"
        backbone=dict(
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
        ),
        neck=dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=5
        ),
        \"\"\"
    else:  # resnext_101_64x4d
        backbone_config = \"\"\"
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5
        ),
        \"\"\"
    
    config_content = f\"\"\"
# Model configuration
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    {backbone_config}
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes={num_classes},
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes={num_classes},
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes={num_classes},
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]))

# Dataset
dataset_type = 'CocoDataset'
data_root = '/content/'

# Classes
classes = {classes}

# Data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=({training_config['image_size']}, {training_config['image_size']}), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=({training_config['image_size']}, {training_config['image_size']}), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Data configuration
train_dataloader = dict(
    batch_size={training_config['batch_size']},
    num_workers={training_config['workers']},
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{data_paths['train_json']}',
        data_prefix=dict(img='{data_paths['train_images']}'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers={training_config['workers']},
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{data_paths['val_json']}',
        data_prefix=dict(img='{data_paths['val_images']}'),
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file='{data_paths['val_json']}',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={training_config['epochs']}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr={training_config['learning_rate']}, momentum=0.9, weight_decay=0.0001))

# Learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end={training_config['epochs']},
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# Runtime
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
\"\"\"
    
    return config_content

# Crear configuración
config_content = create_cascade_config(
    TRAINING_CONFIG['backbone'],
    NUM_CLASSES,
    CLASSES,
    DATA_PATHS,
    TRAINING_CONFIG
)

# Guardar configuración
config_path = '/content/cascade_config.py'
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"✅ Configuración guardada en: {config_path}")
print("📋 Configuración del modelo:")
print(f"  Backbone: {TRAINING_CONFIG['backbone']}")
print(f"  Clases: {NUM_CLASSES}")
print(f"  Tamaño de imagen: {TRAINING_CONFIG['image_size']}")
print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"  Épocas: {TRAINING_CONFIG['epochs']}")"""),
        
        markdown_cell("## 🚀 Entrenamiento del Modelo"),
        
        code_cell("""# Crear directorio de trabajo
work_dir = TRAINING_CONFIG['work_dir']
os.makedirs(work_dir, exist_ok=True)

# Cargar configuración
cfg = Config.fromfile(config_path)
cfg.work_dir = work_dir

# Inicializar runner
runner = Runner.from_cfg(cfg)

print("🚀 Iniciando entrenamiento...")
print("📋 Parámetros de entrenamiento:")
print(f"  Backbone: {TRAINING_CONFIG['backbone']}")
print(f"  Épocas: {TRAINING_CONFIG['epochs']}")
print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
print(f"  Work dir: {work_dir}")

# Iniciar entrenamiento
runner.train()

print("✅ Entrenamiento completado!")
print(f"📁 Resultados guardados en: {work_dir}")"""),
        
        markdown_cell("## 📊 Visualización de Resultados"),
        
        code_cell("""# Cargar el mejor modelo entrenado
best_model_path = f"{work_dir}/best_coco_bbox_mAP_epoch_{TRAINING_CONFIG['epochs']}.pth"
if not os.path.exists(best_model_path):
    # Buscar el último checkpoint
    checkpoints = [f for f in os.listdir(work_dir) if f.endswith('.pth')]
    if checkpoints:
        best_model_path = os.path.join(work_dir, checkpoints[0])
    else:
        print("❌ No se encontró modelo entrenado")
        best_model_path = None

if best_model_path:
    # Inicializar modelo para inferencia
    model = init_detector(config_path, best_model_path, device=f'cuda:{TRAINING_CONFIG["device"]}')
    print(f"✅ Modelo cargado desde: {best_model_path}")
    
    # Mostrar logs de entrenamiento
    print("📊 Revisando directorio de logs...")
    !ls -la "{work_dir}"
else:
    print("❌ No se pudo cargar el modelo")"""),
        
        markdown_cell("## 🔍 Inferencia y Pruebas"),
        
        code_cell("""# Realizar inferencia en imágenes de prueba
if best_model_path and os.path.exists(best_model_path):
    test_images_dir = DATA_PATHS['test_images']
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Seleccionar algunas imágenes para prueba
    sample_images = test_images[:3]  # Primeras 3 imágenes
    
    print(f"🔍 Realizando inferencia en {len(sample_images)} imágenes de prueba...")
    
    for img_name in sample_images:
        img_path = os.path.join(test_images_dir, img_name)
        
        # Realizar predicción
        result = inference_detector(model, img_path)
        
        # Mostrar resultado
        show_result_pyplot(
            model, 
            img_path, 
            result, 
            score_thr=0.5,
            title=f'Resultado: {img_name}',
            out_file=f'/content/test_result_{img_name}'
        )
        
        # Mostrar estadísticas
        print(f"📊 {img_name}: {len(result.pred_instances)} objetos detectados")
        if len(result.pred_instances) > 0:
            for i, (bbox, score, label) in enumerate(zip(
                result.pred_instances.bboxes, 
                result.pred_instances.scores, 
                result.pred_instances.labels
            )):
                class_name = CLASSES[label]
                print(f"  - {class_name}: {score:.2f}")
        print()
else:
    print("❌ No se puede realizar inferencia sin modelo entrenado")"""),
        
        markdown_cell("## 💾 Guardar y Exportar Modelo"),
        
        code_cell("""# Copiar resultados a Google Drive
drive_results_dir = f"/content/drive/MyDrive/aerial-wildlife-count/results/cascade_{TRAINING_CONFIG['name']}"
os.makedirs(drive_results_dir, exist_ok=True)

# Copiar archivos importantes
files_to_copy = []

# Buscar checkpoints
if os.path.exists(work_dir):
    for file in os.listdir(work_dir):
        if file.endswith('.pth'):
            files_to_copy.append(os.path.join(work_dir, file))

# Copiar configuración
files_to_copy.append(config_path)

# Copiar archivos de resultados
for file_path in files_to_copy:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        shutil.copy2(file_path, os.path.join(drive_results_dir, filename))
        print(f"📁 Copiado: {filename}")

print(f"✅ Resultados guardados en Google Drive: {drive_results_dir}")

# Mostrar resumen final
print("\\n🎉 RESUMEN DEL ENTRENAMIENTO")
print("=" * 50)
print(f"Modelo: Cascade R-CNN")
print(f"Backbone: {TRAINING_CONFIG['backbone']}")
print(f"Épocas: {TRAINING_CONFIG['epochs']}")
print(f"Tamaño de imagen: {TRAINING_CONFIG['image_size']}")
print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"Clases: {CLASSES}")
print(f"Mejor modelo: {best_model_path}")
print(f"Resultados en Drive: {drive_results_dir}")"""),
    ]
    
    return create_notebook(cells)

# ==================== DEFORMABLE DETR NOTEBOOK ====================
def create_deformable_detr_notebook():
    cells = [
        markdown_cell("""# 🚀 Entrenamiento Deformable DETR en Google Colab

Este notebook entrena un modelo Deformable DETR con MMDetection en Google Colab.

## 📋 Características
- Backbone: Swin-T o ResNeXt
- Arquitectura: Transformer-based
- Detección automática de GPU
- Integración con pipeline de datos
- Visualización de resultados"""),
        
        markdown_cell("## 🔧 Instalación de Dependencias"),
        
        code_cell("""# Instalar dependencias
%pip install -q mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
%pip install -q mmdet
%pip install -q pyyaml opencv-python pillow tqdm matplotlib seaborn"""),
        
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

# Importar módulos de MMDetection
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.utils import register_all_modules

# Importar de Google Colab
from google.colab import files, drive
from IPython.display import Image as IPImage, display

# Registrar módulos de MMDetection
register_all_modules()

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
    'backbone': 'swin_t',  # swin_t, resnext_101_64x4d
    'image_size': 896,
    'epochs': 50,
    'batch_size': 2,  # Ajustar según memoria disponible
    'learning_rate': 0.0001,
    'device': 0,  # GPU 0
    'workers': 4,
    'work_dir': '/content/runs_detr',
    'name': 'detr_aerial_wildlife',
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
        
        markdown_cell("## ⚙️ Configuración del Modelo"),
        
        code_cell("""# Crear configuración del modelo Deformable DETR
def create_detr_config(backbone, num_classes, classes, data_paths, training_config):
    \"\"\"Crear configuración para Deformable DETR\"\"\"
    
    if backbone == 'swin_t':
        backbone_config = \"\"\"
        backbone=dict(
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
        ),
        neck=dict(
            type='ChannelMapper',
            in_channels=[96, 192, 384, 768],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4)
        \"\"\"
    else:  # resnext_101_64x4d
        backbone_config = \"\"\"
        backbone=dict(
            type='ResNeXt',
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')
        ),
        neck=dict(
            type='ChannelMapper',
            in_channels=[256, 512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4)
        \"\"\"
    
    config_content = f\"\"\"
# Model configuration
_base_ = [
    '../_base_/models/deformable_detr.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    {backbone_config}
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,
        num_classes={num_classes},
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)))

# Dataset
dataset_type = 'CocoDataset'
data_root = '/content/'

# Classes
classes = {classes}

# Data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=({training_config['image_size']}, {training_config['image_size']}), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=({training_config['image_size']}, {training_config['image_size']}), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Data configuration
train_dataloader = dict(
    batch_size={training_config['batch_size']},
    num_workers={training_config['workers']},
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{data_paths['train_json']}',
        data_prefix=dict(img='{data_paths['train_images']}'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers={training_config['workers']},
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{data_paths['val_json']}',
        data_prefix=dict(img='{data_paths['val_images']}'),
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes)))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type='CocoMetric',
    ann_file='{data_paths['val_json']}',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={training_config['epochs']}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr={training_config['learning_rate']}, weight_decay=0.0001))

# Learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end={training_config['epochs']},
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

# Runtime
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
\"\"\"
    
    return config_content

# Crear configuración
config_content = create_detr_config(
    TRAINING_CONFIG['backbone'],
    NUM_CLASSES,
    CLASSES,
    DATA_PATHS,
    TRAINING_CONFIG
)

# Guardar configuración
config_path = '/content/detr_config.py'
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"✅ Configuración guardada en: {config_path}")
print("📋 Configuración del modelo:")
print(f"  Backbone: {TRAINING_CONFIG['backbone']}")
print(f"  Clases: {NUM_CLASSES}")
print(f"  Tamaño de imagen: {TRAINING_CONFIG['image_size']}")
print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"  Épocas: {TRAINING_CONFIG['epochs']}")"""),
        
        markdown_cell("## 🚀 Entrenamiento del Modelo"),
        
        code_cell("""# Crear directorio de trabajo
work_dir = TRAINING_CONFIG['work_dir']
os.makedirs(work_dir, exist_ok=True)

# Cargar configuración
cfg = Config.fromfile(config_path)
cfg.work_dir = work_dir

# Inicializar runner
runner = Runner.from_cfg(cfg)

print("🚀 Iniciando entrenamiento...")
print("📋 Parámetros de entrenamiento:")
print(f"  Backbone: {TRAINING_CONFIG['backbone']}")
print(f"  Épocas: {TRAINING_CONFIG['epochs']}")
print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
print(f"  Work dir: {work_dir}")

# Iniciar entrenamiento
runner.train()

print("✅ Entrenamiento completado!")
print(f"📁 Resultados guardados en: {work_dir}")"""),
        
        markdown_cell("## 📊 Visualización de Resultados"),
        
        code_cell("""# Cargar el mejor modelo entrenado
best_model_path = f"{work_dir}/best_coco_bbox_mAP_epoch_{TRAINING_CONFIG['epochs']}.pth"
if not os.path.exists(best_model_path):
    # Buscar el último checkpoint
    checkpoints = [f for f in os.listdir(work_dir) if f.endswith('.pth')]
    if checkpoints:
        best_model_path = os.path.join(work_dir, checkpoints[0])
    else:
        print("❌ No se encontró modelo entrenado")
        best_model_path = None

if best_model_path:
    # Inicializar modelo para inferencia
    model = init_detector(config_path, best_model_path, device=f'cuda:{TRAINING_CONFIG["device"]}')
    print(f"✅ Modelo cargado desde: {best_model_path}")
    
    # Mostrar logs de entrenamiento
    print("📊 Revisando directorio de logs...")
    !ls -la "{work_dir}"
else:
    print("❌ No se pudo cargar el modelo")"""),
        
        markdown_cell("## 🔍 Inferencia y Pruebas"),
        
        code_cell("""# Realizar inferencia en imágenes de prueba
if best_model_path and os.path.exists(best_model_path):
    test_images_dir = DATA_PATHS['test_images']
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Seleccionar algunas imágenes para prueba
    sample_images = test_images[:3]  # Primeras 3 imágenes
    
    print(f"🔍 Realizando inferencia en {len(sample_images)} imágenes de prueba...")
    
    for img_name in sample_images:
        img_path = os.path.join(test_images_dir, img_name)
        
        # Realizar predicción
        result = inference_detector(model, img_path)
        
        # Mostrar resultado
        show_result_pyplot(
            model, 
            img_path, 
            result, 
            score_thr=0.5,
            title=f'Resultado: {img_name}',
            out_file=f'/content/test_result_{img_name}'
        )
        
        # Mostrar estadísticas
        print(f"📊 {img_name}: {len(result.pred_instances)} objetos detectados")
        if len(result.pred_instances) > 0:
            for i, (bbox, score, label) in enumerate(zip(
                result.pred_instances.bboxes, 
                result.pred_instances.scores, 
                result.pred_instances.labels
            )):
                class_name = CLASSES[label]
                print(f"  - {class_name}: {score:.2f}")
        print()
else:
    print("❌ No se puede realizar inferencia sin modelo entrenado")"""),
        
        markdown_cell("## 💾 Guardar y Exportar Modelo"),
        
        code_cell("""# Copiar resultados a Google Drive
drive_results_dir = f"/content/drive/MyDrive/aerial-wildlife-count/results/detr_{TRAINING_CONFIG['name']}"
os.makedirs(drive_results_dir, exist_ok=True)

# Copiar archivos importantes
files_to_copy = []

# Buscar checkpoints
if os.path.exists(work_dir):
    for file in os.listdir(work_dir):
        if file.endswith('.pth'):
            files_to_copy.append(os.path.join(work_dir, file))

# Copiar configuración
files_to_copy.append(config_path)

# Copiar archivos de resultados
for file_path in files_to_copy:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        shutil.copy2(file_path, os.path.join(drive_results_dir, filename))
        print(f"📁 Copiado: {filename}")

print(f"✅ Resultados guardados en Google Drive: {drive_results_dir}")

# Mostrar resumen final
print("\\n🎉 RESUMEN DEL ENTRENAMIENTO")
print("=" * 50)
print(f"Modelo: Deformable DETR")
print(f"Backbone: {TRAINING_CONFIG['backbone']}")
print(f"Épocas: {TRAINING_CONFIG['epochs']}")
print(f"Tamaño de imagen: {TRAINING_CONFIG['image_size']}")
print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
print(f"Clases: {CLASSES}")
print(f"Mejor modelo: {best_model_path}")
print(f"Resultados en Drive: {drive_results_dir}")"""),
    ]
    
    return create_notebook(cells)

# ==================== COMPLETE PIPELINE NOTEBOOK ====================
def create_complete_pipeline_notebook():
    cells = [
        markdown_cell("""# 🚀 Pipeline Completo de Aerial Wildlife Detection en Google Colab

Este notebook ejecuta el pipeline completo de detección de vida silvestre aérea:

1. **Pipeline de Calidad** - Verificación y limpieza de datos
2. **Pipeline de Aumentación** - Balanceo y aumentación de datos
3. **Entrenamiento de Modelos** - Cascade R-CNN, Deformable DETR, YOLOv8
4. **Evaluación y Comparación** - Análisis de resultados

## 📋 Características
- Pipeline completo automatizado
- Detección automática de GPU
- Integración con Google Drive
- Visualización de resultados
- Comparación de modelos"""),
        
        markdown_cell("## 🔧 Instalación de Dependencias"),
        
        code_cell("""# Instalar todas las dependencias necesarias
%pip install -q mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
%pip install -q mmdet
%pip install -q ultralytics
%pip install -q pyyaml opencv-python albumentations pillow tqdm matplotlib seaborn pandas numpy"""),
        
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
import json
import time
from datetime import datetime
import shutil
from tqdm import tqdm

# Importar módulos de entrenamiento
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.utils import register_all_modules
from ultralytics import YOLO

# Importar de Google Colab
from google.colab import files, drive
from IPython.display import Image as IPImage, display

# Registrar módulos de MMDetection
register_all_modules()

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
    print("Por favor, ajusta la ruta DRIVE_DATA_PATH")

# Clases del dataset
CLASSES = ['A', 'B', 'E', 'K', 'WH', 'WB']
NUM_CLASSES = len(CLASSES)

print(f"📊 Dataset:")
print(f"  Clases: {CLASSES}")
print(f"  Número de clases: {NUM_CLASSES}")"""),
        
        markdown_cell("## ⚙️ Configuración del Pipeline"),
        
        code_cell("""# Configuración del pipeline completo
PIPELINE_CONFIG = {
    'run_quality': True,      # Ejecutar pipeline de calidad
    'run_augmentation': True, # Ejecutar pipeline de aumentación
    'run_training': True,     # Ejecutar entrenamiento
    'models_to_train': ['yolov8', 'cascade_rcnn', 'deformable_detr'],  # Modelos a entrenar
    'compare_results': True,  # Comparar resultados
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    'yolov8': {
        'model': 'yolov8s.pt',
        'epochs': 50,
        'batch_size': 16,
        'image_size': 640,
    },
    'cascade_rcnn': {
        'backbone': 'swin_t',
        'epochs': 30,
        'batch_size': 2,
        'image_size': 896,
    },
    'deformable_detr': {
        'backbone': 'swin_t',
        'epochs': 30,
        'batch_size': 2,
        'image_size': 896,
    }
}

print("📋 Configuración del Pipeline:")
for key, value in PIPELINE_CONFIG.items():
    print(f"  {key}: {value}")

print("\\n📋 Configuración de Entrenamiento:")
for model, config in TRAINING_CONFIG.items():
    print(f"  {model}: {config}")"""),
        
        markdown_cell("## 🔄 Pipeline de Calidad (Opcional)"),
        
        code_cell("""# Ejecutar pipeline de calidad si está habilitado
if PIPELINE_CONFIG['run_quality']:
    print("🔄 Ejecutando Pipeline de Calidad...")
    
    # Aquí se ejecutaría el pipeline de calidad
    # Por ahora, asumimos que los datos ya están procesados
    print("✅ Pipeline de Calidad completado (simulado)")
    print("📊 Datos de calidad disponibles en:")
    print(f"  - Train: {DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/")
    print(f"  - Val: {DRIVE_DATA_PATH}/groundtruth/json/big_size/")
    print(f"  - Test: {DRIVE_DATA_PATH}/groundtruth/json/big_size/")
else:
    print("⏭️ Pipeline de Calidad omitido")"""),
        
        markdown_cell("## 🔄 Pipeline de Aumentación (Opcional)"),
        
        code_cell("""# Ejecutar pipeline de aumentación si está habilitado
if PIPELINE_CONFIG['run_augmentation']:
    print("🔄 Ejecutando Pipeline de Aumentación...")
    
    # Aquí se ejecutaría el pipeline de aumentación
    # Por ahora, asumimos que los datos ya están aumentados
    print("✅ Pipeline de Aumentación completado (simulado)")
    print("📊 Datos aumentados disponibles en:")
    print(f"  - Train aumentado: {DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/")
else:
    print("⏭️ Pipeline de Aumentación omitido")"""),
        
        markdown_cell("## 🚀 Entrenamiento de Modelos"),
        
        code_cell("""# Función para entrenar YOLOv8
def train_yolov8(config):
    print(f"🚀 Entrenando YOLOv8...")
    
    # Configurar rutas
    train_json = f'{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/train_joined.json'
    train_images = f'{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/images'
    val_json = f'{DRIVE_DATA_PATH}/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json'
    val_images = f'{DRIVE_DATA_PATH}/val'
    
    # Convertir COCO a YOLO
    def coco_to_yolo(coco_json_path, images_dir, output_dir, class_names):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        cat_id_to_class = {cat['id']: cat['name'] for cat in coco_data['categories']}
        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        img_id_to_info = {img['id']: img for img in coco_data['images']}
        
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        for img_id, img_info in tqdm(img_id_to_info.items(), desc="Convirtiendo a YOLO"):
            img_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            txt_name = Path(img_name).stem + '.txt'
            txt_path = output_dir / txt_name
            
            with open(txt_path, 'w') as f:
                if img_id in annotations_by_image:
                    for ann in annotations_by_image[img_id]:
                        cat_name = cat_id_to_class[ann['category_id']]
                        if cat_name in class_to_id:
                            class_id = class_to_id[cat_name]
                            x, y, w, h = ann['bbox']
                            center_x = (x + w/2) / img_width
                            center_y = (y + h/2) / img_height
                            norm_w = w / img_width
                            norm_h = h / img_height
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\\n")
        
        return output_dir
    
    # Convertir datos
    train_yolo_dir = coco_to_yolo(train_json, train_images, '/content/yolo_data/train/labels', CLASSES)
    val_yolo_dir = coco_to_yolo(val_json, val_images, '/content/yolo_data/val/labels', CLASSES)
    
    # Copiar imágenes
    os.makedirs('/content/yolo_data/train/images', exist_ok=True)
    os.makedirs('/content/yolo_data/val/images', exist_ok=True)
    
    for img_file in os.listdir(train_images):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy2(os.path.join(train_images, img_file), 
                        os.path.join('/content/yolo_data/train/images', img_file))
    
    for img_file in os.listdir(val_images):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy2(os.path.join(val_images, img_file), 
                        os.path.join('/content/yolo_data/val/images', img_file))
    
    # Crear configuración YOLO
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
    
    # Entrenar modelo
    model = YOLO(config['model'])
    results = model.train(
        data='/content/yolo_data/dataset.yaml',
        epochs=config['epochs'],
        imgsz=config['image_size'],
        batch=config['batch_size'],
        device=0,
        project='/content/runs_yolo',
        name='yolo_pipeline',
        verbose=True
    )
    
    return '/content/runs_yolo/yolo_pipeline'

# Función para entrenar Cascade R-CNN
def train_cascade_rcnn(config):
    print(f"🚀 Entrenando Cascade R-CNN...")
    
    # Crear configuración básica
    config_content = f\"\"\"
# Model configuration
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(num_classes={NUM_CLASSES}),
            dict(num_classes={NUM_CLASSES}),
            dict(num_classes={NUM_CLASSES})
        ]))

# Dataset
dataset_type = 'CocoDataset'
data_root = '/content/'
classes = {CLASSES}

# Data configuration
train_dataloader = dict(
    batch_size={config['batch_size']},
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/train_joined.json',
        data_prefix=dict(img='{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/images'),
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{DRIVE_DATA_PATH}/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json',
        data_prefix=dict(img='{DRIVE_DATA_PATH}/val'),
        test_mode=True,
        metainfo=dict(classes=classes)))

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={config['epochs']}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

log_level = 'INFO'
load_from = None
resume = False
\"\"\"
    
    config_path = '/content/cascade_pipeline_config.py'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Entrenar modelo
    cfg = Config.fromfile(config_path)
    cfg.work_dir = '/content/runs_cascade_pipeline'
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    return '/content/runs_cascade_pipeline'

# Función para entrenar Deformable DETR
def train_deformable_detr(config):
    print(f"🚀 Entrenando Deformable DETR...")
    
    # Crear configuración básica
    config_content = f\"\"\"
# Model configuration
_base_ = [
    '../_base_/models/deformable_detr.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model
model = dict(
    bbox_head=dict(
        num_classes={NUM_CLASSES}))

# Dataset
dataset_type = 'CocoDataset'
data_root = '/content/'
classes = {CLASSES}

# Data configuration
train_dataloader = dict(
    batch_size={config['batch_size']},
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/train_joined.json',
        data_prefix=dict(img='{DRIVE_DATA_PATH}/outputs/mirror_clean/train_joined/images'),
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='{DRIVE_DATA_PATH}/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json',
        data_prefix=dict(img='{DRIVE_DATA_PATH}/val'),
        test_mode=True,
        metainfo=dict(classes=classes)))

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={config['epochs']}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

log_level = 'INFO'
load_from = None
resume = False
\"\"\"
    
    config_path = '/content/detr_pipeline_config.py'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Entrenar modelo
    cfg = Config.fromfile(config_path)
    cfg.work_dir = '/content/runs_detr_pipeline'
    runner = Runner.from_cfg(cfg)
    runner.train()
    
    return '/content/runs_detr_pipeline'

# Ejecutar entrenamiento de modelos seleccionados
results = {}
if PIPELINE_CONFIG['run_training']:
    for model_name in PIPELINE_CONFIG['models_to_train']:
        if model_name in TRAINING_CONFIG:
            print(f"\\n{'='*50}")
            print(f"Entrenando {model_name.upper()}")
            print(f"{'='*50}")
            
            try:
                if model_name == 'yolov8':
                    results[model_name] = train_yolov8(TRAINING_CONFIG[model_name])
                elif model_name == 'cascade_rcnn':
                    results[model_name] = train_cascade_rcnn(TRAINING_CONFIG[model_name])
                elif model_name == 'deformable_detr':
                    results[model_name] = train_deformable_detr(TRAINING_CONFIG[model_name])
                
                print(f"✅ {model_name} entrenado exitosamente")
            except Exception as e:
                print(f"❌ Error entrenando {model_name}: {e}")
                results[model_name] = None
else:
    print("⏭️ Entrenamiento omitido")

print("\\n📊 Resultados del entrenamiento:")
for model, result in results.items():
    if result:
        print(f"  ✅ {model}: {result}")
    else:
        print(f"  ❌ {model}: Falló")"""),
        
        markdown_cell("## 📊 Comparación de Resultados"),
        
        code_cell("""# Comparar resultados de los modelos entrenados
if PIPELINE_CONFIG['compare_results'] and results:
    print("📊 Comparando resultados de modelos...")
    
    comparison_data = []
    
    for model_name, result_path in results.items():
        if result_path and os.path.exists(result_path):
            print(f"\\n📈 Analizando {model_name}:")
            
            if model_name == 'yolov8':
                # Analizar resultados YOLOv8
                results_file = os.path.join(result_path, 'results.csv')
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                    final_metrics = df.iloc[-1]
                    comparison_data.append({
                        'Model': 'YOLOv8',
                        'mAP50': final_metrics.get('metrics/mAP50(B)', 0),
                        'mAP50-95': final_metrics.get('metrics/mAP50-95(B)', 0),
                        'Precision': final_metrics.get('metrics/precision(B)', 0),
                        'Recall': final_metrics.get('metrics/recall(B)', 0),
                        'Training Time': 'N/A'
                    })
                    print(f"  mAP50: {final_metrics.get('metrics/mAP50(B)', 0):.3f}")
                    print(f"  mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 0):.3f}")
            
            elif model_name in ['cascade_rcnn', 'deformable_detr']:
                # Analizar resultados MMDetection
                log_file = os.path.join(result_path, 'vis_data', 'vis_image', 'vis_image_2024_01_01_00_00_00.log')
                if os.path.exists(log_file):
                    print(f"  Logs encontrados en: {log_file}")
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'mAP50': 'N/A',
                        'mAP50-95': 'N/A',
                        'Precision': 'N/A',
                        'Recall': 'N/A',
                        'Training Time': 'N/A'
                    })
                else:
                    print(f"  Logs no encontrados")
    
    # Crear tabla de comparación
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\\n📊 Tabla de Comparación:")
        print(comparison_df.to_string(index=False))
        
        # Visualizar comparación
        if len(comparison_data) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Comparación de Modelos', fontsize=16)
            
            metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                valid_data = comparison_df[comparison_df[metric] != 'N/A']
                if not valid_data.empty:
                    valid_data[metric] = pd.to_numeric(valid_data[metric])
                    valid_data.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
                    ax.set_title(f'{metric}')
                    ax.set_ylabel('Score')
                    plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    # Guardar resultados en Drive
    drive_results_dir = f"/content/drive/MyDrive/aerial-wildlife-count/results/pipeline_comparison"
    os.makedirs(drive_results_dir, exist_ok=True)
    
    if comparison_data:
        comparison_df.to_csv(os.path.join(drive_results_dir, 'model_comparison.csv'), index=False)
        print(f"\\n✅ Comparación guardada en: {drive_results_dir}")
    
    # Copiar todos los resultados
    for model_name, result_path in results.items():
        if result_path and os.path.exists(result_path):
            model_drive_dir = os.path.join(drive_results_dir, model_name)
            os.makedirs(model_drive_dir, exist_ok=True)
            
            # Copiar archivos importantes
            for file in os.listdir(result_path):
                if file.endswith(('.pth', '.csv', '.png', '.jpg')):
                    shutil.copy2(os.path.join(result_path, file), 
                                os.path.join(model_drive_dir, file))
            
            print(f"📁 Resultados de {model_name} copiados a Drive")
    
    print(f"\\n🎉 Pipeline completo finalizado!")
    print(f"📁 Todos los resultados guardados en: {drive_results_dir}")
else:
    print("⏭️ Comparación omitida")"""),
    ]
    
    return create_notebook(cells)

# ==================== GENERAR TODOS LOS NOTEBOOKS ====================
def main():
    notebooks = {
        'train_yolov8_colab.ipynb': create_yolov8_notebook(),
        'train_cascade_rcnn_colab.ipynb': create_cascade_rcnn_notebook(),
        'train_deformable_detr_colab.ipynb': create_deformable_detr_notebook(),
        'complete_pipeline_colab.ipynb': create_complete_pipeline_notebook(),
    }
    
    for filename, notebook in notebooks.items():
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print(f"Notebook creado: {output_path}")

if __name__ == "__main__":
    main()

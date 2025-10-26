#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración para entrenamiento YOLOv8 - Aerial Wildlife Detection
Archivo de configuración centralizado para todos los parámetros del entrenamiento.
"""

import os
from pathlib import Path


class YOLOConfig:
    """Configuración centralizada para entrenamiento YOLOv8"""
    
    def __init__(self):
        # ============================================================
        # PARÁMETROS DEL MODELO
        # ============================================================
        self.model = 'yolov8s.pt'  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        self.image_size = 640
        self.epochs = 100
        self.batch_size = 32  # Optimizado para velocidad
        self.learning_rate = 0.01
        self.patience = 10
        self.device = 0  # GPU 0
        self.workers = 8  # Optimizado para paralelización
        self.fp16 = True  # Mixed precision
        
        # ============================================================
        # CONFIGURACIÓN DE PROYECTO
        # ============================================================
        self.project = '/content/runs_yolo'
        self.name = 'yolo_aerial_wildlife_v1'
        self.save_period = 5  # Guardar cada 5 épocas
        
        # ============================================================
        # CONFIGURACIÓN DE BACKUP
        # ============================================================
        self.drive_backup_period = 10  # Backup en Drive cada 10 épocas
        self.drive_backup_dir = '/content/drive/MyDrive/aerial-wildlife-count-results/yolo_v1'
        self.resume_training = True  # Permitir reanudar entrenamiento
        
        # ============================================================
        # CLASES DEL DATASET
        # ============================================================
        self.classes = [
            "Buffalo", "Elephant", "Kob",
            "Alcelaphinae", "Warthog", "Waterbuck"
        ]
        
        # ============================================================
        # RUTAS DE DATOS
        # ============================================================
        self.base_dir = Path("/content/drive/MyDrive/aerial-wildlife-count")
        
        # Rutas estándar COCO
        self.train_ann_file = self.base_dir / "data" / "coco" / "train" / "train_annotations.json"
        self.val_ann_file = self.base_dir / "data" / "coco" / "val" / "val_annotations.json"
        self.test_ann_file = self.base_dir / "data" / "coco" / "test" / "test_annotations.json"
        
        self.train_img_dir = self.base_dir / "data" / "images" / "train"
        self.val_img_dir = self.base_dir / "data" / "images" / "val"
        self.test_img_dir = self.base_dir / "data" / "images" / "test"
        
        # Rutas alternativas (fallback)
        self.train_ann_file_alt = self.base_dir / "data" / "groundtruth" / "json" / "big_size" / "train_big_size_A_B_E_K_WH_WB.json"
        self.val_ann_file_alt = self.base_dir / "data" / "groundtruth" / "json" / "big_size" / "val_big_size_A_B_E_K_WH_WB.json"
        self.test_ann_file_alt = self.base_dir / "data" / "groundtruth" / "json" / "big_size" / "test_big_size_A_B_E_K_WH_WB.json"
        
        self.train_img_dir_alt = self.base_dir / "data" / "train"
        self.val_img_dir_alt = self.base_dir / "data" / "val"
        self.test_img_dir_alt = self.base_dir / "data" / "test"
        
        # ============================================================
        # RUTAS DE SALIDA
        # ============================================================
        self.yolo_data_dir = Path("/content/yolo_data")
        self.dataset_yaml_path = self.yolo_data_dir / "dataset.yaml"
        
    def print_config(self):
        """Imprimir configuración actual"""
        print("🔧 Configuración YOLOv8 V1 (Optimizada):")
        print(f"  Modelo: {self.model}")
        print(f"  Tamaño de imagen: {self.image_size}")
        print(f"  Épocas: {self.epochs}")
        print(f"  Batch size: {self.batch_size} (OPTIMIZADO)")
        print(f"  Workers: {self.workers} (OPTIMIZADO)")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Dispositivo: {self.device}")
        print(f"  Mixed precision: {self.fp16}")
        print(f"  Save period: {self.save_period} épocas")
        print(f"  Drive backup: {self.drive_backup_period} épocas")
        print(f"  Resume training: {self.resume_training}")
        print(f"  Clases: {len(self.classes)} especies")
        
    def get_train_args(self, yaml_path):
        """Obtener argumentos de entrenamiento como diccionario"""
        return {
            'data': str(yaml_path),
            'epochs': self.epochs,
            'imgsz': self.image_size,
            'batch': self.batch_size,
            'workers': self.workers,
            'device': self.device,
            'project': self.project,
            'name': self.name,
            'patience': self.patience,
            'save_period': self.save_period,
            'fp16': self.fp16,
            'resume': self.resume_training,
            'lr0': self.learning_rate,
            'amp': self.fp16,
            'save': True,
            'val': True,
            'plots': True,
            'verbose': True,
        }
    
    def detect_data_structure(self):
        """Detectar qué estructura de datos está disponible"""
        if self.train_ann_file.exists() and self.train_img_dir.exists():
            return "standard", self.train_ann_file, self.val_ann_file, self.test_ann_file
        elif self.train_ann_file_alt.exists() and self.train_img_dir_alt.exists():
            return "groundtruth", self.train_ann_file_alt, self.val_ann_file_alt, self.test_ann_file_alt
        else:
            return "legacy", None, None, None


# Instancia global de configuración
yolo_config = YOLOConfig()

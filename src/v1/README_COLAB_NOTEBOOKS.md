# 📓 Notebooks de Google Colab - Guía de Uso

Este directorio contiene notebooks optimizados para Google Colab que permiten entrenar los modelos de detección de vida silvestre aérea directamente en la nube.

## 📋 Notebooks Disponibles

### 1. **`train_cascade_rcnn_colab.ipynb`**
- **Modelo**: Cascade R-CNN con MMDetection
- **Backbones**: Swin-T, ResNeXt
- **Características**: 
  - Configuración automática de GPU
  - Visualización de curvas de entrenamiento
  - Inferencia en tiempo real
  - Exportación de resultados

### 2. **`train_yolov8_colab.ipynb`**
- **Modelo**: YOLOv8 con Ultralytics
- **Variantes**: YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **Características**:
  - Entrenamiento optimizado para Colab
  - Mixed precision (FP16)
  - Exportación a ONNX
  - Visualización de resultados

### 3. **`train_deformable_detr_colab.ipynb`**
- **Modelo**: Deformable DETR con MMDetection
- **Arquitectura**: Transformer-based
- **Características**:
  - Detección automática de GPU
  - Configuración optimizada para Colab
  - Visualización de resultados

### 4. **`complete_pipeline_colab.ipynb`**
- **Pipeline**: Completo automatizado
- **Incluye**: Calidad + Aumentación + Entrenamiento
- **Características**:
  - Ejecución secuencial de todo el pipeline
  - Comparación de modelos
  - Análisis de resultados

## 🚀 Cómo Usar los Notebooks

### Paso 1: Preparar los Datos
1. Sube tus datos a Google Drive
2. Organiza la estructura de directorios:
   ```
   /content/drive/MyDrive/aerial-wildlife-count/
   ├── data/
   │   ├── train/
   │   ├── val/
   │   ├── test/
   │   └── groundtruth/
   │       └── json/
   └── results/
   ```

### Paso 2: Abrir en Colab
1. Ve a [Google Colab](https://colab.research.google.com/)
2. Sube el notebook deseado
3. Conecta a una GPU (Runtime → Change runtime type → GPU)

### Paso 3: Configurar Rutas
Ajusta las rutas en la celda de configuración:
```python
DRIVE_DATA_PATH = '/content/drive/MyDrive/aerial-wildlife-count/data'
```

### Paso 4: Ejecutar
1. Ejecuta las celdas en orden
2. Monitorea el progreso
3. Descarga los resultados

## ⚙️ Configuración Recomendada

### Para Colab Gratuito:
- **Modelo**: YOLOv8s (más ligero)
- **Batch Size**: 8-16
- **Épocas**: 50-100
- **Image Size**: 640

### Para Colab Pro/Pro+:
- **Modelo**: YOLOv8l o Cascade R-CNN
- **Batch Size**: 16-32
- **Épocas**: 100-200
- **Image Size**: 896

## 📊 Estructura de Datos Esperada

### Datos Originales:
```
data/
├── train/                    # Imágenes de entrenamiento
├── val/                      # Imágenes de validación
├── test/                     # Imágenes de prueba
└── groundtruth/
    └── json/
        └── big_size/
            ├── train_big_size_A_B_E_K_WH_WB.json
            ├── val_big_size_A_B_E_K_WH_WB.json
            └── test_big_size_A_B_E_K_WH_WB.json
```

### Datos Procesados (opcional):
```
data/outputs/
└── mirror_clean/
    ├── train_joined/
    │   ├── train_joined.json
    │   └── images/
    ├── train_final/
    │   ├── train_final.json
    │   └── images/
    └── reports/
```

## 🔧 Configuración de Entrenamiento

### YOLOv8:
```python
TRAINING_CONFIG = {
    'model': 'yolov8s.pt',
    'image_size': 640,
    'epochs': 100,
    'batch_size': 16,
    'device': 0,
    'fp16': True,
}
```

### Cascade R-CNN:
```python
TRAINING_CONFIG = {
    'backbone': 'swin_t',
    'image_size': 896,
    'epochs': 50,
    'batch_size': 2,
    'learning_rate': 0.0001,
}
```

### Deformable DETR:
```python
TRAINING_CONFIG = {
    'backbone': 'swin_t',
    'image_size': 896,
    'epochs': 50,
    'batch_size': 2,
    'learning_rate': 0.0001,
}
```

## 📈 Monitoreo del Entrenamiento

### Métricas Importantes:
- **Loss**: Pérdida de entrenamiento
- **mAP**: Mean Average Precision
- **Precision**: Precisión por clase
- **Recall**: Recuerdo por clase

### Visualizaciones:
- Curvas de entrenamiento
- Matrices de confusión
- Ejemplos de inferencia
- Distribución de clases

## 💾 Guardar y Descargar Resultados

### Archivos Generados:
- `best_model.pth`: Mejor modelo según mAP
- `latest_model.pth`: Último modelo entrenado
- `config.py`: Configuración del modelo
- `results.csv`: Métricas de entrenamiento
- `confusion_matrix.png`: Matriz de confusión

### Opciones de Guardado:
1. **Google Drive**: Automático
2. **Descarga Local**: Manual
3. **Exportación ONNX**: Para deployment

## 🚨 Solución de Problemas

### Error: "CUDA out of memory"
- Reducir `batch_size`
- Reducir `image_size`
- Usar `fp16=True`

### Error: "No module named 'mmdet'"
- Ejecutar celda de instalación
- Reiniciar runtime

### Error: "File not found"
- Verificar rutas de datos
- Asegurar que Drive esté montado

### Error: "Permission denied"
- Verificar permisos de Drive
- Remontar Drive

## 📊 Comparación de Modelos

### YOLOv8:
- ✅ Rápido entrenamiento
- ✅ Fácil de usar
- ✅ Buen rendimiento
- ❌ Menos preciso que R-CNN

### Cascade R-CNN:
- ✅ Muy preciso
- ✅ Buen para objetos pequeños
- ❌ Lento entrenamiento
- ❌ Requiere más memoria

### Deformable DETR:
- ✅ Arquitectura moderna
- ✅ Buen rendimiento
- ❌ Lento entrenamiento
- ❌ Requiere más memoria

## 🎯 Recomendaciones

### Para Principiantes:
1. Usar `train_yolov8_colab.ipynb`
2. Empezar con YOLOv8s
3. Usar configuración por defecto

### Para Experimentados:
1. Usar `complete_pipeline_colab.ipynb`
2. Comparar múltiples modelos
3. Ajustar hiperparámetros

### Para Producción:
1. Entrenar con más épocas
2. Usar validación cruzada
3. Exportar a ONNX

## 📞 Soporte

Si encuentras problemas:
1. Verificar configuración de GPU
2. Revisar rutas de datos
3. Consultar logs de error
4. Verificar memoria disponible

## 🔄 Actualizaciones

Los notebooks se actualizan regularmente para:
- Mejor compatibilidad con Colab
- Optimizaciones de rendimiento
- Nuevas características
- Corrección de bugs

# Pipeline de Procesamiento de Datos - Guía de Uso

Este directorio contiene los pipelines ejecutables de procesamiento de datos, creados a partir de los notebooks originales. Los pipelines están diseñados para ser ejecutados desde línea de comandos y se integran automáticamente con los scripts de entrenamiento.

## 📁 Archivos del Pipeline

### Pipelines Principales
- **`quality_pipeline.py`** - Pipeline de verificación y limpieza de calidad de datasets
- **`augment_pipeline.py`** - Pipeline de aumentación de datos con balanceo adaptativo
- **`main_pipeline.py`** - Pipeline principal que ejecuta todo el flujo completo

### Utilidades
- **`pipeline_utils.py`** - Utilidades para integración con scripts de entrenamiento
- **`test_pipeline_integration.py`** - Script de prueba para verificar la integración

### Scripts de Entrenamiento Actualizados
- **`train_cascade_rcnn.py`** - Entrenamiento de Cascade R-CNN (actualizado)
- **`train_deformable_detr.py`** - Entrenamiento de Deformable DETR (actualizado)
- **`train_yolov8.py`** - Entrenamiento de YOLOv8 (actualizado)

### Configuraciones
- **`quality_config.yaml`** - Configuración del pipeline de calidad
- **`augmentation_config.yaml`** - Configuración del pipeline de aumentación

## 🚀 Uso Rápido

### 1. Ejecutar Pipeline Completo
```bash
# Ejecutar todo el pipeline (calidad + aumentación)
python main_pipeline.py --config quality_config.yaml --augment-config augmentation_config.yaml

# Solo pipeline de calidad
python main_pipeline.py --config quality_config.yaml --skip-augment

# Ejecutar etapas específicas
python main_pipeline.py --config quality_config.yaml --stages quality,augment
```

### 2. Ejecutar Pipelines Individuales
```bash
# Pipeline de calidad
python quality_pipeline.py --config quality_config.yaml --verbose

# Pipeline de aumentación
python augment_pipeline.py --config augmentation_config.yaml --split train_joined
```

### 3. Entrenar Modelos con Pipeline Automático
```bash
# Cascade R-CNN con pipeline automático
python train_cascade_rcnn.py --auto-pipeline --dataset-type auto

# YOLOv8 con pipeline automático
python train_yolov8.py --auto-pipeline --dataset-type augmented

# Deformable DETR con pipeline automático
python train_deformable_detr.py --auto-pipeline --dataset-type quality
```

## 📋 Opciones de Dataset

Los scripts de entrenamiento ahora soportan diferentes tipos de dataset:

- **`auto`** (por defecto) - Detecta automáticamente el mejor dataset disponible
- **`original`** - Usa los datos originales sin procesar
- **`quality`** - Usa datos procesados por el pipeline de calidad
- **`augmented`** - Usa datos aumentados (preferido para entrenamiento)

## 🔧 Configuración

### Pipeline de Calidad (`quality_config.yaml`)
```yaml
# Directorios de salida (rutas relativas desde src/v1/)
output_dir: "../../data/outputs"

# Directorios de imágenes
image_dirs:
  train: "../../data/train"
  val: "../../data/val"
  test: "../../data/test"

# Archivos de anotaciones
splits:
  train: "../../data/groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json"
  val: "../../data/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json"
  test: "../../data/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json"

# Clases del dataset
classes: ["A", "B", "E", "K", "WH", "WB"]
```

### Pipeline de Aumentación (`augmentation_config.yaml`)
```yaml
# Directorio de salida
output_dir: "../../data/outputs"

# Rutas de datos limpios
paths:
  clean_dir: "../../data/outputs/mirror_clean/train_joined/images"
  clean_json: "../../data/outputs/mirror_clean/train_joined/train_joined.json"

# Configuración de aumentación
augmentations:
  enabled: true
  mode: "over"  # over, under, none
  proportion: 0.25
  tolerance: 0.10
```

## 🧪 Pruebas

### Ejecutar Todas las Pruebas
```bash
python test_pipeline_integration.py --test-all
```

### Pruebas Específicas
```bash
# Probar detección de datasets
python test_pipeline_integration.py --test-detection

# Probar configuración de entrenamiento
python test_pipeline_integration.py --test-config

# Probar inicialización de pipelines
python test_pipeline_integration.py --test-pipeline
```

## 📊 Flujo de Trabajo Completo

### 1. Preparación de Datos
```bash
# Ejecutar pipeline completo
python main_pipeline.py --config quality_config.yaml --augment-config augmentation_config.yaml
```

### 2. Entrenamiento de Modelos
```bash
# Entrenar todos los modelos con datos aumentados
python train_cascade_rcnn.py --auto-pipeline --dataset-type augmented --epochs 50
python train_deformable_detr.py --auto-pipeline --dataset-type augmented --epochs 50
python train_yolov8.py --auto-pipeline --dataset-type augmented --epochs 50
```

### 3. Verificación
```bash
# Verificar que todo funciona correctamente
python test_pipeline_integration.py --test-all
```

## 🔍 Detección Automática de Datasets

El sistema detecta automáticamente el mejor dataset disponible en este orden:

1. **Dataset aumentado final** (`train_final.json`)
2. **Dataset aumentado rebalanceado** (`train_rebalance_1.json`)
3. **Dataset aumentado proporcional** (`train_prop.json`)
4. **Dataset de calidad unificado** (`train_joined.json`)
5. **Dataset de calidad individual** (`train_clean.json`)
6. **Dataset listo para entrenamiento** (`train_final.json` en training_ready/)
7. **Dataset original** (fallback)

## 📁 Estructura de Salida

```
data/outputs/
├── mirror_clean/
│   ├── train_joined/          # Dataset unificado de calidad
│   ├── train_prop/            # Dataset con aumentación proporcional
│   ├── train_rebalance_1/     # Dataset con rebalanceo
│   ├── train_zoom/            # Dataset con zoom focal
│   └── train_final/           # Dataset final aumentado
├── reports/                   # Reportes y estadísticas
└── training_ready/            # Dataset listo para entrenamiento
```

## ⚠️ Notas Importantes

1. **Rutas Relativas**: Todos los archivos de configuración usan rutas relativas desde `src/v1/`
2. **Dependencias**: Asegúrate de tener instaladas todas las dependencias necesarias
3. **Memoria**: Los pipelines pueden usar mucha memoria, ajusta `num_workers` según tu sistema
4. **Espacio en Disco**: El procesamiento puede generar muchos archivos, asegúrate de tener espacio suficiente

## 🐛 Solución de Problemas

### Error: "No suitable dataset found"
- Verifica que los archivos de configuración existan
- Ejecuta el pipeline de calidad primero
- Revisa las rutas en los archivos de configuración

### Error: "Pipeline failed"
- Revisa los logs en `data/outputs/quality_check.log`
- Verifica que los datos originales existan
- Asegúrate de tener permisos de escritura en el directorio de salida

### Error: "Import failed"
- Verifica que estés ejecutando desde el directorio `src/v1/`
- Instala las dependencias faltantes: `pip install -r requirements.txt`

## 📞 Soporte

Si encuentras problemas:
1. Ejecuta las pruebas: `python test_pipeline_integration.py --test-all`
2. Revisa los logs de error
3. Verifica la configuración de rutas
4. Consulta la documentación de los notebooks originales

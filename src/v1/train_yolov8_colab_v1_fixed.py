#!/usr/bin/env python
# coding: utf-8

# # 🚀 Entrenamiento YOLOv8 en Google Colab - V1 (Optimizado)
# 
# Este notebook entrena un modelo YOLOv8 con Ultralytics en Google Colab para detección de vida silvestre aérea.
# 
# ## 🔥 **MEJORAS V1:**
# - ✅ **Guardado automático en Drive** cada X épocas
# - ✅ **Recuperación de entrenamiento** interrumpido
# - ✅ **Configuración optimizada** para velocidad
# - ✅ **Monitoreo en tiempo real** del progreso
# - ✅ **Backup automático** de checkpoints
# 
# ## 📋 Características
# - **Modelo**: YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
# - **Detección automática de GPU**
# - **Conversión automática COCO a YOLO**
# - **Visualización de resultados**
# - **Exportación a ONNX/TorchScript**
# - **Análisis de métricas detallado**
# 
# ## 🎯 Clases de Animales
# - Buffalo
# - Elephant  
# - Kob
# - Alcelaphinae
# - Warthog
# - Waterbuck
# 
# ## ⚙️ Configuración por Defecto
# - **Modelo**: YOLOv8s (balanceado entre velocidad y precisión)
# - **Tamaño de imagen**: 640x640
# - **Épocas**: 100
# - **Batch size**: 16
# - **Optimizador**: AdamW
# - **Mixed Precision**: Habilitado
# 
# ## 🔬 Ventajas de YOLOv8
# - **Rápido**: Entrenamiento e inferencia eficientes
# - **Preciso**: Mejor rendimiento que versiones anteriores
# - **Fácil de usar**: API simple de Ultralytics
# - **Flexible**: Múltiples tamaños de modelo

# ## 🔧 Instalación de Dependencias

# ## 📦 Importar Librerías

# In[ ]:


# ============================================================
# VARIABLES GLOBALES PARA BACKUP AUTOMÁTICO
# ============================================================

# Variables globales para el sistema de backup
backup_thread_running = False
backup_thread = None

print("✅ Variables globales de backup inicializadas")


# ## 📁 Configuración de Datos

# ## 📊 Análisis de Datos
# 

# ## 🔧 Funciones de Backup Mejoradas
# 

# ## 🚀 Inicialización del Sistema de Backup
# 

# ## ⚙️ Configuración del Entrenamiento

# ## 🔄 Conversión de Datos COCO a YOLO
# 

# ## 📊 Monitoreo en Tiempo Real
# 

# In[ ]:


# Funciones de monitoreo en tiempo real
def monitor_training_progress():
    """Monitorear el progreso del entrenamiento en tiempo real"""
    try:
        results_dir = Path(f"{yolo_config.project}/{yolo_config.name}")
        
        if not results_dir.exists():
            print("❌ Directorio de resultados no encontrado")
            return
        
        # Verificar archivos de resultados
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            if not df.empty:
                latest_epoch = df.iloc[-1]
                print(f"📊 Progreso actual:")
                print(f"  Época: {latest_epoch.get('epoch', 'N/A')}")
                print(f"  mAP: {latest_epoch.get('metrics/mAP50(B)', 'N/A'):.4f}")
                print(f"  Loss: {latest_epoch.get('train/box_loss', 'N/A'):.4f}")
                print(f"  Val Loss: {latest_epoch.get('val/box_loss', 'N/A'):.4f}")
        
        # Verificar checkpoints
        weights_dir = results_dir / "weights"
        if weights_dir.exists():
            checkpoints = list(weights_dir.glob("*.pt"))
            print(f"📁 Checkpoints disponibles: {len(checkpoints)}")
            for ckpt in checkpoints:
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
        
        # Verificar backups en Drive
        backup_dir = Path(yolo_config.drive_backup_dir)
        if backup_dir.exists():
            backups = list(backup_dir.glob("epoch_*"))
            print(f"💾 Backups en Drive: {len(backups)}")
        
    except Exception as e:
        print(f"❌ Error en monitoreo: {e}")

def get_training_status():
    """Obtener estado actual del entrenamiento"""
    try:
        results_dir = Path(f"{yolo_config.project}/{yolo_config.name}")
        
        if not results_dir.exists():
            return "No iniciado"
        
        # Verificar si hay resultados
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            if not df.empty:
                latest_epoch = df.iloc[-1]['epoch']
                total_epochs = yolo_config.epochs
                progress = (latest_epoch / total_epochs) * 100
                return f"En progreso: {latest_epoch}/{total_epochs} épocas ({progress:.1f}%)"
        
        return "Iniciando"
        
    except Exception as e:
        return f"Error: {e}"

def estimate_remaining_time():
    """Estimar tiempo restante de entrenamiento"""
    try:
        results_dir = Path(f"{yolo_config.project}/{yolo_config.name}")
        results_csv = results_dir / "results.csv"
        
        if not results_csv.exists():
            return "No disponible"
        
        import pandas as pd
        df = pd.read_csv(results_csv)
        
        if len(df) < 2:
            return "Calculando..."
        
        # Calcular tiempo promedio por época basado en el número de épocas
        # Asumir que cada época toma aproximadamente el mismo tiempo
        current_epoch = df.iloc[-1]['epoch']
        remaining_epochs = yolo_config.epochs - current_epoch
        
        # Estimación simple: asumir 2-5 minutos por época
        estimated_minutes = remaining_epochs * 3  # 3 minutos promedio por época
        
        if estimated_minutes < 60:
            return f"Tiempo estimado restante: {estimated_minutes:.0f} minutos"
        else:
            hours = estimated_minutes / 60
            return f"Tiempo estimado restante: {hours:.1f} horas"
        
    except Exception as e:
        return f"Error: {e}"

# Función para mostrar estado completo
def show_training_status():
    """Mostrar estado completo del entrenamiento"""
    print("=" * 60)
    print("📊 ESTADO DEL ENTRENAMIENTO YOLOv8 V1")
    print("=" * 60)
    print(f"Estado: {get_training_status()}")
    print(f"Tiempo restante: {estimate_remaining_time()}")
    print()
    monitor_training_progress()
    print("=" * 60)

print("✅ Funciones de monitoreo cargadas")


# In[ ]:


# Ejecutar monitoreo en tiempo real
print("📊 Sistema de monitoreo en tiempo real disponible")
print("💡 Usa 'show_training_status()' para ver el progreso durante el entrenamiento")
print("💡 Usa 'monitor_training_progress()' para ver detalles del progreso")
print("💡 Usa 'get_training_status()' para obtener el estado actual")
print("💡 Usa 'estimate_remaining_time()' para estimar tiempo restante")
print()
print("✅ Funciones de monitoreo cargadas y listas para usar")


# ## 🚀 Instrucciones de Uso - V1 Optimizado
# 
# ### 📋 **Antes de Ejecutar:**
# 1. **Montar Google Drive** (se hace automáticamente)
# 2. **Verificar que tienes Colab Pro/Pro+** para mejor rendimiento
# 3. **Configurar datos** en la estructura correcta
# 
# ### 🔥 **Características V1:**
# - **Backup automático**: Los checkpoints se guardan en Drive cada 10 épocas
# - **Recuperación automática**: Si se interrumpe, reanuda desde el último checkpoint
# - **Configuración optimizada**: Batch size 32, workers 8 para mayor velocidad
# - **Monitoreo en tiempo real**: Usa `show_training_status()` para ver progreso
# 
# ### 📊 **Comandos Útiles Durante el Entrenamiento:**
# ```python
# # Ver estado del entrenamiento
# show_training_status()
# 
# # Hacer backup manual
# backup_to_drive()
# 
# # Verificar backups en Drive
# monitor_training_progress()
# ```
# 
# ### ⚠️ **Importante:**
# - **NO cierres la pestaña** de Colab durante el entrenamiento
# - **Los backups se hacen automáticamente** cada 10 épocas
# - **Si se interrumpe**, simplemente ejecuta de nuevo y reanudará automáticamente
# - **Los checkpoints se guardan en**: `/content/drive/MyDrive/aerial-wildlife-count-results/yolo_v1/`
# 
# ### 🎯 **Optimizaciones de Velocidad:**
# - **Batch size**: 32 (vs 16 original)
# - **Workers**: 8 (vs 4 original)  
# - **Save period**: 5 épocas (vs 10 original)
# - **Mixed precision**: Habilitado
# - **Early stopping**: 10 épocas de paciencia
# 

# ## 🚀 Entrenamiento Optimizado con Backup Automático
# 

# ## 🎉 ¡Entrenamiento Completado! - V1 Optimizado
# 
# ### 📋 Resumen del Entrenamiento V1
# - **Modelo**: YOLOv8 con configuración {yolo_config.model}
# - **Tamaño de imagen**: {yolo_config.image_size}x{yolo_config.image_size}
# - **Épocas**: {yolo_config.epochs}
# - **Early Stopping**: {yolo_config.patience} épocas de paciencia
# - **Batch size optimizado**: {yolo_config.batch_size}
# - **Workers optimizados**: {yolo_config.workers}
# - **Clases detectadas**: {len(yolo_config.classes)} especies de animales
# 
# ### 🔥 **Mejoras V1 Implementadas:**
# - ✅ **Guardado automático en Drive** cada {yolo_config.drive_backup_period} épocas
# - ✅ **Recuperación automática** de entrenamiento interrumpido
# - ✅ **Configuración optimizada** para mayor velocidad
# - ✅ **Monitoreo en tiempo real** del progreso
# - ✅ **Backup automático** de checkpoints
# 
# ### 📊 Próximos Pasos
# 1. **Evaluar métricas**: Revisar mAP, precision, recall
# 2. **Ajustar hiperparámetros**: Si es necesario mejorar el rendimiento
# 3. **Exportar modelo**: Convertir a ONNX o TorchScript para deployment
# 4. **Probar en nuevas imágenes**: Validar en datos no vistos
# 
# ### 🔧 Configuración Personalizada
# Para modificar parámetros, edita la clase `YOLOConfig` en la celda de configuración:
# - Cambiar modelo: `"yolov8s.pt"`, `"yolov8m.pt"`, `"yolov8l.pt"`, `"yolov8x.pt"`
# - Ajustar épocas: `epochs = 200`
# - Modificar tamaño de imagen: `image_size = 1024`
# - Cambiar batch size: `batch_size = 32`
# 
# ### 📚 Recursos Adicionales
# - [Documentación Ultralytics](https://docs.ultralytics.com/)
# - [YOLOv8 Paper](https://arxiv.org/abs/2305.09972)
# - [GitHub Ultralytics](https://github.com/ultralytics/ultralytics)
# 

# ## 📊 Visualización de Resultados

# ## 🔍 Inferencia y Pruebas

# ## 📊 Cálculo de Métricas de Clasificación
# 

# ## 💾 Guardar y Exportar Modelo

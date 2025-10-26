# Augmentation & Quality Control Pipelines

Este módulo forma parte del proyecto **Dense Herd Wildlife Detection** y contiene dos notebooks principales enfocados en el procesamiento y validación del dataset utilizado para el entrenamiento del modelo.

## `augment.ipynb`
Pipeline completo de **augmentación de datos**, encargado de ampliar y balancear el conjunto de imágenes de entrenamiento.

Incluye:
- Aumento proporcional y rebalanceo adaptativo.
- Transformaciones visuales conservadoras (Albumentations).
- Registro detallado del proceso y verificación final del dataset.
- Configuración centralizada mediante `augmentation_config.yaml`.

## `quality.ipynb`
Pipeline de **control de calidad posterior a la augmentación**, encargado de verificar la consistencia y limpieza del dataset final.

Incluye:
- Detección y eliminación de imágenes con exceso de anotaciones.
- Auditoría de integridad de imágenes y anotaciones.
- Gráficas de distribución por clase, anotaciones por imagen y áreas de bounding boxes.
- Configuración controlada mediante `quality_config.yaml`.

# 📖 Manual de Usuario - Wildlife Vision

Bienvenido al manual de usuario de **Wildlife Vision**, el sistema de detección y conteo automático de fauna africana.

## 📋 Tabla de Contenidos

- [Introducción](#introducción)
- [Acceso a la Aplicación](#acceso-a-la-aplicación)
- [Interfaz de Usuario](#interfaz-de-usuario)
- [Guía Paso a Paso](#guía-paso-a-paso)
- [Interpretación de Resultados](#interpretación-de-resultados)
- [Descarga de Datos](#descarga-de-datos)
- [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Introducción

Wildlife Vision es una herramienta de inteligencia artificial que permite detectar y contar automáticamente 6 especies de mamíferos africanos en imágenes aéreas:

| Especie | Emoji | Nombre Común |
|---------|-------|--------------|
| Buffalo | 🦬 | Búfalo africano |
| Elephant | 🐘 | Elefante africano |
| Kob | 🦌 | Antílope Kob |
| Topi | 🫎 | Antílope Topi |
| Warthog | 🐗 | Jabalí verrugoso |
| Waterbuck | 🦌 | Antílope acuático |

---

## Acceso a la Aplicación

### Opción 1: Aplicación en Producción (Recomendado)

Acceda directamente a la aplicación desplegada:

🌐 **URL**: [https://wildlife.vision](https://wildlife.vision)

### Opción 2: Ejecución Local

Si tiene el proyecto instalado localmente:

```bash
python app.py
```

Luego abra su navegador en: `http://localhost:7860`

---

## Interfaz de Usuario

La interfaz de Wildlife Vision está diseñada para ser intuitiva y fácil de usar.

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│  👁️ Wildlife Vision                         ● Modelo activo │
├─────────────────────────────────────────────────────────────┤
│  ℹ️ Información del Modelo ▼                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📷 Imagen Aérea          │    🎯 Detecciones               │
│  ┌───────────────────┐    │    ┌───────────────────┐        │
│  │                   │    │    │                   │        │
│  │   Drop Image      │    │    │   Imagen con      │        │
│  │   Here            │    │    │   detecciones     │        │
│  │                   │    │    │                   │        │
│  └───────────────────┘    │    └───────────────────┘        │
│                           │                                 │
│  [▶ Ejecutar Detección]   │                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  📊 Conteo por Especie                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 🦬 Buffalo    ████████░░░░░░░░░░░░░░░░░░░░░░░░  5   │    │
│  │ 🐘 Elephant   ████████████████████░░░░░░░░░░░░  12  │    │
│  │ 🦌 Kob        ██████████████████████████████░░  25  │    │
│  │ ...                                                 │    │
│  │ ┌─────────────────────────────────────────────┐     │    │
│  │ │ Total detectado                          42 │     │    │
│  │ └─────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  [📥 Conteos (CSV)]    [📥 Detecciones (CSV)]               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Elementos de la Interfaz

1. **Header**: Muestra el logo y el estado del modelo (● Modelo activo)
2. **Panel de Información**: Detalles técnicos del modelo (expandible)
3. **Área de Imagen Aérea**: Zona para subir la imagen a analizar
4. **Área de Detecciones**: Muestra la imagen con las detecciones marcadas
5. **Conteo por Especie**: Barras de progreso con el conteo de cada especie
6. **Botones de Descarga**: Exportar resultados en formato CSV

---

## Guía Paso a Paso

### Paso 1: Subir una Imagen

Hay tres formas de subir una imagen:

#### Opción A: Arrastrar y Soltar
1. Localice la imagen en su explorador de archivos
2. Arrástrela directamente sobre el área "Drop Image Here"
3. Suelte el botón del ratón

#### Opción B: Hacer Clic para Seleccionar
1. Haga clic en el área "Click to Upload"
2. Se abrirá un diálogo de selección de archivos
3. Navegue hasta su imagen y selecciónela
4. Haga clic en "Abrir"

#### Opción C: Pegar desde Portapapeles
1. Copie una imagen (Ctrl+C o Cmd+C)
2. Haga clic en el icono de portapapeles (📋)
3. La imagen se cargará automáticamente

### Paso 2: Ejecutar la Detección

1. Una vez cargada la imagen, verá una vista previa
2. Haga clic en el botón dorado **"▶ Ejecutar Detección"**
3. Espere mientras el modelo procesa la imagen
   - El tiempo depende del tamaño de la imagen y del hardware
   - Típicamente: 5-30 segundos

### Paso 3: Revisar los Resultados

Una vez completado el procesamiento:

1. **Imagen Anotada**: En el panel derecho verá la imagen original con puntos rojos marcando cada detección
2. **Conteo por Especie**: Debajo verá barras de progreso mostrando:
   - Nombre de cada especie con emoji
   - Barra visual proporcional al conteo
   - Número exacto de individuos detectados
3. **Total**: Al final, un recuadro dorado muestra el total de animales

### Paso 4: Descargar Resultados

Para guardar los resultados:

1. **Conteos (CSV)**: Haga clic en "📥 Conteos (CSV)"
   - Descarga un archivo con el resumen por especie
   
2. **Detecciones (CSV)**: Haga clic en "📥 Detecciones (CSV)"
   - Descarga un archivo con las coordenadas de cada detección

---

## Interpretación de Resultados

### Imagen Anotada

- **Puntos Rojos**: Cada punto indica un animal detectado
- **Posición**: El centro del punto corresponde al centroide estimado del animal

### Archivo de Conteos (species_counts.csv)

```csv
Especie,Conteo
buffalo,5
elephant,12
kob,25
topi,3
warthog,2
waterbuck,8
Total,55
```

### Archivo de Detecciones (detections.csv)

```csv
images,loc,labels,scores,species
imagen.jpg,"(1234, 567)",2,0.95,elephant
imagen.jpg,"(890, 123)",1,0.87,buffalo
...
```

| Columna | Descripción |
|---------|-------------|
| `images` | Nombre de la imagen procesada |
| `loc` | Coordenadas (x, y) de la detección |
| `labels` | ID numérico de la especie |
| `scores` | Confianza de la detección (0-1) |
| `species` | Nombre de la especie |

---

## Preguntas Frecuentes

### ¿Qué formatos de imagen son compatibles?

- **Formatos soportados**: JPG, JPEG, PNG, TIFF, BMP
- **Resolución recomendada**: 1000x1000 píxeles o superior
- **Tamaño máximo**: 50 MB

### ¿Por qué no se detectan animales en mi imagen?

Posibles razones:
1. La imagen no es aérea (vista desde arriba)
2. Los animales están muy pequeños o muy grandes
3. La calidad de la imagen es baja
4. La especie no está entre las 6 detectables

### ¿Cuánto tiempo tarda el procesamiento?

| Tamaño de Imagen | Tiempo (GPU) | Tiempo (CPU) |
|------------------|--------------|--------------|
| 1000x1000 | ~5 segundos | ~30 segundos |
| 2000x2000 | ~10 segundos | ~60 segundos |
| 4000x4000 | ~30 segundos | ~3 minutos |

### ¿Qué significan las métricas del modelo?

- **F1-score (0.84)**: Equilibrio entre precisión y recall
- **Precision (0.84)**: % de detecciones correctas
- **Recall (0.84)**: % de animales reales detectados
- **MAE (1.80)**: Error promedio en el conteo
- **RMSE (3.49)**: Error cuadrático medio

### ¿Puedo usar la aplicación en mi teléfono?

Sí, la interfaz es responsive y funciona en dispositivos móviles. Sin embargo, se recomienda usar una pantalla más grande para mejor visualización.

### ¿Los datos son privados?

- Las imágenes se procesan en tiempo real
- No se almacenan permanentemente en el servidor
- Los archivos temporales se eliminan después del procesamiento

---

## Soporte

Si tiene problemas o preguntas adicionales:

- **GitHub Issues**: [Reportar problema](https://github.com/jaimevera1107/aerial-wildlife-count/issues)
- **Email**: proyecto-guacamaya@uniandes.edu.co

---

<div align="center">

**Wildlife Vision - Manual de Usuario v1.0**

*Universidad de los Andes - Maestría en Inteligencia Artificial*

</div>


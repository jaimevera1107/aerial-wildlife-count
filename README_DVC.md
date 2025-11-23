# Guía rápida de DVC para este proyecto

Este repositorio ya está integrado con DVC para versionar:

- Carpeta de datos: `data/`
- Modelos principales:
  - `modelos/herdnet_best.pth`
  - `resources/models/herdnet_best.pth`

Git rastrea únicamente los archivos `.dvc` y la metadata; los archivos grandes se gestionan vía DVC.

## Remote configurado

Hay un remote SSH definido como **por defecto**:

- Nombre: `storage`
- URL: `ssh://dvc@rinconseguro.com:33/share/DVC`

La autenticación se hace por SSH (contraseña o llave, según la configuración local).  
La contraseña **no** se guarda en el repositorio.

## Comandos básicos

### 1. Preparar entorno

```bash
pip install dvc
```

### 2. Obtener datos y modelos (clon nuevo)

```bash
git clone <URL_DEL_REPO>
cd aerial-wildlife-count-main
dvc pull
```

Esto descargará:

- Contenido de `data/`
- `modelos/herdnet_best.pth`
- `resources/models/herdnet_best.pth`

### 3. Subir cambios de datos/modelos

Tras modificar datos o regenerar modelos:

```bash
# Asegurarse de que los .dvc se han actualizado
# (DVC lo hace automáticamente al sobreescribir los archivos seguidos por DVC)

dvc status            # Opcional, ver qué ha cambiado
dvc push              # Sube los datos/modelos al remote SSH

git add .
git commit -m "Update data/models via DVC"
git push
```

## Notas

- No añadas manualmente a Git los archivos grandes (`.pth`, imágenes, etc.) ya controlados por DVC.
- Si creas nuevos artefactos grandes (por ejemplo, otro modelo), añádelos con:

```bash
dvc add ruta/al/archivo_grande
git add ruta/al/archivo_grande.dvc
git commit -m "Track new artifact with DVC"
```

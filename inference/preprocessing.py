import os
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

from animaloc.datasets import CSVDataset
from animaloc.data.transforms import DownSample
from inference.utils_io import mkdir, get_temp_image_path

def build_normalize_transform(mean: list, std: list) -> A.Normalize:
    """
    Construye una transformación de normalización idéntica
    a la utilizada durante el entrenamiento.
    """
    return A.Normalize(mean=mean, std=std, p=1.0)


def build_end_transforms(down_ratio: int = 2):
    """
    Construye el conjunto de transformaciones finales utilizadas
    durante la inferencia.
    """
    return [DownSample(down_ratio=down_ratio, anno_type="point")]


def create_single_image_dataset(
    image_pil,
    mean: list,
    std: list,
    down_ratio: int = 2
):
    """
    Crea un CSVDataset temporal y su DataLoader a partir de una única imagen PIL.
    Guarda la imagen temporalmente en disco (resources/uploads) para la API de HerdNet.

    Retorna
    -------
    dataset : CSVDataset
        Dataset temporal con una sola imagen.
    dataloader : DataLoader
        Cargador de datos correspondiente.
    temp_path : str
        Ruta absoluta de la imagen guardada temporalmente.
    """
    # Crear directorio y archivo temporal
    upload_dir = "resources/uploads"
    mkdir(upload_dir)
    temp_path = get_temp_image_path(upload_dir)
    image_pil.save(temp_path, format="JPEG")

    # Construir DataFrame para CSVDataset
    df = pd.DataFrame({
        "images": [os.path.basename(temp_path)],
        "x": [0],
        "y": [0],
        "labels": [1],
    })

    # Normalización Albumentations
    normalize = A.Normalize(mean=mean, std=std, p=1.0)
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type="point")]

    # Crear dataset y dataloader
    dataset = CSVDataset(
        csv_file=df,
        root_dir=os.path.dirname(temp_path),
        albu_transforms=[normalize],
        end_transforms=end_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return dataset, dataloader, temp_path


def create_single_image_dataset_safe(
    image_pil,
    down_ratio: int = 2,
    patch_size: int = 512,
    overlap: int = 160,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
):
    """
    Prepara una imagen para inferencia con HerdNet, asegurando compatibilidad
    geométrica, cromática y numérica con los parámetros del modelo.

    Esta función reemplaza completamente la antigua create_single_image_dataset
    y evita fallos comunes al recibir imágenes de distinta fuente, tamaño o formato.

    Parámetros
    ----------
    image_pil : PIL.Image
        Imagen de entrada (cualquier resolución o fuente).
    down_ratio : int
        Factor de reducción espacial del modelo (por defecto 2).
    patch_size : int
        Tamaño del parche usado en entrenamiento (por defecto 512).
    overlap : int
        Superposición entre parches (por defecto 160).
    mean, std : tuple
        Parámetros de normalización (por defecto ImageNet).

    Retorna
    -------
    dataset : CSVDataset
        Dataset temporal con una única imagen lista para inferencia.
    dataloader : DataLoader
        Cargador de datos asociado al dataset.
    temp_path : str
        Ruta absoluta del archivo temporal guardado.
    """

    # =======================================================
    # 1. Corrección de orientación y canales
    # =======================================================
    image_pil = ImageOps.exif_transpose(image_pil)
    image_pil = image_pil.convert("RGB")

    # =======================================================
    # 2. Limpieza y validación de valores
    # =======================================================
    arr = np.array(image_pil).astype(np.uint8)
    arr = np.clip(arr, 0, 255)
    image_pil = Image.fromarray(arr)

    # =======================================================
    # 3. Ajuste de resolución compatible con el modelo
    # =======================================================
    w, h = image_pil.size

    # Forzar múltiplos del down_ratio
    new_w = int(np.ceil(w / down_ratio) * down_ratio)
    new_h = int(np.ceil(h / down_ratio) * down_ratio)
    pad_w, pad_h = new_w - w, new_h - h

    if pad_w > 0 or pad_h > 0:
        image_pil = ImageOps.expand(image_pil, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))

    # Forzar múltiplos de patch_size - overlap
    step = patch_size - overlap
    w, h = image_pil.size
    if (w % step != 0) or (h % step != 0):
        new_w = int(np.ceil(w / step) * step)
        new_h = int(np.ceil(h / step) * step)
        image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)

    # =======================================================
    # 4. Guardado temporal
    # =======================================================
    upload_dir = "resources/uploads"
    mkdir(upload_dir)
    temp_path = get_temp_image_path(upload_dir)
    image_pil.save(temp_path, format="JPEG", quality=95)

    # =======================================================
    # 5. DataFrame temporal (estructura compatible con CSVDataset)
    # =======================================================
    df = pd.DataFrame({
        "images": [os.path.basename(temp_path)],
        "x": [0],
        "y": [0],
        "labels": [1],
    })

    # =======================================================
    # 6. Normalización + transformaciones finales
    # =======================================================
    normalize = A.Normalize(mean=mean, std=std, p=1.0)
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type="point")]

    dataset = CSVDataset(
        csv_file=df,
        root_dir=os.path.dirname(temp_path),
        albu_transforms=[normalize],
        end_transforms=end_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # =======================================================
    # 7. Log de diagnóstico
    # =======================================================
    print("[PREPROCESS] Imagen preparada para inferencia:")
    print(f"  - Guardada en: {temp_path}")
    print(f"  - Tamaño final: {image_pil.size}")
    print(f"  - down_ratio:   {down_ratio}")
    print(f"  - patch_size:   {patch_size}")
    print(f"  - overlap:      {overlap}")

    return dataset, dataloader, temp_path

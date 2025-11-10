import os
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader
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

import os
import pandas as pd
from PIL import Image
from animaloc.vizual import draw_points, draw_text
from inference.utils_io import mkdir, save_csv


def draw_detections_on_image(
    image_path: str,
    detections_df: pd.DataFrame,
    output_path: str = None
) -> Image.Image:
    """
    Dibuja puntos visibles sobre una imagen y añade una leyenda
    con el total de detecciones y el desglose por especie.

    Parámetros
    ----------
    image_path : str
        Ruta de la imagen original.
    detections_df : pd.DataFrame
        DataFrame con las detecciones (columnas: x, y, species, scores, etc.).
    output_path : str, opcional
        Ruta donde se guardará la imagen anotada.

    Retorna
    -------
    Image.Image
        Imagen con los puntos y leyenda dibujados.
    """
    img = Image.open(image_path)
    img_cpy = img.copy()

    # Extraer coordenadas (y, x)
    pts = list(detections_df[["y", "x"]].to_records(index=False))
    pts = [(y, x) for y, x in pts]

    # Dibujar puntos sobre la imagen
    output = draw_points(img_cpy, pts, color=(255, 0, 0), size=60)

    # Construir texto de leyenda
    species_counts = detections_df["species"].value_counts().to_dict()
    total = sum(species_counts.values())
    legend = f"Detecciones: {total} | " + ", ".join(
        [f"{sp}: {n}" for sp, n in species_counts.items()]
    )

    # Posicionar texto en parte inferior
    overlay_y = img_cpy.height - int(0.08 * img_cpy.height)
    output = draw_text(
        output,
        text=legend,
        position=(20, overlay_y),
        font_size=int(0.04 * img_cpy.height),
    )

    # Guardar imagen si se especifica ruta de salida
    if output_path:
        mkdir(os.path.dirname(output_path))
        output.save(output_path, quality=95)

    return output


def compute_species_counts(detections_df: pd.DataFrame) -> dict:
    """
    Calcula el número de detecciones por especie.

    Retorna
    -------
    dict
        Diccionario con las especies y sus conteos.
        Retorna un diccionario vacío si no hay detecciones.
    """
    if detections_df.empty:
        return {}
    return detections_df["species"].value_counts().to_dict()


def generate_thumbnails(
    image_path: str,
    detections_df: pd.DataFrame,
    output_dir: str,
    thumb_size: int = 256
) -> None:
    """
    Genera miniaturas recortadas alrededor de cada detección,
    con el nombre de la especie y su puntaje de confianza.

    Parámetros
    ----------
    image_path : str
        Ruta de la imagen original.
    detections_df : pd.DataFrame
        DataFrame con las detecciones.
    output_dir : str
        Directorio donde se guardarán las miniaturas.
    thumb_size : int
        Tamaño (en píxeles) de cada miniatura cuadrada.
    """
    mkdir(output_dir)
    img = Image.open(image_path)
    img_cpy = img.copy()

    sp_score = list(detections_df[["species", "scores"]].to_records(index=False))
    pts = list(detections_df[["y", "x"]].to_records(index=False))

    for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score)):
        off = thumb_size // 2
        coords = (x - off, y - off, x + off, y + off)

        # Recortar miniatura
        thumbnail = img_cpy.crop(coords)

        # Dibujar texto con especie y score
        score = round(score * 100, 1)
        thumbnail = draw_text(
            thumbnail,
            f"{sp} | {score}%",
            position=(10, 5),
            font_size=int(0.08 * thumb_size),
        )

        filename = os.path.basename(image_path)[:-4] + f"_{i}.JPG"
        thumbnail.save(os.path.join(output_dir, filename))


def save_detections(
    detections_df: pd.DataFrame,
    output_dir: str,
    logger=None
) -> str:
    """
    Guarda las detecciones en formato CSV dentro del directorio de salida.

    Parámetros
    ----------
    detections_df : pd.DataFrame
        DataFrame con las detecciones.
    output_dir : str
        Directorio de salida.
    logger : logging.Logger, opcional
        Logger para registrar el proceso.

    Retorna
    -------
    str
        Ruta del archivo CSV guardado.
    """
    mkdir(output_dir)
    csv_path = os.path.join(output_dir, "detections.csv")
    save_csv(detections_df, csv_path, logger)
    return csv_path

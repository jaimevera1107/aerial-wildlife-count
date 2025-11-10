import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from animaloc.models import HerdNet, LossWrapper
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher
from animaloc.eval.metrics import PointsMetrics

from inference.preprocessing import (
    build_normalize_transform,
    build_end_transforms,
    create_single_image_dataset,
)
from inference.postprocessing import (
    compute_species_counts,
    draw_detections_on_image,
    generate_thumbnails,
    save_detections,
)
from inference.utils_io import (
    mkdir,
    get_timestamp_dir,
    init_logger,
    load_yaml_config,
)


class HerdNetInference:
    """
    Clase envoltoria para cargar un modelo HerdNet entrenado y ejecutar
    inferencias sobre imágenes individuales o carpetas completas,
    incluyendo posprocesamiento y exportación de resultados.
    """

    def __init__(self, config_path: str = "resources/configs/default.yaml"):
        self.cfg = load_yaml_config(config_path)
        self.logger = init_logger(self.cfg["paths"]["logs_dir"])

        # Parámetros principales
        self.device = torch.device(
            self.cfg["model"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.patch_size = self.cfg["model"]["patch_size"]
        self.overlap = self.cfg["model"]["overlap"]
        self.down_ratio = self.cfg["model"]["down_ratio"]
        self.save_plots = self.cfg["inference"]["save_plots"]
        self.save_csv = self.cfg["inference"]["save_csv"]
        self.save_thumbnails = self.cfg["inference"]["save_thumbnails"]

        # Directorio de salida
        self.outputs_base = self.cfg["paths"]["outputs_dir"]
        mkdir(self.outputs_base)
        self.output_dir = get_timestamp_dir(self.outputs_base)
        self.logger.info(f"[INIT] Directorio de salida: {self.output_dir}")

        self._load_model()

    # -----------------------------------------------------------
    def _load_model(self):
        """
        Carga el modelo HerdNet desde el archivo .pth y lo prepara para inferencia.
        """
        pth_path = self.cfg["model"]["path"]
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"[ERROR] No se encontró el modelo → {pth_path}")

        checkpoint = torch.load(pth_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.num_classes = len(self.classes) + 1
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

        model = HerdNet(num_classes=self.num_classes, pretrained=False)
        self.model = LossWrapper(model, [])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(
            f"[MODEL] Modelo HerdNet cargado ({self.num_classes} clases) desde {pth_path}"
        )

    # -----------------------------------------------------------
    def infer_single(self, image_pil):
        """
        Ejecuta la inferencia sobre una imagen PIL.

        Retorna
        -------
        annotated_image : PIL.Image
            Imagen anotada con las detecciones.
        counts_per_species : dict
            Diccionario con el conteo por especie.
        """
        dataset, dataloader, temp_path = create_single_image_dataset(
            image_pil, mean=self.mean, std=self.std, down_ratio=self.down_ratio
        )

        stitcher = HerdNetStitcher(
            model=self.model,
            size=(self.patch_size, self.patch_size),
            overlap=self.overlap,
            down_ratio=self.down_ratio,
            up=True,
            reduction="mean",
            device_name=self.device,
        )

        metrics = PointsMetrics(5, num_classes=self.num_classes)
        evaluator = HerdNetEvaluator(
            model=self.model,
            dataloader=dataloader,
            metrics=metrics,
            device_name=self.device,
            stitcher=stitcher,
            work_dir=self.output_dir,
            header="[INFERENCE]",
        )

        self.logger.info("[RUN] Iniciando inferencia sobre imagen individual...")
        evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)

        detections = evaluator.detections.dropna().copy()
        detections["species"] = detections["labels"].map(self.classes)

        # Guardar CSV con detecciones
        if self.save_csv:
            save_detections(detections, self.output_dir, self.logger)

        counts = compute_species_counts(detections)
        self.logger.info(f"[COUNTS] {counts}")

        # Dibujar detecciones sobre la imagen original
        annotated_image = draw_detections_on_image(
            image_path=temp_path,
            detections_df=detections,
        )

        # Generar miniaturas si está habilitado
        if self.save_thumbnails:
            thumbs_dir = os.path.join(self.output_dir, "thumbnails")
            generate_thumbnails(temp_path, detections, thumbs_dir)

        # Limpieza del archivo temporal
        try:
            os.remove(temp_path)
            self.logger.info(f"[CLEANUP] Archivo temporal eliminado: {temp_path}")
        except Exception as e:
            self.logger.warning(f"[CLEANUP] No se pudo eliminar el archivo temporal: {e}")

        return annotated_image, counts

import warnings
import gradio as gr
from PIL import Image
import pandas as pd
import os

from inference.herdnet_infer import HerdNetInference
from inference.utils_io import load_yaml_config, mkdir
from animaloc.utils.seed import set_seed

# Ignorar advertencias no críticas
warnings.filterwarnings(
    "ignore",
    message="Got processor for keypoints, but no transform to process it",
)

# Fijar semilla para reproducibilidad
set_seed(9292)

# ===============================================================
# Configuración inicial
# ===============================================================
CONFIG_PATH = "resources/configs/default.yaml"
cfg = load_yaml_config(CONFIG_PATH)
mkdir(cfg["paths"]["uploads_dir"])

print("[INIT] Cargando modelo HerdNet... esto puede tardar unos segundos.")
infer_engine = HerdNetInference(CONFIG_PATH)
print("[READY] Modelo cargado y listo para inferencia.")

# Información del modelo (actualizada con tablas)
MODEL_INFO = """
### Arquitectura y datos
- **Modelo:** HerdNet (FPN + Density Maps)
- **Dataset:** ULiège-AIR (6 especies + background)
- **Última actualización:** 07 Nov 2025

### Desempeño general (Fine-Tuning oficial)

| Métrica     | Valor   |
|--------------|---------|
| F1-score     | 0.8405  |
| Precision    | 0.8407  |
| Recall       | 0.8404  |
| MAE          | 1.8023  |
| RMSE         | 3.4892  |

### Matriz de confusión (normalizada)

| Real \\ Predicha | buffalo | elephant | kob | topi | warthog | waterbuck |
|------------------|----------|-----------|------|------|----------|-------------|
| **buffalo**      | 0.94 | 0.00 | 0.05 | 0.01 | 0.00 | 0.00 |
| **elephant**     | 0.01 | 0.91 | 0.00 | 0.07 | 0.01 | 0.00 |
| **kob**          | 0.08 | 0.00 | 0.92 | 0.00 | 0.00 | 0.00 |
| **topi**         | 0.03 | 0.00 | 0.00 | 0.94 | 0.03 | 0.00 |
| **warthog**      | 0.06 | 0.06 | 0.06 | 0.00 | 0.81 | 0.00 |
| **waterbuck**    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
"""


def run_inference(image: Image.Image):
    """
    Ejecuta la inferencia sobre una imagen PIL y devuelve los resultados.
    """
    if image is None:
        return None, pd.DataFrame(), None, None

    annotated_img, counts = infer_engine.infer_single(image)

    # Construir tabla completa de especies
    all_species = list(infer_engine.classes.values())
    df_counts = pd.DataFrame({
        "Especie": all_species,
        "Conteo": [counts.get(sp, 0) for sp in all_species],
    })
    df_counts.loc[len(df_counts)] = ["Total", df_counts["Conteo"].sum()]

    # Guardar conteos
    csv_counts_path = os.path.join(infer_engine.output_dir, "species_counts.csv")
    df_counts.to_csv(csv_counts_path, index=False)

    # Comprobar existencia de detecciones
    detections_csv = os.path.join(infer_engine.output_dir, "detections.csv")
    if not os.path.exists(detections_csv):
        detections_csv = None

    return annotated_img, df_counts, csv_counts_path, detections_csv


# ===============================================================
# Interfaz de Gradio (versión estética mejorada)
# ===============================================================
custom_css = """
#main-title h1 {
    font-size: 2.2em !important;
    color: #e0f2fe !important;
    font-weight: 700 !important;
    margin-bottom: 0.3em;
}

h2 {
    color: #93c5fd !important;
    font-weight: 600 !important;
    margin-top: 1em;
    margin-bottom: 0.3em;
}

.block-section {
    background-color: #0f172a;
    border-radius: 10px;
    padding: 15px 20px;
    margin-bottom: 25px;
    box-shadow: 0 0 10px #00000040;
}

.img-bordered img {
    border-radius: 10px;
    box-shadow: 0 0 10px #00000040;
}

.data-table table {
    font-size: 15px !important;
}

.download-btn {
    background-color: #1e3a8a !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    border: 1px solid #3b82f6 !important;
    transition: all 0.2s ease-in-out;
}

.download-btn:hover {
    background-color: #2563eb !important;
    transform: scale(1.05);
}

.big-button button {
    background-color: #1d4ed8 !important;
    color: #f8fafc !important;
    font-weight: 700 !important;
    font-size: 20px !important;
    padding: 16px 28px !important;
    border-radius: 10px !important;
    width: 100% !important;
    transition: all 0.25s ease-in-out;
}

.big-button button:hover {
    background-color: #2563eb !important;
    transform: scale(1.03);
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=custom_css,
) as demo:
    # Encabezado principal
    gr.Markdown("# Detección y Conteo de Mamíferos Africanos", elem_id="main-title")

    # Información del modelo (colapsable con tablas)
    with gr.Accordion("Información del modelo", open=False):
        gr.Markdown(MODEL_INFO)

    # Bloque: resultados visuales
    gr.Markdown("## Resultados de inferencia")
    with gr.Row(elem_classes=["block-section"]):
        image_input = gr.Image(
            type="pil",
            label="Subir imagen aérea",
            height=380,
            elem_classes=["img-bordered"],
        )
        image_output = gr.Image(
            label="Detecciones (puntos resaltados)",
            height=380,
            elem_classes=["img-bordered"],
        )

    # Botón principal más grande y visible
    btn = gr.Button(
        "Ejecutar detección y conteo",
        variant="primary",
        elem_classes=["big-button"],
    )

    # Bloque: conteo detallado
    gr.Markdown("## Conteo detallado por especie")
    with gr.Column(elem_classes=["block-section"]):
        counts_output = gr.Dataframe(
            headers=["Especie", "Conteo"],
            label="Resultados de detección",
            interactive=False,
            elem_classes=["data-table"],
        )

        with gr.Row():
            download_counts = gr.File(
                label="Descargar conteos (CSV)",
                elem_classes=["download-btn"],
            )
            download_detections = gr.File(
                label="Descargar anotaciones (detections.csv)",
                elem_classes=["download-btn"],
            )

    btn.click(
        fn=run_inference,
        inputs=image_input,
        outputs=[image_output, counts_output, download_counts, download_detections],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Escuchar en todas las interfaces
        server_port=7860,
        share=False
    )

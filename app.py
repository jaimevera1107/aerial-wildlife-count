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

# Emojis para especies
SPECIES_EMOJI = {
    "buffalo": "🦬",
    "elephant": "🐘", 
    "kob": "🦌",
    "topi": "🫎",
    "warthog": "🐗",
    "waterbuck": "🦌",
}

# Información del modelo
MODEL_INFO = """
| Característica | Valor |
|----------------|-------|
| **Modelo** | HerdNet (FPN + Density Maps) |
| **Dataset** | ULiège-AIR (6 especies) |
| **F1-score** | 0.8405 |
| **Precision** | 0.8407 |
| **Recall** | 0.8404 |

**Especies detectables:** Buffalo, Elephant, Kob, Topi, Warthog, Waterbuck
"""


def run_inference(image: Image.Image):
    """
    Ejecuta la inferencia sobre una imagen PIL y devuelve los resultados.
    """
    if image is None:
        return None, "", None, None

    annotated_img, counts = infer_engine.infer_single(image)

    # Construir HTML con barras de progreso visuales
    all_species = list(infer_engine.classes.values())
    total = sum(counts.values())
    max_count = max(counts.values()) if counts.values() else 1
    
    # Generar HTML para el conteo visual
    html_content = '<div class="species-grid">'
    for species in all_species:
        count = counts.get(species, 0)
        emoji = SPECIES_EMOJI.get(species, "🐾")
        percentage = (count / max_count * 100) if max_count > 0 else 0
        
        html_content += f'''
        <div class="species-card">
            <div class="species-header">
                <span class="species-emoji">{emoji}</span>
                <span class="species-name">{species.capitalize()}</span>
                <span class="species-count">{count}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {percentage}%"></div>
            </div>
        </div>
        '''
    
    html_content += f'''
        <div class="total-card">
            <span class="total-label">Total detectado</span>
            <span class="total-count">{total}</span>
        </div>
    </div>
    '''

    # Guardar CSV
    df_counts = pd.DataFrame({
        "Especie": all_species + ["Total"],
        "Conteo": [counts.get(sp, 0) for sp in all_species] + [total],
    })
    csv_counts_path = os.path.join(infer_engine.output_dir, "species_counts.csv")
    df_counts.to_csv(csv_counts_path, index=False)

    # Comprobar existencia de detecciones
    detections_csv = os.path.join(infer_engine.output_dir, "detections.csv")
    if not os.path.exists(detections_csv):
        detections_csv = None

    return annotated_img, html_content, csv_counts_path, detections_csv


# ===============================================================
# CSS Moderno estilo Wildlife Vision
# ===============================================================
custom_css = """
/* Variables de color */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #141414;
    --bg-card: #1a1a1a;
    --border-color: #2a2a2a;
    --accent-gold: #d4a439;
    --accent-gold-light: #e5b84a;
    --text-primary: #f5f5f5;
    --text-secondary: #a0a0a0;
    --success-green: #22c55e;
}

/* Fondo general */
.gradio-container {
    background: var(--bg-primary) !important;
    max-width: 1400px !important;
}

/* Header principal */
#header-container {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-title {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-title .logo {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.header-subtitle {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}

.model-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    background: var(--success-green);
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Cards principales */
.main-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    color: var(--text-primary);
    font-weight: 600;
    font-size: 1.1rem;
}

.card-header .icon {
    color: var(--accent-gold);
    font-size: 1.3rem;
}

/* Área de imagen */
.image-upload-area {
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--bg-secondary) !important;
    transition: all 0.3s ease;
}

.image-upload-area:hover {
    border-color: var(--accent-gold) !important;
    background: rgba(212, 164, 57, 0.05) !important;
}

/* Botón principal */
.run-button {
    background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%) !important;
    color: #0a0a0a !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    text-transform: none !important;
}

.run-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(212, 164, 57, 0.3) !important;
}

.run-button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* Grid de especies */
.species-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.species-card {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 12px 16px;
}

.species-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}

.species-emoji {
    font-size: 1.5rem;
}

.species-name {
    flex: 1;
    color: var(--text-primary);
    font-weight: 500;
}

.species-count {
    color: var(--accent-gold);
    font-weight: 700;
    font-size: 1.1rem;
}

.progress-bar {
    height: 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
    border-radius: 4px;
    transition: width 0.5s ease;
}

.total-card {
    background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
}

.total-label {
    color: #0a0a0a;
    font-weight: 600;
}

.total-count {
    color: #0a0a0a;
    font-weight: 800;
    font-size: 1.8rem;
}

/* Sección de información */
.info-accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
}

/* Botones de descarga */
.download-section {
    display: flex;
    gap: 12px;
    margin-top: 1rem;
}

.download-btn button {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
}

.download-btn button:hover {
    border-color: var(--accent-gold) !important;
    background: rgba(212, 164, 57, 0.1) !important;
}

/* Ocultar elementos por defecto de Gradio */
.gradio-container footer {
    display: none !important;
}

/* Responsive */
@media (max-width: 768px) {
    .header-title {
        font-size: 1.4rem !important;
    }
    
    .main-card {
        padding: 1rem !important;
    }
}

/* Títulos de sección */
.section-title {
    color: var(--text-primary) !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin: 1.5rem 0 1rem 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title .icon {
    color: var(--accent-gold);
}

/* Labels de Gradio */
label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* Tablas */
.dataframe {
    background: var(--bg-secondary) !important;
    border-radius: 8px !important;
}

table {
    color: var(--text-primary) !important;
}

th {
    background: var(--bg-primary) !important;
    color: var(--accent-gold) !important;
}
"""

# ===============================================================
# Interfaz de Gradio - Wildlife Vision
# ===============================================================
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        body_background_fill="#0a0a0a",
        body_background_fill_dark="#0a0a0a",
        block_background_fill="#1a1a1a",
        block_background_fill_dark="#1a1a1a",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        input_background_fill="#141414",
        input_background_fill_dark="#141414",
    ),
    css=custom_css,
    title="Wildlife Vision - Detección de Fauna Africana"
) as demo:
    
    # Header personalizado
    gr.HTML("""
        <div id="header-container">
            <div>
                <div class="header-title">
                    <div class="logo">👁️</div>
                    Wildlife Vision
                </div>
                <p class="header-subtitle">wildlife.vision</p>
            </div>
            <div class="model-status">
                <span class="status-dot"></span>
                <span>Modelo activo</span>
            </div>
        </div>
    """)

    # Información del modelo (colapsable)
    with gr.Accordion("ℹ️ Información del Modelo", open=False, elem_classes=["info-accordion"]):
        gr.Markdown(MODEL_INFO)

    # Layout principal de dos columnas
    with gr.Row():
        # Columna izquierda - Entrada
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title"><span class="icon">📷</span> Imagen Aérea</div>')
            
            with gr.Group(elem_classes=["main-card"]):
                image_input = gr.Image(
                    type="pil",
                    label="Arrastra o selecciona una imagen",
                    height=350,
                    elem_classes=["image-upload-area"],
                )
                
                btn = gr.Button(
                    "▶ Ejecutar Detección",
                    variant="primary",
                    elem_classes=["run-button"],
                )

        # Columna derecha - Resultados
        with gr.Column(scale=1):
            gr.HTML('<div class="section-title"><span class="icon">🎯</span> Detecciones</div>')
            
            with gr.Group(elem_classes=["main-card"]):
                image_output = gr.Image(
                    label="Imagen con detecciones",
                    height=350,
                    elem_classes=["image-upload-area"],
                )

    # Sección de conteo
    gr.HTML('<div class="section-title"><span class="icon">📊</span> Conteo por Especie</div>')
    
    with gr.Group(elem_classes=["main-card"]):
        counts_html = gr.HTML(
            value='<div style="color: #a0a0a0; text-align: center; padding: 2rem;">Sube una imagen para ver los resultados</div>'
        )
        
        with gr.Row(elem_classes=["download-section"]):
            download_counts = gr.File(
                label="📥 Conteos (CSV)",
                elem_classes=["download-btn"],
            )
            download_detections = gr.File(
                label="📥 Detecciones (CSV)",
                elem_classes=["download-btn"],
            )

    # Evento del botón
    btn.click(
        fn=run_inference,
        inputs=image_input,
        outputs=[image_output, counts_html, download_counts, download_detections],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

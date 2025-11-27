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
### Arquitectura y datos
- **Modelo:** HerdNet (FPN + Density Maps)
- **Dataset:** ULiège-AIR (6 especies + background)
- **Última actualización:** 07 Nov 2025

### Desempeño general (Fine-Tuning oficial)

| Métrica | Valor |
|---------|-------|
| F1-score | 0.8405 |
| Precision | 0.8407 |
| Recall | 0.8404 |
| MAE | 1.8023 |
| RMSE | 3.4892 |

### Matriz de confusión (normalizada)

| Real \\ Predicha | buffalo | elephant | kob | topi | warthog | waterbuck |
|------------------|---------|----------|-----|------|---------|-----------|
| **buffalo** | 0.94 | 0.00 | 0.05 | 0.01 | 0.00 | 0.00 |
| **elephant** | 0.01 | 0.91 | 0.00 | 0.07 | 0.01 | 0.00 |
| **kob** | 0.08 | 0.00 | 0.92 | 0.00 | 0.00 | 0.00 |
| **topi** | 0.03 | 0.00 | 0.00 | 0.94 | 0.03 | 0.00 |
| **warthog** | 0.06 | 0.06 | 0.06 | 0.00 | 0.81 | 0.00 |
| **waterbuck** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
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
/* Variables de color - Mejorado contraste */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #141414;
    --bg-card: #1c1c1c;
    --border-color: #333333;
    --accent-gold: #d4a439;
    --accent-gold-light: #e5b84a;
    --text-primary: #ffffff;
    --text-secondary: #c0c0c0;
    --text-muted: #888888;
    --success-green: #22c55e;
}

/* Eliminar scroll horizontal */
html, body {
    overflow-x: hidden !important;
}

/* Fondo general */
.gradio-container {
    background: var(--bg-primary) !important;
    max-width: 1400px !important;
    overflow-x: hidden !important;
}

/* Header principal - sin overflow */
#header-container {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 1.5rem;
    margin: 0 0 1.5rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 12px;
}

.header-title {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-title .logo {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

.header-subtitle {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    margin: 0.2rem 0 0 0 !important;
}

.model-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 0.9rem;
    font-weight: 500;
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

/* Títulos de sección - MÁS GRANDES */
.section-title {
    color: var(--text-primary) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin: 1.5rem 0 1rem 0 !important;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.02em;
}

.section-title .icon {
    color: var(--accent-gold);
    font-size: 1.4rem;
}

/* Footer/Copyright */
.footer-copyright {
    text-align: center;
    padding: 2rem 1rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.9rem;
}

.footer-copyright .team-title {
    color: var(--text-secondary);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.footer-copyright .team-members {
    color: var(--accent-gold);
    font-weight: 500;
    line-height: 1.6;
}

.footer-copyright .university {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-weight: 500;
}

.footer-copyright .year {
    margin-top: 0.8rem;
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Labels de Gradio - Mejor contraste */
label, .label-wrap span {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* Textos generales */
p, span, div {
    color: var(--text-primary);
}

/* Input labels específicos */
.wrap label {
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

/* Acordeón - mejor contraste */
.accordion {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
}

.accordion button {
    color: var(--text-primary) !important;
}

.accordion button span {
    color: var(--text-primary) !important;
}

/* Markdown content */
.prose, .markdown-text {
    color: var(--text-primary) !important;
}

.prose p, .markdown-text p {
    color: var(--text-secondary) !important;
}

.prose strong, .markdown-text strong {
    color: var(--text-primary) !important;
}

/* Fix placeholder text */
input::placeholder, textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Contenedor principal sin scroll */
.contain {
    overflow-x: hidden !important;
}

/* Mensaje vacío con mejor contraste */
.empty-message {
    color: var(--text-secondary) !important;
}

/* File upload labels */
.file-preview span {
    color: var(--text-secondary) !important;
}

/* Botón dentro de upload */
.upload-button {
    color: var(--text-secondary) !important;
}

/* Iconos y textos de Gradio */
.svelte-1p9xokt, .wrap span {
    color: var(--text-secondary) !important;
}

/* === CRÍTICO: Arreglar labels con fondo blanco === */
/* Labels de imagen */
.image-container label,
.image-frame label,
[data-testid="image"] label,
.gr-image label,
.upload-container label {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

/* Labels con fondo blanco - selector universal */
span.svelte-s1r2yt,
label.svelte-s1r2yt,
div.svelte-s1r2yt,
.label-wrap,
.label-wrap span,
.block label span {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

/* Cualquier label o span dentro de blocks */
.block label,
.block label span,
.block .label-wrap,
.block .label-wrap span,
.wrap > label,
.wrap > label span {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

/* Labels específicos de componentes de imagen */
.image label span,
.gr-box label span,
.gr-form label span,
.gr-panel label span {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

/* Forzar en todos los labels dentro del container */
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-secondary) !important;
}

/* Selector muy específico para etiquetas flotantes */
[class*="label"],
[class*="Label"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
}

/* File labels también */
.file-preview label,
.file label,
[data-testid="file"] label {
    background: transparent !important;
    color: var(--text-secondary) !important;
}

/* Inputs y textareas fondo oscuro */
input, textarea, select {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

/* Dropdown/select oscuro */
.dropdown, .choices, .choices__inner {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

/* ===== FIX DEFINITIVO: Labels con pill/badge ===== */
/* Estas son las clases exactas de Gradio para los labels */
.label-wrap {
    background: #1a1a1a !important;
    background-color: #1a1a1a !important;
}

.label-wrap > span {
    background: #1a1a1a !important;
    background-color: #1a1a1a !important;
    color: #c0c0c0 !important;
    border: none !important;
}

/* Target directo a los span dentro de labels de imagen */
[data-testid="block"] .label-wrap span,
.block .label-wrap span,
.wrap .label-wrap span {
    background: #1a1a1a !important;
    background-color: #1a1a1a !important;
    color: #c0c0c0 !important;
}

/* Selector universal para cualquier span que sea label */
span[data-testid="block-label"],
.block-label,
.svelte-1gfkn6j {
    background: #1a1a1a !important;
    background-color: #1a1a1a !important;
    color: #c0c0c0 !important;
}

/* Específico para componentes Image */
.image-container .label-wrap span,
.gr-image .label-wrap span {
    background: #1a1a1a !important;
    color: #c0c0c0 !important;
}

/* Máxima prioridad - selector de todos los spans en wraps */
div.wrap span.svelte-1gfkn6j,
div span.svelte-1gfkn6j,
span.svelte-1gfkn6j {
    background: #1a1a1a !important;
    background-color: #1a1a1a !important;
    color: #c0c0c0 !important;
    border: 1px solid #333333 !important;
}

/* File component labels */
.file .label-wrap span,
[data-testid="file"] .label-wrap span {
    background: #1a1a1a !important;
    color: #c0c0c0 !important;
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
        # Fondos principales
        body_background_fill="#0a0a0a",
        body_background_fill_dark="#0a0a0a",
        block_background_fill="#1a1a1a",
        block_background_fill_dark="#1a1a1a",
        block_border_color="#333333",
        block_border_color_dark="#333333",
        # Inputs
        input_background_fill="#141414",
        input_background_fill_dark="#141414",
        input_border_color="#333333",
        input_border_color_dark="#333333",
        # Labels - FONDO TRANSPARENTE
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_color="transparent",
        block_label_border_color_dark="transparent",
        block_label_text_color="#c0c0c0",
        block_label_text_color_dark="#c0c0c0",
        # Textos
        body_text_color="#ffffff",
        body_text_color_dark="#ffffff",
        body_text_color_subdued="#888888",
        body_text_color_subdued_dark="#888888",
        # Bordes
        border_color_primary="#333333",
        border_color_primary_dark="#333333",
        # Secundarios
        background_fill_secondary="#141414",
        background_fill_secondary_dark="#141414",
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

    # Footer con copyright
    gr.HTML("""
        <div class="footer-copyright">
            <div class="team-title">🦁 Desarrollado por</div>
            <div class="team-members">
                Julian F. Cujabante Villamil &nbsp;•&nbsp; 
                Rafael A. Ortega Pabón &nbsp;•&nbsp; 
                Uldy D. Paloma Rozo &nbsp;•&nbsp; 
                Jaime A. Vera Jaramillo
            </div>
            <div class="university">Universidad de los Andes</div>
            <div class="year">© 2025 - Proyecto de Detección de Fauna Africana</div>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

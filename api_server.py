"""
API Server para Wildlife Detection
Backend FastAPI que expone endpoints para el frontend Next.js
"""
import warnings
import base64
import io
from typing import List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

# ===============================================================
# FastAPI App
# ===============================================================
app = FastAPI(
    title="Wildlife Detection API",
    description="API para detección y conteo de fauna africana en imágenes aéreas",
    version="1.0.0"
)

# CORS - Permitir conexiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================
# Modelos de respuesta
# ===============================================================
class Detection(BaseModel):
    species: str
    count: int

class DetectionResult(BaseModel):
    detections: List[Detection]
    totalCount: int
    annotatedImage: str  # Base64 encoded image

class HealthResponse(BaseModel):
    status: str
    model: str
    classes: List[str]

# ===============================================================
# Endpoints
# ===============================================================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Verificar estado del modelo"""
    try:
        return HealthResponse(
            status="online",
            model=cfg["model"]["name"],
            classes=list(infer_engine.classes.values())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect", response_model=DetectionResult)
async def detect_animals(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0)
):
    """
    Detectar animales en una imagen aérea
    
    - **file**: Imagen a analizar (PNG, JPG, TIFF)
    - **confidence**: Umbral de confianza (0.0 - 1.0)
    """
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Ejecutar inferencia
        annotated_img, counts = infer_engine.infer_single(image)
        
        # Convertir imagen anotada a base64
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # Preparar detecciones
        detections = [
            Detection(species=species.capitalize(), count=count)
            for species, count in counts.items()
            if count > 0
        ]
        
        total_count = sum(counts.values())
        
        return DetectionResult(
            detections=detections,
            totalCount=total_count,
            annotatedImage=img_data_url
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {str(e)}")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Wildlife Detection API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_yolov8.py
Entrena YOLOv8 usando:
 - TRAIN: augmentation_config.yaml -> mirror_clean/train_final (aumentado)
 - VAL/TEST: quality_config.yaml -> COCO big_size (limpios)

Requisitos: pip install ultralytics pyyaml tqdm
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
import yaml
from tqdm import tqdm

# Ultralytics
from ultralytics import YOLO

# ---------------------------
# Utilidades de paths/config
# ---------------------------
def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_train_final(aug_cfg):
    """Encuentra mirror_clean/train_final + train_final.json a partir de augmentation_config.yaml."""
    out = Path(aug_cfg.get("output_dir", "data/outputs")).resolve()
    mirror = out / "mirror_clean"
    # por defecto usamos train_final
    final_dir = mirror / "train_final"
    final_json = final_dir / "train_final.json"
    if not final_dir.exists() or not final_json.exists():
        raise FileNotFoundError(f"No encontré {final_dir} o {final_json}. Corre antes el pipeline de augment.")
    return final_dir, final_json

def find_val_test(qual_cfg):
    """Lee quality_config.yaml y retorna paths a val/test (COCO JSON) y las carpetas de imágenes."""
    splits = qual_cfg.get("splits", {})
    val_json = Path(splits["val"]).resolve()
    test_json = Path(splits["test"]).resolve()
    if not val_json.exists() or not test_json.exists():
        raise FileNotFoundError("No encontré val/test JSON en quality_config.yaml (revisa 'splits').")
    # Asumimos que las imágenes viven junto al COCO original (puedes ajustar si es distinto)
    # YOLOv8 necesita listas de imágenes o formato YOLO; aquí convertiremos COCO a YOLO temporalmente.
    return val_json, test_json

# -------------------------------------------------------------
# COCO -> YOLO (solo cajas) para usar con Ultralytics fácilmente
# -------------------------------------------------------------
def coco_to_yolo(coco_json, images_root, yolo_labels_dir):
    """
    Convierte un COCO a etiquetas YOLOv8 (txt por imagen).
    - images_root: carpeta donde están las imágenes referenciadas en COCO
    - yolo_labels_dir: a dónde escribir *.txt
    Retorna:
      - lista absoluta de rutas de imágenes (para hacer train.txt/val.txt/test.txt)
    """
    import json
    from PIL import Image

    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    coco = json.load(open(coco_json, "r", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    name2id = {v: k for k, v in cats.items()}
    # reindex categories to 0..C-1
    ordered = {name: i for i, name in enumerate(sorted(name2id.keys()))}
    cat_old_to_new = {old_id: ordered[name] for old_id, name in cats.items()}

    img_map = {im["id"]: im for im in coco["images"]}
    ann_map = {}
    for a in coco["annotations"]:
        ann_map.setdefault(a["image_id"], []).append(a)

    image_paths = []
    for img_id, im in tqdm(img_map.items(), desc=f"COCO→YOLO ({coco_json.name})"):
        fn = im["file_name"]
        src = Path(images_root) / Path(fn).name  # asunción: mismo nombre
        if not src.exists():
            # intenta ruta absoluta si venía completa
            maybe = Path(fn)
            src = maybe if maybe.exists() else src
        if not src.exists():
            # si no está, saltamos (o podrías lanzar error)
            continue

        # tamaño real desde PIL (por seguridad)
        try:
            W, H = Image.open(src).size
        except Exception:
            W, H = im.get("width", 0), im.get("height", 0)

        label_path = yolo_labels_dir / (src.stem + ".txt")
        lines = []
        for a in ann_map.get(img_id, []):
            x, y, w, h = a["bbox"]
            if W <= 0 or H <= 0 or w <= 0 or h <= 0:
                continue
            # convertir a cx,cy,w,h normalizado
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            cls = cat_old_to_new[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        image_paths.append(str(src.resolve()))
    return ordered, image_paths

def write_yolo_set(image_list, txt_path):
    txt_path.write_text("\n".join(image_list), encoding="utf-8")

# ---------------------------
# Entrenamiento YOLOv8
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_yaml", default="augmentation_config.yaml")
    ap.add_argument("--qual_yaml", default="quality_config.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs_yolo")
    ap.add_argument("--name", default="yolo_big_final")
    ap.add_argument("--fp16", action="store_true", help="Mixed precision")
    ap.add_argument("--close_mosaic", type=int, default=10, help="Cerrar mosaic en últimas N épocas")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--tmp_yolo", action="store_true",
                    help="Convierte COCO->YOLO en tmp para val/test (recomendado si no tienes formato YOLO).")
    args = ap.parse_args()

    aug_cfg = load_yaml(args.aug_yaml)
    qual_cfg = load_yaml(args.qual_yaml)

    # TRAIN (aumentado)
    train_dir, train_json = find_train_final(aug_cfg)

    # VAL/TEST (limpios, COCO)
    val_json, test_json = find_val_test(qual_cfg)

    workdir = Path(args.project) / args.name
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "run.yaml").write_text(
        yaml.safe_dump(vars(args), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # -------------------------------------------------
    # Si no tienes etiquetas YOLO, convertimos COCO→YOLO
    # -------------------------------------------------
    tmp = None
    if args.tmp_yolo:
        tmp = Path(tempfile.mkdtemp(prefix="yolo_data_"))
        labels = tmp / "labels"
        labels.mkdir(exist_ok=True)
        # convertimos VAL y TEST desde COCO (para que YOLO pueda validarlos)
        names_val, images_val = coco_to_yolo(val_json, val_json.parent.parent.parent / "images", labels)
        names_test, images_test = coco_to_yolo(test_json, test_json.parent.parent.parent / "images", labels)
        # Para TRAIN, tu pipeline de augment ya deja etiquetas en COCO; convertimos también:
        names_train, images_train = coco_to_yolo(train_json, train_dir, labels)

        # YOLO dataset: listas de rutas
        write_yolo_set(images_train, tmp / "train.txt")
        write_yolo_set(images_val,   tmp / "val.txt")
        write_yolo_set(images_test,  tmp / "test.txt")

        # YAML de nombres
        names = sorted(set(list(names_train.keys()) + list(names_val.keys()) + list(names_test.keys())))
        names = {i: n for i, n in enumerate(names)}
        data_yaml = {
            "path": str(tmp.resolve()),
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "names": names
        }
        (tmp / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        data_path = str(tmp / "data.yaml")
    else:
        # Si ya tienes dataset en YOLO nativamente, simplemente prepara tu data.yaml y colócalo aquí.
        raise SystemExit("Activa --tmp_yolo o provee un data.yaml con rutas YOLO. (Ver comentarios en el script).")

    # -----------------------
    # Entrena con Ultralytics
    # -----------------------
    model = YOLO(args.model)
    results = model.train(
        data=data_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        amp=args.fp16,
        verbose=True
    )

    # Evalúa en test
    model.val(split="test")

    # Exporta mejor modelo a ONNX/torchscript si quieres:
    # model.export(format="onnx", opset=12)

    print("\nEntrenamiento YOLOv8 finalizado.")
    if tmp and tmp.exists():
        print(f"Limpiando tmp: {tmp}")
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()

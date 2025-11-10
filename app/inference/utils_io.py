import os
import yaml
import logging
from datetime import datetime

# ===============================================================
#  Utility Functions for I/O, Config, and Logging
# ===============================================================

def mkdir(path: str) -> None:
    """
    Creates a directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def load_yaml_config(path: str) -> dict:
    """
    Loads a YAML configuration file and returns it as a dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[CONFIG] File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def get_timestamp() -> str:
    """
    Returns a timestamp string for naming logs and output folders.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_timestamp_dir(base_dir: str) -> str:
    """
    Creates a timestamped subdirectory inside the given base directory.
    Example: resources/outputs/infer_20251106_233000/
    """
    timestamp = get_timestamp()
    new_dir = os.path.join(base_dir, f"infer_{timestamp}")
    mkdir(new_dir)
    return new_dir


def init_logger(log_dir: str, name: str = "herdnet_infer") -> logging.Logger:
    """
    Initializes a logger that writes both to console and to a file.
    """
    mkdir(log_dir)
    timestamp = get_timestamp()
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if reloaded
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

        # File handler
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.info(f"[LOGGER] Initialized → {log_path}")
    return logger


def save_csv(df, path: str, logger: logging.Logger = None) -> None:
    """
    Saves a DataFrame as CSV, logging the event.
    """
    df.to_csv(path, index=False)
    if logger:
        logger.info(f"[OUTPUT] Saved CSV → {path}")
    else:
        print(f"[OUTPUT] Saved CSV → {path}")


def clean_uploads(upload_dir: str, logger: logging.Logger = None) -> None:
    """
    Cleans temporary uploaded files in the uploads directory.
    """
    if not os.path.exists(upload_dir):
        return

    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    for f in files:
        try:
            os.remove(os.path.join(upload_dir, f))
        except Exception as e:
            if logger:
                logger.warning(f"[CLEANUP] Could not remove {f}: {e}")

    if logger:
        logger.info(f"[CLEANUP] Cleared {len(files)} files from {upload_dir}")


def get_temp_image_path(upload_dir: str, filename: str = None) -> str:
    """
    Generates a unique temporary image path inside the uploads directory.
    """
    mkdir(upload_dir)
    if filename is None:
        filename = f"tmp_{get_timestamp()}.jpg"
    return os.path.join(upload_dir, filename)

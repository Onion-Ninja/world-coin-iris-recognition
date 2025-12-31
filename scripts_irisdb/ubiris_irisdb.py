import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
UBIRIS_ROOT = Path(
    "/home/nishkal/datasets/iris_datasets/UBIRIS/UBIRIS.v2 - Done/ubiris.v2/UBIRIS_v2"
)

IRIS_DB_ROOT = Path(
    "/home/nishkal/datasets/iris_db/UBIRIS_v2"
)

ORIG_DIR = IRIS_DB_ROOT / "orig"

VALID_EXTENSIONS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")
# ---------------------------------------


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_ubiris_filename(filename: str):
    """
    Parses: 14_L.tiff
    Returns: (image_number, eye_side, extension)
    """
    match = re.match(r"(\d+)_([LR])(\.[^.]+)$", filename, re.IGNORECASE)
    if not match:
        return None

    image_number, eye_side, ext = match.groups()
    return int(image_number), eye_side.upper(), ext.lower()


def convert_ubiris():
    print("=== Converting UBIRIS to iris_db format ===")

    ensure_dir(ORIG_DIR)

    # ---- collect numeric directories only ----
    user_dirs = [
        d for d in UBIRIS_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit()
    ]

    # ---- compute max user id (excluding '000') ----
    numeric_ids = [int(d.name) for d in user_dirs if d.name != "000"]
    max_user_id = max(numeric_ids) if numeric_ids else 0

    print(f"Max UBIRIS user id found: {max_user_id}")

    # ---- sort directories numerically, keep 000 last ----
    user_dirs.sort(
        key=lambda d: (d.name == "000", int(d.name))
    )

    # ---- progress bar over users ----
    for user_dir in tqdm(user_dirs, desc="Processing UBIRIS users", unit="user"):
        folder_name = user_dir.name

        # ---- assign user_id ----
        if folder_name == "000":
            user_id = max_user_id + 1
            tqdm.write(f"[INFO] Mapping folder '000' → user_id {user_id}")
        else:
            user_id = int(folder_name)

        image_files = sorted(user_dir.iterdir())

        for img_path in image_files:
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            parsed = parse_ubiris_filename(img_path.name)
            if parsed is None:
                tqdm.write(f"[SKIP] Invalid filename: {img_path}")
                continue

            image_number, eye_side, ext = parsed

            # Target directory: {user_id}_{eye}
            target_dir = ORIG_DIR / f"{user_id}_{eye_side}"
            ensure_dir(target_dir)

            # Target filename
            target_filename = f"{user_id}_{eye_side}_{image_number}{ext}"
            target_path = target_dir / target_filename

            if target_path.exists():
                continue  # restart-safe

            shutil.copy2(img_path, target_path)

    print("=== UBIRIS conversion complete ===")


if __name__ == "__main__":
    convert_ubiris()

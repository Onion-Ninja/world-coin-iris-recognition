import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
MMU_ROOT = Path(
    "/home/nishkal/datasets/iris_datasets/MMUIrisDatabase/MMU Iris Database"
)

IRIS_DB_ROOT = Path(
    "/home/nishkal/datasets/iris_db/MMUIrisDatabase"
)

ORIG_DIR = IRIS_DB_ROOT / "orig"

VALID_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
# ---------------------------------------


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_mmu_filename(filename: str):
    """
    Parses: aeval1.bmp
    Returns: image_number
    """
    match = re.search(r"(\d+)", filename)
    if not match:
        return None
    return int(match.group(1))


def convert_mmu():
    print("=== Converting MMU Iris Database to iris_db format ===")

    ensure_dir(ORIG_DIR)

    # numeric user directories only
    user_dirs = sorted(
        [d for d in MMU_ROOT.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )

    print(f"Found {len(user_dirs)} MMU users")

    for user_dir in tqdm(user_dirs, desc="Processing MMU users", unit="user"):
        user_id = int(user_dir.name)

        for eye_dir in ["left", "right"]:
            eye_path = user_dir / eye_dir
            if not eye_path.exists():
                continue

            eye_side = "L" if eye_dir == "left" else "R"

            image_files = sorted(eye_path.iterdir())

            for img_path in image_files:
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                image_number = parse_mmu_filename(img_path.name)
                if image_number is None:
                    tqdm.write(f"[SKIP] Cannot parse image number: {img_path}")
                    continue

                # Target directory: {user_id}_{eye}
                target_dir = ORIG_DIR / f"{user_id}_{eye_side}"
                ensure_dir(target_dir)

                # Target filename: {user_id}_{eye}_{image_no}.bmp
                target_filename = f"{user_id}_{eye_side}_{image_number}{img_path.suffix.lower()}"
                target_path = target_dir / target_filename

                if target_path.exists():
                    continue  # restart-safe

                shutil.copy2(img_path, target_path)

    print("=== MMU conversion complete ===")


if __name__ == "__main__":
    convert_mmu()

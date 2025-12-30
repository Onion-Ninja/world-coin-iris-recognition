import iris
import cv2
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib

matplotlib.use("Agg")
print(iris.__version__)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_already_processed(seg_npz_root, rel_dir, unique_name):
    """
    Returns True if segmentation NPZ already exists.
    """
    seg_npz_path = os.path.join(
        seg_npz_root,
        rel_dir,
        f"{unique_name}_seg.npz"
    )
    return os.path.exists(seg_npz_path)

def scan_single_dataset(dataset_root):
    orig_path = os.path.join(dataset_root, "orig")
    if not os.path.isdir(orig_path):
        raise ValueError(f"'orig' folder not found in {dataset_root}")

    all_files = []
    user_ids = set()

    for root, dirs, files in os.walk(orig_path):
        dirs.sort()    # 🔑 ensures user folders are ordered
        files.sort()   # 🔑 ensures images are ordered

        folder = os.path.basename(root)

        match_folder = re.match(r"(\d+)_([LR])", folder, re.IGNORECASE)
        if not match_folder:
            continue

        user_id, eye_side = match_folder.groups()

        for f in files:
            if not f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
                continue

            name, _ = os.path.splitext(f)
            match_file = re.match(r"\d+_[LR]_(\d+)", name, re.IGNORECASE)
            if not match_file:
                continue

            image_number = int(match_file.group(1))  # int for numeric sorting

            meta = {
                "user_id": user_id,
                "eye_side": eye_side.upper(),
                "image_number": image_number,
                "session_number": "1",
                "filepath": os.path.join(root, f)
            }

            all_files.append(meta)
            user_ids.add(user_id)

    # 🔑 OPTIONAL: final global sort (extra safety)
    all_files.sort(
        key=lambda x: (int(x["user_id"]), x["eye_side"], x["image_number"])
    )

    return all_files, sorted(user_ids)


def pipeline(
    dataset_root,
    save_visuals=True,
    save_intermediates=True
):
    """
    Runs iris pipeline on ONE standardized dataset.
    Reuses a single initialized IRISPipeline instance.
    Preserves directory structure from orig/.
    """

    print(f"\n=== Processing Dataset: {os.path.basename(dataset_root)} ===")

    # -------------------- SCAN DATASET --------------------
    files, user_ids = scan_single_dataset(dataset_root)
    print(f"Found {len(files)} images | {len(user_ids)} users")

    if not files:
        print("No valid images found. Exiting.")
        return

    # -------------------- PATHS --------------------
    orig_root = os.path.join(dataset_root, "orig")

    vis_root = os.path.join(dataset_root, "worldcoin_outputs_images")
    npz_root = os.path.join(dataset_root, "worldcoin_outputs_npz")

    seg_vis_dir  = os.path.join(vis_root, "segmentation")
    norm_vis_dir = os.path.join(vis_root, "normalized")
    code_vis_dir = os.path.join(vis_root, "codes")

    temp_npz_dir = os.path.join(npz_root, "templates")
    seg_npz_dir  = os.path.join(npz_root, "segmentation")
    norm_npz_dir = os.path.join(npz_root, "normalized")

    if save_visuals:
        ensure_dir(seg_vis_dir)
        ensure_dir(norm_vis_dir)
        ensure_dir(code_vis_dir)

    if save_intermediates:
        ensure_dir(temp_npz_dir)
        ensure_dir(seg_npz_dir)
        ensure_dir(norm_npz_dir)

    # -------------------- INIT MACHINE ONCE --------------------
    print("Initializing IRIS Pipeline")
    iris_pipeline = iris.IRISPipeline(
        env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT
    )

    iris_visualizer = iris.visualisation.IRISVisualizer()

    # -------------------- MAIN LOOP --------------------
    for meta in tqdm(files, desc="Processing iris images"):
        img_path = meta["filepath"]

        rel_path = os.path.relpath(img_path, orig_root)
        rel_dir  = os.path.dirname(rel_path)

        unique_name = f"{meta['user_id']}_{meta['eye_side']}_{meta['image_number']}"
        # print("Processing:", unique_name, flush=True)

        if is_already_processed(seg_npz_dir, rel_dir, unique_name):
            print("skipped")
            continue

        try:
            # iris_pipeline.call_trace.clear()

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError("Image could not be read")

            eye = "left" if meta["eye_side"].lower() == "l" else "right"

            iris_image = iris.IRImage(
                img_data=image,
                image_id=unique_name,
                eye_side=eye
            )

            output = iris_pipeline(iris_image)
            # print(output.keys())
            # ---------------- SAVE INTERMEDIATES ----------------
            if save_intermediates:
                temp_npz_subdir = os.path.join(temp_npz_dir, rel_dir)
                seg_npz_subdir  = os.path.join(seg_npz_dir, rel_dir)
                norm_npz_subdir = os.path.join(norm_npz_dir, rel_dir)

                ensure_dir(temp_npz_subdir)
                ensure_dir(seg_npz_subdir)
                ensure_dir(norm_npz_subdir)

                # print("this")
                seg_arr = output['segmentation_map']
                if seg_arr is not None:
                    temp_path = os.path.join(seg_npz_subdir, f"{unique_name}_seg.npz")
                    np.savez_compressed(temp_path, predictions=seg_arr['predictions'], 
                                        index2class=np.array(list(seg_arr["index2class"].items()), dtype=object))
                    # print("saved seg")

                
                # print("saving norm")
                norm_arr = output['normalized_iris']
                if norm_arr is not None:  
                    np.savez_compressed(
                        os.path.join(norm_npz_subdir, f"{unique_name}_norm.npz"),
                        normalized_image=norm_arr["normalized_image"],
                        normalized_mask=norm_arr["normalized_mask"]
                    )
                    # print("saved norm")
                flg = False
                if(output['iris_template'] is None):
                    flg = True
                
                else:
                    iris_code = output["iris_template"].iris_codes[0]
                    mask_code = output["iris_template"].iris_codes[1]

                    np.savez_compressed(
                        os.path.join(temp_npz_subdir, f"{unique_name}.npz"),
                        iris_code=iris_code,
                        mask_code=mask_code
                    )
                # print("saved templates")

            # ---------------- SAVE VISUALS ----------------
            if save_visuals:
                seg_vis_subdir  = os.path.join(seg_vis_dir, rel_dir)
                norm_vis_subdir = os.path.join(norm_vis_dir, rel_dir)
                code_vis_subdir = os.path.join(code_vis_dir, rel_dir)

                ensure_dir(seg_vis_subdir)
                ensure_dir(norm_vis_subdir)
                ensure_dir(code_vis_subdir)

                canvas = iris_visualizer.plot_segmentation_map(
                    ir_image=iris.IRImage(img_data=image, eye_side=eye),
                    segmap=iris_pipeline.call_trace["segmentation"]
                )
                plt.savefig(
                    os.path.join(seg_vis_subdir, f"{unique_name}_seg.jpg"),
                    bbox_inches="tight"
                )
                plt.close("all")

                if(flg):
                    continue
                canvas = iris_visualizer.plot_normalized_iris(
                    normalized_iris=iris_pipeline.call_trace["normalization"]
                )
                plt.savefig(
                    os.path.join(norm_vis_subdir, f"{unique_name}_norm.jpg"),
                    bbox_inches="tight",
                    pad_inches=0
                )
                plt.close("all")

                canvas = iris_visualizer.plot_iris_template(
                    iris_template=iris_pipeline.call_trace["encoder"]
                )
                plt.savefig(
                    os.path.join(code_vis_subdir, f"{unique_name}_code.jpg"),
                    bbox_inches="tight"
                )
                plt.close("all")

        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")


home = "/home/nishkal/datasets/iris_db/"
cv1 = os.path.join(home, "CASIA_v1/")
cvt = os.path.join(home, "CASIA_iris_thousand/")
iitd = os.path.join(home, "IITD_v1/")

allfiles, user_ids = scan_single_dataset(cvt)
print(len(allfiles))
print(len(user_ids))

pipeline(cvt)
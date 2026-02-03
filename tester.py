import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from homography import Homography
from predict import OrientAny


def get_nerf_ccs_to_normal_ccs_T():
    return np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def deduce_R_blenderw_to_objw(row_data):
    # Unpack tuple data
    phi, theta_elev, target_img_idx, uid = row_data

    # GT rotation
    R_objw_to_normal_ccs_GT = OrientAny.get_R_objw2cam(phi, theta_elev, 0)

    # Load saved rotation
    path = f"/robodata/smodak/datasets/orientany/Objaverse_render_random/filter80k_archive_extra3/{uid.split('/')[0]}/filter80k/filter80k_render_extra3/{uid}/random_rt{target_img_idx}.npy"
    saved_npy = np.load(path)
    R_blenderw_to_nerf_ccs = saved_npy[:3, :3]

    # Transformations
    R_blenderw_to_normal_ccs = get_nerf_ccs_to_normal_ccs_T()[:3, :3] @ R_blenderw_to_nerf_ccs

    # Deduce R_blenderw_to_objw
    R_blenderw_to_objw = R_objw_to_normal_ccs_GT.T @ R_blenderw_to_normal_ccs

    # Save the deduced matrix
    save_path = f"/robodata/smodak/datasets/orientany/Objaverse_render_random/filter80k_archive_extra3/{uid.split('/')[0]}/filter80k/filter80k_render_extra3/{uid}/R_blenderw_to_objw{target_img_idx}.npy"
    np.save(save_path, R_blenderw_to_objw)

    return True


if __name__ == "__main__":
    df = pd.read_csv("/robodata/smodak/datasets/orientany/Objaverse_render_random/final_ans/obja_render_train_ex3_dataset.csv")

    # Filter rows with has_direction=True
    valid_rows = df[df["has_direction"]].reset_index(drop=True)
    skipped = len(df) - len(valid_rows)

    print(f"Processing {len(valid_rows)} rows with 128 parallel workers...")

    # Convert to simple tuples for pickling
    row_data = [(row.angle_ax, row.angle_pl, row.random_idx, row.uid)
                for row in valid_rows.itertuples(index=False)]

    # Process in parallel
    with Pool(128) as pool:
        results = list(tqdm(pool.imap(deduce_R_blenderw_to_objw, row_data),
                            total=len(row_data), desc="Deducing matrices"))

    print(f"Final: Processed: {len(valid_rows)}, Skipped: {skipped}")

"""
Orient-Anything: Object Orientation Estimation
Replicates Hugging Face Space demo exactly: https://huggingface.co/spaces/Viglong/Orient-Anything
"""
import torch
import torch.nn.functional as F
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoImageProcessor
import rembg
import matplotlib.pyplot as plt
import json
import os
import math
import glob
from tqdm import tqdm

from vision_tower import DINOv2_MLP
from homography import Homography


class OrientAny:
    def __init__(self, ckpt_dir='ckpts', model_name='croplargeEX2_dino_weight.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(os.path.join(ckpt_dir, "model_config.json"), "r") as f:
            self.model_config = json.load(f)[model_name.split('_')[0]]
        self._load_model(ckpt_path=f"{ckpt_dir}/{model_name}")

    def _load_model(self, ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        self.model = DINOv2_MLP(dino_mode=self.model_config['dino_mode'],
                                in_dim=self.model_config['in_dim'],
                                out_dim=self.model_config['out_dim'],
                                evaluate=True, mask_dino=False, frozen_back=False)
        self.model.eval()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.dinov2 = self.model.dinov2.to(self.device)
        preprocessors = {'small': "facebook/dinov2-small", 'base': "facebook/dinov2-base", 'large': "facebook/dinov2-large"}
        self.val_preprocess = AutoImageProcessor.from_pretrained(preprocessors[self.model_config['dino_mode']])

    def get_model_outputs(self, image):
        """
        azimuth(phi): 0-360, angle from x-axis to y-axis (about +z-axis) [normal spherical coordinates convention]
        polar(theta): 0-180, angle from +z-axis to rho vector joining origin to the point [normal spherical coordinates convention]
            * theta = pi - theta_model
            * theta_elev = pi/2 - theta
            => theta_elev = theta_model - pi/2
        roll(delta): the rotation of the camera about the +z of opencv-style normal CCS coordinate system facing the origin
            * ro_offset is ro_range / 2
        final ranges:
            * phi: (0, 360)
            * theta_elev: (-90, 90)
            * delta: (-ro_range / 2, ro_range / 2)
        """
        image_inputs = self.val_preprocess(images=image)
        image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(self.device)
        with torch.no_grad():
            preds = self.model(image_inputs)
        gaus_ax_pred = torch.argmax(preds[:, 0:360], dim=-1)
        gaus_pl_pred = torch.argmax(preds[:, 360:540], dim=-1)
        gaus_ro_pred = torch.argmax(preds[:, 540:540 + self.model_config['ro_range']], dim=-1)
        confidence = F.softmax(preds[:, -2:], dim=-1)[0][0]
        return {
            'phi': float(gaus_ax_pred),
            'theta_model': float(gaus_pl_pred),
            'theta_elev': float(gaus_pl_pred) - 90,
            'theta': 180 - float(gaus_pl_pred),
            'delta': float(gaus_ro_pred) - self.model_config['ro_offset'],
            'confidence': float(confidence),    # confidence < 0.5: no axes was plotted
        }

    @staticmethod
    def get_K(r=0.5, t=0.5, n=3.0, img_w=512, img_h=512):
        fx = (img_w / 2.0) * (n / r)
        fy = (img_h / 2.0) * (n / t)
        cx, cy = img_w / 2.0, img_h / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def get_T_from_R(R):
        R = np.asarray(R)
        T = np.zeros((4, 4))
        T[:3, :3] = R
        T[3, 3] = 1
        return T

    @staticmethod
    def get_R_objw2cam(phi, theta_elev, delta):
        T1 = Homography.get_std_rot("Z", np.deg2rad(phi))
        T2 = Homography.get_std_rot("Y", -np.deg2rad(theta_elev))
        T3 = Homography.get_std_rot("X", -np.deg2rad(delta))
        T_wcs_to_wcs_intermed = T3 @ T2 @ T1
        T_wcs_intermed_to_ccs_facing_origin = np.asarray([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_wcs_to_ccs_facing_origin = T_wcs_intermed_to_ccs_facing_origin @ T_wcs_to_wcs_intermed
        return T_wcs_to_ccs_facing_origin[:3, :3]

    @staticmethod
    def draw_axes_on_image(image, R_objw2cam, radius=16, axes_len=2, axes_width=3):
        T_objw2cam = OrientAny.get_T_from_R(R_objw2cam)
        T5 = Homography.get_std_trans(cz=-radius)
        T_wcs_to_ccs = T5 @ T_objw2cam
        wcs_pts = [
            np.array([0, 0, 0]),
            np.array([axes_len, 0, 0]),
            np.array([0, axes_len, 0]),
            np.array([0, 0, axes_len])
        ]
        ccs_pts = Homography.general_project_A_to_B(wcs_pts, T_wcs_to_ccs)
        K = OrientAny.get_K(r=0.5, t=0.5, n=3.0, img_w=image.width, img_h=image.height)
        pcs_pts, _ = Homography.projectCCStoPCS(ccs_pts, K, image.width, image.height)
        if pcs_pts is None:
            print(f"Warning: No valid projected points. ccs_pts shape: {ccs_pts.shape}")
            return image
        draw = ImageDraw.Draw(image)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx in range(3):
            if len(pcs_pts) > idx + 1:
                start_point = (int(pcs_pts[0][0]), int(pcs_pts[0][1]))
                end_point = (int(pcs_pts[idx + 1][0]), int(pcs_pts[idx + 1][1]))
                draw.line([start_point, end_point], fill=colors[idx], width=axes_width)
        return image, T_wcs_to_ccs

    @staticmethod
    def preprocess_remove_bkg(input_image, do_remove_background):
        input_image = input_image.convert('RGB')
        if not do_remove_background:
            return input_image
        rembg_session = rembg.new_session()
        image = rembg.remove(input_image, session=rembg_session)
        image_array = np.array(image)
        # find non-transparent pixels (ie, foreground), black (0,0,0,0) in background
        alpha = np.where(image_array[..., 3] > 0)
        if len(alpha[0]) == 0:
            # if no foreground found
            return image.convert('RGB')
        # crop to foreground
        y1, y2 = alpha[0].min(), alpha[0].max()
        x1, x2 = alpha[1].min(), alpha[1].max()
        foreground = image_array[y1:y2, x1:x2]
        # create a square image from foreground by padding with zeros
        size = max(foreground.shape[0], foreground.shape[1])
        ph0, pw0 = (size - foreground.shape[0]) // 2, (size - foreground.shape[1]) // 2
        ph1, pw1 = size - foreground.shape[0] - ph0, size - foreground.shape[1] - pw0
        square_image = np.pad(foreground, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))
        # add 15% margins
        ratio = 0.85
        new_size = int(square_image.shape[0] / ratio)
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        final_image = np.pad(square_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))
        return Image.fromarray(final_image).convert('RGB')

    @staticmethod
    def _angles_from_R(R):
        """
        Returns:
        phi in [0, 360)
        theta_elev in [-90, 90]
        delta in (-180, 180]
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError("Expected a 3x3 rotation matrix.")

        # 1) Camera-center direction in world coords: v = -R^T e_z
        v = -R.T @ np.array([0.0, 0.0, 1.0])
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            raise ValueError("Degenerate rotation: cannot extract viewing direction.")
        v /= nv

        # 2) Azimuth and elevation
        phi = math.degrees(math.atan2(v[1], v[0])) % 360.0       # [0, 360)
        theta_elev = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))  # [-90, 90]

        # 3) Zero-roll reference and residual roll about +Z_C
        A2 = np.array([[0, 1, 0],
                       [0, 0, -1],
                       [-1, 0, 0]], dtype=np.float64)
        R1 = Homography.get_std_rot("Z", math.radians(phi))[:3, :3]
        R2 = Homography.get_std_rot("Y", -math.radians(theta_elev))[:3, :3]
        R0 = A2 @ R2 @ R1

        R_delta = R @ R0.T
        delta = math.degrees(math.atan2(R_delta[0, 1], R_delta[0, 0]))
        delta = ((delta + 180.0) % 360.0) - 180.0                # (-180, 180]
        return {'phi': phi, 'theta_elev': theta_elev, 'delta': delta}

    def viz_dir(self, image_dir_path, do_remove_background=True):
        """Visualize 4 images at a time from directory"""
        image_files = sorted(glob.glob(os.path.join(image_dir_path, "*.png")) + glob.glob(os.path.join(image_dir_path, "*.jpg")))

        # Process all images
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            origin_img = Image.open(img_path).convert('RGB')
            rm_bkg_img = self.preprocess_remove_bkg(origin_img, do_remove_background)
            outs = self.get_model_outputs(rm_bkg_img)
            R_objw2cam = self.get_R_objw2cam(outs['phi'], outs['theta_elev'], outs['delta'])
            result_img, _ = self.draw_axes_on_image(rm_bkg_img, R_objw2cam, radius=16, axes_len=2, axes_width=5)
            results.append((result_img, outs, img_path))

        # Visualize 4 at a time
        for i in tqdm(range(0, len(results), 4), desc="Visualizing"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            for j in range(4):
                if i + j < len(results):
                    result_img, outs, img_path = results[i + j]
                    axes[j].imshow(result_img)
                    axes[j].set_title(f"{os.path.basename(img_path)}\nφ:{outs['phi']:.1f}° θ:{outs['theta_elev']:.1f}° δ:{outs['delta']:.1f}° conf:{outs['confidence']:.2f}")
                axes[j].axis('off')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    image_dir = "/robodata/smodak/repos/f3rm/datasets/f3rm/opt/objaverse/car2/images"
    orient_any = OrientAny("ckpts", "ronormsigma1_dino_weight.pt")
    orient_any.viz_dir(image_dir, do_remove_background=False)

    # image_path = "/robodata/smodak/repos/f3rm/datasets/f3rm/custom/betabook/images/frame_00001.png"
    # orient_any = OrientAny("ckpts", "ronormsigma1_dino_weight.pt")
    # origin_img = Image.open(image_path).convert('RGB')
    # rm_bkg_img = orient_any.preprocess_remove_bkg(origin_img, do_remove_background=True)
    # outs = orient_any.get_model_outputs(rm_bkg_img)
    # R_objw2cam = orient_any.get_R_objw2cam(outs['phi'], outs['theta_elev'], outs['delta'])
    # result_img, T_viz_wcs_to_ccs = orient_any.draw_axes_on_image(rm_bkg_img, R_objw2cam, radius=16, axes_len=2)
    # print(f"Azimuth: {round(outs['phi'], 2)}°")
    # print(f"Elevation: {round(outs['theta_elev'], 2)}°")
    # print(f"Rotation: {round(outs['delta'], 2)}°")
    # print(f"Confidence: {round(outs['confidence'], 2)}")
    # # result_img.save("output.png")
    # plt.imshow(result_img)
    # plt.show()

    # debug_angles = orient_any._angles_from_R(R_objw2cam)
    # print(f"Debug angles rot: phi: {debug_angles['phi']}, theta_elev: {debug_angles['theta_elev']}, delta: {debug_angles['delta']}")
    # debug_angles = orient_any._angles_from_R(T_viz_wcs_to_ccs[:3, :3])
    # print(f"Debug angles: phi: {debug_angles['phi']}, theta_elev: {debug_angles['theta_elev']}, delta: {debug_angles['delta']}")

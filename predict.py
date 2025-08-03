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
        preprocessors = {'small': "facebook/dinov2-small", 'base': "facebook/dinov2-base", 'large': "facebook/dinov2-large"}
        self.val_preprocess = AutoImageProcessor.from_pretrained(preprocessors[self.model_config['dino_mode']])

    def get_model_outputs(self, image):
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
            'theta': float(gaus_pl_pred),
            'delta': float(gaus_ro_pred) - self.model_config['ro_offset'],
            'confidence': float(confidence),    # confidence < 0.5: no axes was plotted
        }

    def predict(self, image_path, remove_bg=True):
        origin_img = Image.open(image_path).convert('RGB')
        rm_bkg_img = self.preprocess_remove_bkg(origin_img, remove_bg)
        outs = self.get_model_outputs(rm_bkg_img)
        result_img = self.draw_axes_on_image(rm_bkg_img, phi=outs['phi'], theta=outs['theta'], delta=outs['delta'])
        return result_img, outs

    @staticmethod
    def get_K(r=0.5, t=0.5, n=3.0, img_w=512, img_h=512):
        fx = (img_w / 2.0) * (n / r)
        fy = (img_h / 2.0) * (n / t)
        cx, cy = img_w / 2.0, img_h / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def draw_axes_on_image(self, image, phi, theta, delta, radius=16, axes_len=2):
        T1 = Homography.get_std_rot("Y", np.deg2rad(phi))
        theta_elev = theta - 90
        T2 = Homography.get_std_rot("Z", np.deg2rad(theta_elev))
        T3 = Homography.get_std_trans(cx=radius)
        T_wcs_to_wcs_at_rho_theta_phi = T3 @ T2 @ T1
        T_wcs_at_rho_theta_phi_to_ccs_facing_origin = np.asarray([[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_wcs_at_rho_theta_phi_to_ccs_facing_origin_norot = T_wcs_at_rho_theta_phi_to_ccs_facing_origin @ T_wcs_to_wcs_at_rho_theta_phi
        T_ccs_facing_origin_norot_to_ccs_facing_origin = Homography.get_std_rot("Z", np.deg2rad(delta))
        T_wcs_to_ccs_facing_origin = T_ccs_facing_origin_norot_to_ccs_facing_origin @ T_wcs_at_rho_theta_phi_to_ccs_facing_origin_norot
        wcs_pts = [
            np.array([0, 0, 0]),
            np.array([axes_len, 0, 0]),
            np.array([0, axes_len, 0]),
            np.array([0, 0, axes_len])
        ]
        ccs_pts = Homography.general_project_A_to_B(wcs_pts, T_wcs_to_ccs_facing_origin)
        K = OrientAny.get_K(r=0.5, t=0.5, n=3.0, img_w=image.width, img_h=image.height)
        pcs_pts, _ = Homography.projectCCStoPCS(ccs_pts, K, image.width, image.height)
        draw = ImageDraw.Draw(image)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx in range(3):
            start_point = (int(pcs_pts[0][0]), int(pcs_pts[0][1]))
            end_point = (int(pcs_pts[idx + 1][0]), int(pcs_pts[idx + 1][1]))
            draw.line([start_point, end_point], fill=colors[idx], width=3)
        return image

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


if __name__ == "__main__":
    orient_any = OrientAny("ckpts", "ronormsigma1_dino_weight.pt")
    result_img, outs = orient_any.predict("tt1.png")
    print(f"Azimuth: {round(outs['phi'], 2)}°")
    print(f"Elevation: {round(outs['theta'] - 90, 2)}°")
    print(f"Rotation: {round(outs['delta'], 2)}°")
    print(f"Confidence: {round(outs['confidence'], 2)}")
    result_img.save("output.png")

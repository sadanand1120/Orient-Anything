"""
Orient-Anything: Object Orientation Estimation
Replicates Hugging Face Space demo exactly: https://huggingface.co/spaces/Viglong/Orient-Anything
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoImageProcessor
import rembg

from vision_tower import DINOv2_MLP
from homography import Homography


def get_world_axes_ori_to_cam_facing_origin_ccs_T():
    return np.asarray([[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])


def get_K(r=0.5, t=0.5, n=3.0, img_w=512, img_h=512):
    fx = (img_w / 2.0) * (n / r)
    fy = (img_h / 2.0) * (n / t)
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def draw_thick_bresenham(img, p1, p2, color):
    w, h = img.size
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                xp, yp = x1 + tx, y1 + ty
                if 0 <= xp < w and 0 <= yp < h:
                    img.putpixel((xp, yp), color + (255,))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


class OrientAny:
    MODEL_CONFIG = {
        'cropsmallEx03': {'dino_mode': 'small', 'in_dim': 384, 'out_dim': 722, 'ro_offset': 90, 'ro_range': 180},
        'base100p2': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'base100p': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'base75p2': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'base75p': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'base50p': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'base25p': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'cropbaseEx03': {'dino_mode': 'base', 'in_dim': 768, 'out_dim': 722, 'ro_offset': 90, 'ro_range': 180},
        'celarge': {'dino_mode': 'large', 'in_dim': 1024, 'out_dim': 602, 'ro_offset': 30, 'ro_range': 60},
        'croplargeEX03': {'dino_mode': 'large', 'in_dim': 1024, 'out_dim': 722, 'ro_offset': 90, 'ro_range': 180},
        'croplargeEX2': {'dino_mode': 'large', 'in_dim': 1024, 'out_dim': 722, 'ro_offset': 90, 'ro_range': 180},
        'mixreallarge': {'dino_mode': 'large', 'in_dim': 1024, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360},
        'ronormsigma1': {'dino_mode': 'large', 'in_dim': 1024, 'out_dim': 902, 'ro_offset': 180, 'ro_range': 360}
    }

    def __init__(self, ckpt_dir='ckpts', model_name='croplargeEX2_dino_weight.pt'):
        self.ckpt_path = f"{ckpt_dir}/{model_name}"
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_config = self._get_model_config()
        self._load_model()

    def _get_model_config(self):
        for key, config in self.MODEL_CONFIG.items():
            if key in self.model_name:
                return config
        return None

    def _load_model(self):
        print(f"Loading checkpoint: {self.ckpt_path}")
        self.dino = DINOv2_MLP(dino_mode=self.model_config['dino_mode'],
                               in_dim=self.model_config['in_dim'],
                               out_dim=self.model_config['out_dim'],
                               evaluate=True, mask_dino=False, frozen_back=False)
        self.dino.eval()
        self.dino.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.dino = self.dino.to(self.device)
        preprocessors = {'small': "facebook/dinov2-small", 'base': "facebook/dinov2-base", 'large': "facebook/dinov2-large"}
        self.val_preprocess = AutoImageProcessor.from_pretrained(preprocessors[self.model_config['dino_mode']])

    def get_angles(self, image):
        image_inputs = self.val_preprocess(images=image)
        image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(self.device)
        with torch.no_grad():
            dino_pred = self.dino(image_inputs)
        gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
        gaus_pl_pred = torch.argmax(dino_pred[:, 360:540], dim=-1)
        gaus_ro_pred = torch.argmax(dino_pred[:, 540:540 + self.model_config['ro_range']], dim=-1)
        confidence = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]
        angles = torch.zeros(4)
        angles[0] = gaus_ax_pred
        angles[1] = gaus_pl_pred - 90
        angles[2] = gaus_ro_pred - self.model_config['ro_offset']
        angles[3] = confidence
        return angles

    def predict(self, image_path, remove_bg=True):
        origin_img = Image.open(image_path).convert('RGB')
        rm_bkg_img = self.preprocess_remove_bkg(origin_img, remove_bg)
        angles = self.get_angles(rm_bkg_img)
        phi = np.radians(angles[0])
        theta = np.radians(angles[1])
        gamma = np.radians(angles[2])
        confidence = float(angles[3])
        if confidence > 0.5:
            render_axis = self.render_3d_axes(phi, theta, gamma)
            result_img = self.overlay_axes_on_img(render_axis, rm_bkg_img)
        else:
            result_img = origin_img
        return {
            'result_image': result_img,
            'azimuth': round(float(angles[0]), 2),
            'polar': round(float(angles[1]), 2),
            'rotation': round(float(angles[2]), 2),
            'confidence': round(float(angles[3]), 2)
        }

    @staticmethod
    def render_3d_axes(phi, theta_elev, gamma, radius=16, axes_len=2, width=512, height=512):
        T1 = Homography.get_std_rot("Y", phi)
        T2 = Homography.get_std_rot("Z", theta_elev)
        T3 = Homography.get_std_trans(cx=radius)
        T_wcs_to_wcs_at_rho_theta_phi = T3 @ T2 @ T1
        T_wcs_at_rho_theta_phi_to_ccs_facing_origin = get_world_axes_ori_to_cam_facing_origin_ccs_T()
        T_wcs_to_ccs_facing_origin_norot = T_wcs_at_rho_theta_phi_to_ccs_facing_origin @ T_wcs_to_wcs_at_rho_theta_phi
        T_ccs_facing_origin_norot_to_ccs_facing_origin = Homography.get_std_rot("Z", gamma)
        T_wcs_to_ccs_facing_origin = T_ccs_facing_origin_norot_to_ccs_facing_origin @ T_wcs_to_ccs_facing_origin_norot
        wcs_pts = [np.array([0, 0, 0]), np.array([axes_len, 0, 0]), np.array([0, axes_len, 0]), np.array([0, 0, axes_len])]
        ccs_pts = Homography.general_project_A_to_B(wcs_pts, T_wcs_to_ccs_facing_origin)
        K = get_K(r=0.5, t=0.5, n=3.0, img_w=width, img_h=height)
        pcs_pts, _ = Homography.projectCCStoPCS(ccs_pts, K, width, height)
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx in range(3):
            draw_thick_bresenham(img, pcs_pts[0], pcs_pts[idx + 1], colors[idx])
        return img

    def preprocess_remove_bkg(self, input_image, do_remove_background):
        if not do_remove_background:
            return input_image
        rembg_session = rembg.new_session()
        image = input_image
        if not (image.mode == "RGBA" and image.getextrema()[3][0] < 255):
            image = rembg.remove(image, session=rembg_session)
        image_array = np.array(image)
        alpha = np.where(image_array[..., 3] > 0)
        if len(alpha[0]) == 0:
            return image
        y1, y2 = alpha[0].min(), alpha[0].max()
        x1, x2 = alpha[1].min(), alpha[1].max()
        foreground = image_array[y1:y2, x1:x2]
        size = max(foreground.shape[0], foreground.shape[1])
        ph0, pw0 = (size - foreground.shape[0]) // 2, (size - foreground.shape[1]) // 2
        ph1, pw1 = size - foreground.shape[0] - ph0, size - foreground.shape[1] - pw0
        square_image = np.pad(foreground, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))
        ratio = 0.85
        new_size = int(square_image.shape[0] / ratio)
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        final_image = np.pad(square_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))
        return Image.fromarray(final_image)

    def overlay_axes_on_img(self, foreground_img, background_img, target_size=(512, 512)):
        if foreground_img.mode != "RGBA":
            foreground_img = foreground_img.convert("RGBA")
        if background_img.mode != "RGBA":
            background_img = background_img.convert("RGBA")
        foreground_img = foreground_img.resize(target_size)
        bg_width, bg_height = background_img.size
        scale = target_size[0] / max(bg_width, bg_height)
        new_width = int(bg_width * scale)
        new_height = int(bg_height * scale)
        resized_background = background_img.resize((new_width, new_height))
        pad_width = target_size[0] - new_width
        pad_height = target_size[0] - new_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        resized_background = ImageOps.expand(resized_background, border=(left, top, right, bottom), fill=(255, 255, 255, 255))
        result = resized_background.copy()
        result.paste(foreground_img, (0, 0), mask=foreground_img)
        return result


if __name__ == "__main__":
    orient_any = OrientAny("ckpts", "ronormsigma1_dino_weight.pt")
    result = orient_any.predict("tt2.png")
    print(f"Checkpoint: {orient_any.ckpt_path}")
    print(f"Model: {orient_any.model_name}")
    print(f"Azimuth: {result['azimuth']}°")
    print(f"Polar: {result['polar']}°")
    print(f"Rotation: {result['rotation']}°")
    print(f"Confidence: {result['confidence']}")
    result['result_image'].save("output.png")

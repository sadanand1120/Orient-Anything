"""
Orient-Anything: Object Orientation Estimation
Replicates Hugging Face Space demo exactly: https://huggingface.co/spaces/Viglong/Orient-Anything
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageColor, ImageDraw
from transformers import AutoImageProcessor
import os
import math
import rembg
np.set_printoptions(precision=3, suppress=True)

from vision_tower import DINOv2_MLP
from homography import Homography


def normal_spherical_to_cartesian(rho, phi, theta):
    """
    Convert spherical coordinates to Cartesian coordinates (normal convention)
    rho: radius -> distance from origin, +ve
    phi: azimuth (0-2pi) radians -> with x axis, towards y axis
    theta: polar (0-pi) radians -> with z axis
    """
    r = rho * math.sin(theta)
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = rho * math.cos(theta)
    return x, y, z


def spherical_to_cartesian(rho, phi, theta=None, theta_elev=None):
    """
    Convert spherical coordinates to Cartesian coordinates (orient-anything convention)
    Convention: x forward, y up
    rho: radius -> distance from origin, +ve
    phi: azimuth (0-2pi) radians -> with x axis, towards -z axis
    theta: polar (0-pi) radians -> with -y axis
    theta_elev = theta - math.pi / 2  # Elevation angle, -pi/2 to pi/2
    """
    if theta is not None:
        theta_elev = theta - math.pi / 2
    elif theta_elev is not None:
        pass
    else:
        raise ValueError("Either theta or theta_elev must be provided")

    r = rho * math.cos(theta_elev)
    x = r * math.cos(phi)
    y = r * math.tan(theta_elev)
    z = -r * math.sin(phi)
    return x, y, z


def get_world_axes_ori_to_cam_facing_origin_ccs_T():
    T = np.asarray([
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    return T


def get_K(r=0.5, t=0.5, n=3.0, img_w=512, img_h=512):
    """
    Build an OpenCV-style 3×3 intrinsic matrix K whose focal lengths / principal
    point are consistent with the OpenGL-style projection matrix

        build_proj_matrix(r, t, n, f)

    Parameters
    ----------
    r, t : float
        Half–width (r) and half–height (t) of the near plane, expressed in the
        same world units as `n`.
    n    : float
        Near-plane distance (camera → near clipping plane).
    img_w, img_h : int
        Frame buffer / image resolution in **pixels**.

    Returns
    -------
    K : (3,3) ndarray (dtype float64)
        Intrinsic matrix in the pinhole form
            [[f_x,  0,  c_x],
             [ 0,  f_y, c_y],
             [ 0,   0,   1 ]]
    """
    # focal lengths in pixel units
    fx = (img_w / 2.0) * (n / r)
    fy = (img_h / 2.0) * (n / t)

    # principal point: centre of the viewport
    cx = img_w / 2.0
    cy = img_h / 2.0

    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def build_view_matrix(eye):
    # exactly your f, l, u and view matrix
    f = eye / np.linalg.norm(eye)
    up = np.array([0, -1, 0], dtype=np.float64)
    l = np.cross(up, f)
    l /= np.linalg.norm(l)
    u = np.cross(f, l)
    return np.array([
        [l[0], l[1], l[2], -np.dot(l, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [f[0], f[1], f[2], -np.dot(f, eye)],
        [0, 0, 0, 1.0]
    ], dtype=np.float64)


def build_proj_matrix(r=0.5, t=0.5, n=3.0, f=1000.0):
    # exactly your proj
    return np.array([
        [n / r, 0, 0, 0],
        [0, n / t, 0, 0],
        [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0, 0, -1, 0],
    ], dtype=np.float64)


def project_to_screen(axes, view, proj, width=512, height=512):
    pts = []
    for pt in axes:
        clip = proj @ view @ pt
        if clip[3] != 0:
            ndc = clip / clip[3]
            x = width * 0.5 * (ndc[0] + 1)
            y = height * 0.5 * (ndc[1] + 1)
        else:
            x, y = width / 2, height / 2
        pts.append((x, y))
    return pts


def draw_thick_bresenham(img, p1, p2, color):
    # exactly your triple-loop Bresenham for a 3×3 stencil
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
    def __init__(self, ckpt_dir='ckpts', model_name='croplargeEX2_dino_weight.pt'):
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.ckpt_path = f"{ckpt_dir}/{model_name}"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dino = None
        self.val_preprocess = None
        self.output_dim = self._get_output_dim()
        self._load_model()

    def _load_model(self):
        """Load DINOv2 model and preprocessor"""
        # Check if checkpoint exists
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        print(f"Loading checkpoint: {self.ckpt_path}")

        # Determine input dimension and DINO mode based on model name
        if 'cropsmallEx03' in self.model_name:
            dino_mode = 'small'
            in_dim = 384
        elif 'base' in self.model_name or 'cropbase' in self.model_name:
            dino_mode = 'base'
            in_dim = 768
        else:  # Large models
            dino_mode = 'large'
            in_dim = 1024

        self.dino = DINOv2_MLP(
            dino_mode=dino_mode,
            in_dim=in_dim,
            out_dim=self.output_dim,
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        )

        self.dino.eval()
        self.dino.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.dino = self.dino.to(self.device)

        # Load appropriate preprocessor based on DINO mode
        if dino_mode == 'small':
            self.val_preprocess = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        elif dino_mode == 'base':
            self.val_preprocess = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        else:  # large
            self.val_preprocess = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    def _get_3angle(self, image):
        """Extract 3 angles and confidence from image"""
        image_inputs = self.val_preprocess(images=image)
        image_inputs['pixel_values'] = torch.from_numpy(
            np.array(image_inputs['pixel_values'])
        ).to(self.device)

        with torch.no_grad():
            dino_pred = self.dino(image_inputs)

        gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
        gaus_pl_pred = torch.argmax(dino_pred[:, 360:360 + 180], dim=-1)

        # Handle different rotation ranges based on output dimension
        if self.output_dim == 902:  # 360° rotation (360+180+360+2)
            gaus_ro_pred = torch.argmax(dino_pred[:, 360 + 180:360 + 180 + 360], dim=-1)
            ro_offset = 180  # Maps 0-360° to -180° to 180°
        elif self.output_dim == 602:  # celarge: 60° rotation (360+180+60+2)
            gaus_ro_pred = torch.argmax(dino_pred[:, 360 + 180:360 + 180 + 60], dim=-1)
            ro_offset = 30   # Maps 0-60° to -30° to 30°
        else:  # 722 dimensions: 180° rotation (360+180+180+2)
            gaus_ro_pred = torch.argmax(dino_pred[:, 360 + 180:360 + 180 + 180], dim=-1)
            ro_offset = 90   # Maps 0-180° to -90° to 90°

        confidence = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]

        # # TODO: hijack
        # gaus_ax_pred = 5    # 0-360
        # gaus_pl_pred = 92   # 0-180
        # gaus_ro_pred = 190
        # confidence = 1

        angles = torch.zeros(4)
        angles[0] = gaus_ax_pred                    # Azimuth: 0-360°
        angles[1] = gaus_pl_pred - 90               # Elevation: -90° to 90°
        angles[2] = gaus_ro_pred - ro_offset        # Rotation: varies by checkpoint
        angles[3] = confidence                      # Confidence: 0-1

        return angles

    def _get_output_dim(self):
        """Determine output dimensions based on checkpoint name"""
        # Small models (ViT-S) - 722 dimensions (360+180+180+2)
        if 'cropsmallEx03' in self.model_name:
            return 722  # ViT-S

        # Base models (ViT-B) - 902 dimensions (360+180+360+2) except cropbaseEx03
        elif 'base100p2' in self.model_name:
            return 902  # ViT-B
        elif 'base100p' in self.model_name:
            return 902  # ViT-B
        elif 'base75p2' in self.model_name:
            return 902  # ViT-B
        elif 'base75p' in self.model_name:
            return 902  # ViT-B
        elif 'base50p' in self.model_name:
            return 902  # ViT-B
        elif 'base25p' in self.model_name:
            return 902  # ViT-B
        elif 'cropbaseEx03' in self.model_name:
            return 722  # ViT-B (special case)

        # Large models (ViT-L) - various dimensions
        elif 'celarge' in self.model_name:
            return 602  # ViT-L
        elif 'croplargeEX03' in self.model_name:
            return 722  # ViT-L
        elif 'croplargeEX2' in self.model_name:
            return 722  # ViT-L
        elif 'mixreallarge' in self.model_name:
            return 902  # ViT-L
        elif 'ronormsigma1' in self.model_name:
            return 902  # ViT-L

        else:
            raise ValueError(f"Unknown model: {self.model_name}. Supported models: cropsmallEx03, base100p2, base100p, base75p2, base75p, base50p, base25p, cropbaseEx03, celarge, croplargeEX03, croplargeEX2, mixreallarge, ronormsigma1")

    def predict(self, image_path, remove_bg=True):
        """Predict object orientation from image"""
        # Load and preprocess image
        origin_img = Image.open(image_path).convert('RGB')
        rm_bkg_img = self._background_preprocess_detailed(origin_img, remove_bg)

        # Get angles (all degrees)
        angles = self._get_3angle(rm_bkg_img)

        # Convert to radians for rendering
        phi = np.radians(angles[0])
        theta = np.radians(angles[1])
        gamma = np.radians(angles[2])
        confidence = float(angles[3])

        # Render result
        if confidence > 0.5:
            render_axis = self._render_3D_axis(phi, theta, gamma)
            result_img = self._overlay_images_detailed(render_axis, rm_bkg_img)
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
    def _render_3D_axis(phi, theta, gamma,
                        radius=240, axes_len=30,
                        width=512, height=512,
                        save_path=None):
        # 1) camera
        x_cam, y_cam, z_cam = spherical_to_cartesian(radius, phi, theta_elev=theta)
        eye = np.array([x_cam, y_cam, z_cam])
        eye = -eye   # TODO: why? maybe https://github.com/SpatialVision/Orient-Anything/issues/3#issuecomment-2567585994
        # 2) view & proj
        view = build_view_matrix(eye)
        proj = build_proj_matrix()
        # 3) world axes: origin, X, Y, Z
        axes = [
            np.array([0, 0, 0, 1.0]),
            np.array([axes_len, 0, 0, 1.0]),
            np.array([0, axes_len, 0, 1.0]),
            np.array([0, 0, axes_len, 1.0]),
        ]
        # 4) project → screen coords
        screen_pts = project_to_screen(axes, view, proj, width, height)
        # 5) draw each axis in your original color order
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx in range(3):
            draw_thick_bresenham(img,
                                 screen_pts[0],
                                 screen_pts[idx + 1],
                                 colors[idx])
        # 6) optional save
        if save_path:
            img.save(save_path)
        # 7) final in‑plane roll (exactly as original)
        return img.rotate(np.rad2deg(gamma))

    @staticmethod
    def _render_3D_axis_new(phi, theta_elev, gamma,
                            radius=16, axes_len=2,
                            width=512, height=512,
                            save_path=None):
        T1 = Homography.get_std_rot("Y", phi)
        T2 = Homography.get_std_rot("Z", theta_elev)
        T3 = Homography.get_std_trans(cx=radius)
        T_wcs_to_wcs_at_rho_theta_phi = T3 @ T2 @ T1
        T_wcs_at_rho_theta_phi_to_ccs_facing_origin = get_world_axes_ori_to_cam_facing_origin_ccs_T()
        T_wcs_to_ccs_facing_origin_norot = T_wcs_at_rho_theta_phi_to_ccs_facing_origin @ T_wcs_to_wcs_at_rho_theta_phi
        T_ccs_facing_origin_norot_to_ccs_facing_origin = Homography.get_std_rot("Z", gamma)
        T_wcs_to_ccs_facing_origin = T_ccs_facing_origin_norot_to_ccs_facing_origin @ T_wcs_to_ccs_facing_origin_norot
        wcs_pts = [
            np.array([0, 0, 0]),
            np.array([axes_len, 0, 0]),
            np.array([0, axes_len, 0]),
            np.array([0, 0, axes_len])
        ]
        ccs_pts = Homography.general_project_A_to_B(wcs_pts, T_wcs_to_ccs_facing_origin)
        K = get_K(r=0.5, t=0.5, n=3.0, img_w=width, img_h=height)
        pcs_pts, pcs_mask = Homography.projectCCStoPCS(ccs_pts, K, width, height)
        # import ipdb; ipdb.set_trace()
        pcs_pts = pcs_pts.tolist()
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx in range(3):
            draw_thick_bresenham(img,
                                 pcs_pts[0],
                                 pcs_pts[idx + 1],
                                 colors[idx])
        if save_path:
            img.save(save_path)
        return img

    def _background_preprocess_detailed(self, input_image, do_remove_background):
        """Preprocess image with optional background removal"""
        rembg_session = rembg.new_session() if do_remove_background else None

        if do_remove_background:
            input_image = self._remove_background_detailed(input_image, rembg_session)
            input_image = self._resize_foreground_detailed(input_image, 0.85)

        return input_image

    def _remove_background_detailed(self, image, rembg_session=None, force=False):
        """Remove background from image using rembg"""
        do_remove = True
        if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
            do_remove = False
        do_remove = do_remove or force
        if do_remove:
            image = rembg.remove(image, session=rembg_session)
        return image

    def _resize_foreground_detailed(self, image, ratio):
        """Resize foreground image with padding to maintain aspect ratio"""
        image_array = np.array(image)
        assert image_array.shape[-1] == 4, "Image must be RGBA"

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

        new_size = int(square_image.shape[0] / ratio)
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        final_image = np.pad(square_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))

        return Image.fromarray(final_image)

    def _overlay_images_detailed(self, foreground_img, background_img, target_size=(512, 512)):
        """Overlay foreground image on background with scaling"""
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
    # Use HF Space version (if you have ronormsigma1 checkpoint)
    orient_any = OrientAny("ckpts", "ronormsigma1_dino_weight.pt")

    result = orient_any.predict("tt1.png")
    print(f"Checkpoint: {orient_any.ckpt_path}")
    print(f"Model: {orient_any.model_name}")
    print(f"Azimuth: {result['azimuth']}°")
    print(f"Polar: {result['polar']}°")
    print(f"Rotation: {result['rotation']}°")
    print(f"Confidence: {result['confidence']}")
    result['result_image'].save("output.png")

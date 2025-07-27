"""
Orient-Anything: Object Orientation Estimation
Replicates Hugging Face Space demo exactly: https://huggingface.co/spaces/Viglong/Orient-Anything
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
import os

from vision_tower import DINOv2_MLP
from utils import background_preprocess, render_3D_axis, overlay_images_with_scaling


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

        angles = torch.zeros(4)
        angles[0] = gaus_ax_pred                    # Azimuth: 0-360°
        angles[1] = gaus_pl_pred - 90               # Polar: -90° to 90°
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
        rm_bkg_img = background_preprocess(origin_img, remove_bg)

        # Get angles
        angles = self._get_3angle(rm_bkg_img)

        # Convert to radians for rendering
        phi = np.radians(angles[0])
        theta = np.radians(angles[1])
        gamma = angles[2]
        confidence = float(angles[3])

        # Render result
        if confidence > 0.5:
            render_axis = render_3D_axis(phi, theta, gamma)
            result_img = overlay_images_with_scaling(render_axis, rm_bkg_img)
        else:
            result_img = origin_img

        return {
            'result_image': result_img,
            'azimuth': round(float(angles[0]), 2),
            'polar': round(float(angles[1]), 2),
            'rotation': round(float(angles[2]), 2),
            'confidence': round(float(angles[3]), 2)
        }


if __name__ == "__main__":
    # Use HF Space version (if you have ronormsigma1 checkpoint)
    orient_any = OrientAny("ckpts", "ronormsigma1_dino_weight.pt")
    result = orient_any.predict("tt2.png")

    print(f"Checkpoint: {orient_any.ckpt_path}")
    print(f"Model: {orient_any.model_name}")
    print(f"Azimuth: {result['azimuth']}°")
    print(f"Polar: {result['polar']}°")
    print(f"Rotation: {result['rotation']}°")
    print(f"Confidence: {result['confidence']}")

    result['result_image'].save("output.png")

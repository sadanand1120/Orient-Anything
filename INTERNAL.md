# Orient-Anything: Internal Technical Documentation

## Code Structure Overview

### Core Files
- `app.py`: Main Gradio interface and Hugging Face Space entry point
- `inference.py`: Core inference functions for angle prediction
- `vision_tower.py`: DINOv2-based vision model with MLP head
- `utils.py`: Utility functions for background removal, rendering, and visualization
- `paths.py`: DINO model path constants
- `render/`: Custom 3D rendering module for axis visualization

### Model Architecture
- **Backbone**: DINOv2 (Large variant, 1024-dim features)
- **Head**: MLP with BatchNorm and ReLU activation
- **Output**: 902-dimensional vector (360 + 180 + 360 + 2) - HF Space version
  - 360: Azimuth angle (0-360°)
  - 180: Polar angle (0-180°, mapped to -90° to 90°)
  - 360: Rotation angle (0-360°, mapped to -180° to 180°)
  - 2: Confidence logits

## Technical Pipeline

### 1. Model Initialization (`app.py` lines 10-25)
```python
# Download model checkpoint from Hugging Face
ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", 
                           filename="ronormsigma1/dino_weight.pt", 
                           repo_type="model", cache_dir='./', resume_download=True)

# Initialize DINOv2_MLP model
dino = DINOv2_MLP(
    dino_mode='large',      # Uses DINOv2-large backbone
    in_dim=1024,           # DINOv2-large feature dimension
    out_dim=902,           # 360+180+360+2 output dimensions (HF Space version)
    evaluate=True,         # Evaluation mode
    mask_dino=False,       # No masking during inference
    frozen_back=False      # Backbone not frozen
)

# Load weights and set to device
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
```

### 2. Image Preprocessing Pipeline (`app.py` lines 27-42)

#### 2.1 Background Removal (`utils.py` lines 94-102)
```python
def background_preprocess(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)  # Uses rembg library
        input_image = resize_foreground(input_image, 0.85)          # Resize with 0.85 ratio
    
    return input_image
```

#### 2.2 Foreground Resizing (`utils.py` lines 11-49)
- Crops to foreground bounding box (alpha > 0)
- Pads to square aspect ratio
- Scales by ratio (0.85) to add padding around object
- Returns RGBA image with transparent background

### 3. Inference Pipeline

#### 3.1 Standard Inference (`inference.py` lines 7-22)
```python
def get_3angle(image, dino, val_preprocess, device):
    # Preprocess image with DINOv2 processor
    image_inputs = val_preprocess(images=image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    
    # Forward pass
    with torch.no_grad():
        dino_pred = dino(image_inputs)
    
    # Extract predictions
    gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)           # Azimuth
    gaus_pl_pred = torch.argmax(dino_pred[:, 360:360+180], dim=-1)     # Polar
    gaus_ro_pred = torch.argmax(dino_pred[:, 360+180:360+180+360], dim=-1)  # Rotation (360°)
    confidence = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]           # Confidence
    
    # Convert to angles
    angles = torch.zeros(4)
    angles[0] = gaus_ax_pred                    # Azimuth: 0-360°
    angles[1] = gaus_pl_pred - 90               # Polar: -90° to 90°
    angles[2] = gaus_ro_pred - 180              # Rotation: -180° to 180° (HF Space)
    angles[3] = confidence                      # Confidence: 0-1
    
    return angles
```

#### 3.2 Test-Time Augmentation (`inference.py` lines 24-49)
```python
def get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device):
    # Generate 6 augmented images (3 from original + 3 from background-removed)
    image = get_crop_images(origin_img, num=3) + get_crop_images(rm_bkg_img, num=3)
    
    # Batch inference
    image_inputs = val_preprocess(images=image)
    image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(device)
    
    with torch.no_grad():
        dino_pred = dino(image_inputs)
    
    # Extract predictions for all 6 images
    gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1).to(torch.float32)
    gaus_pl_pred = torch.argmax(dino_pred[:, 360:360+180], dim=-1).to(torch.float32)
    gaus_ro_pred = torch.argmax(dino_pred[:, 360+180:360+180+180], dim=-1).to(torch.float32)
    
    # Remove outliers and average
    gaus_ax_pred = remove_outliers_and_average_circular(gaus_ax_pred)  # Circular averaging for azimuth
    gaus_pl_pred = remove_outliers_and_average(gaus_pl_pred)           # Linear averaging for polar
    gaus_ro_pred = remove_outliers_and_average(gaus_ro_pred)           # Linear averaging for rotation
    
    confidence = torch.mean(F.softmax(dino_pred[:, -2:], dim=-1), dim=0)[0]
    
    # Convert to angles
    angles = torch.zeros(4)
    angles[0] = gaus_ax_pred
    angles[1] = gaus_pl_pred - 90
    angles[2] = gaus_ro_pred - 90
    angles[3] = confidence
    
    return angles
```

### 4. Visualization Pipeline (`app.py` lines 44-52)

#### 4.1 3D Axis Rendering (`utils.py` lines 246-260)
```python
def render_3D_axis(phi, theta, gamma):
    radius = 240
    # Calculate camera position based on angles
    camera_location = [-1*radius * math.cos(phi), -1*radius * math.tan(theta), radius * math.sin(phi)]
    
    # Render 3D axis model
    img = render(
        axis_model,  # Loaded from assets/axis.obj with assets/axis.png texture
        height=512,
        width=512,
        filename="tmp_render.png",
        cam_loc=camera_location
    )
    img = img.rotate(gamma)  # Apply final rotation
    return img
```

#### 4.2 Image Overlay (`utils.py` lines 262-305)
```python
def overlay_images_with_scaling(center_image, background_image, target_size=(512, 512)):
    # Ensure RGBA mode
    if center_image.mode != "RGBA":
        center_image = center_image.convert("RGBA")
    if background_image.mode != "RGBA":
        background_image = background_image.convert("RGBA")
    
    # Resize center image (3D axis) to 512x512
    center_image = center_image.resize(target_size)
    
    # Scale background image to fit
    bg_width, bg_height = background_image.size
    scale = target_size[0] / max(bg_width, bg_height)
    new_width = int(bg_width * scale)
    new_height = int(bg_height * scale)
    resized_background = background_image.resize((new_width, new_height))
    
    # Add padding to center the background
    pad_width = target_size[0] - new_width
    pad_height = target_size[0] - new_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    resized_background = ImageOps.expand(resized_background, border=(left, top, right, bottom), fill=(255,255,255,255))
    
    # Overlay axis on background
    result = resized_background.copy()
    result.paste(center_image, (0, 0), mask=center_image)
    
    return result
```

## Exact Pipeline for Hugging Face Space Demo

### Entry Point: `app.py` infer_func (lines 27-52)
1. **Input**: RGB image array, remove_background checkbox, inference_augmentation checkbox
2. **Background Removal**: 
   - If remove_background=True: Uses `background_preprocess()` with rembg
   - If remove_background=False: No background removal
3. **Inference**:
   - If inference_augmentation=True: Uses `get_3angle_infer_aug()` with 6 augmented images
   - If inference_augmentation=False: Uses `get_3angle()` with single image
4. **Visualization**:
   - If confidence > 0.5: Renders 3D axis and overlays on background-removed image
   - If confidence ≤ 0.5: Returns original image unchanged
5. **Output**: [result_image, azimuth, polar, rotation, confidence]

### Key Differences from Gradio Demo
- **Device**: Uses CPU instead of CUDA for Hugging Face Space compatibility
- **Model Loading**: Downloads checkpoint from Hugging Face Hub automatically
- **Default Settings**: remove_background=True, inference_augmentation=False

## Exact Pipeline for Gradio Demo

### Entry Point: `app.py` (same as HF Space)
- Identical pipeline to Hugging Face Space
- Can be run locally with `python app.py`
- Serves on http://127.0.0.1:7860 by default

## Model Details

### DINOv2_MLP Architecture (`vision_tower.py`)
- **Backbone**: DINOv2-large (1024-dim features)
- **MLP Head**: 
  - Linear(1024, 1024) → BatchNorm1d → ReLU
  - Linear(1024, 720) → BatchNorm1d
- **Output Processing**: Argmax for angles, Softmax for confidence

### Training Data
- 2M rendered labeled images from 3D models
- Azimuth: 0-360° (360 classes)
- Polar: -90° to 90° (180 classes, mapped from 0-180)
- Rotation: -90° to 90° (180 classes, mapped from 0-180)

### Inference Augmentation
- **Random Cropping**: 3 crops from original image (0.8-0.95 scale)
- **Background Removal**: 3 crops from background-removed image
- **Outlier Removal**: IQR-based outlier detection with 1.5 threshold
- **Averaging**: Circular averaging for azimuth, linear for polar/rotation

## Dependencies
- **Core**: torch, transformers, numpy, pillow
- **Background Removal**: rembg
- **Visualization**: matplotlib
- **Interface**: gradio==5.9.0
- **Model Loading**: huggingface-hub
- **3D Rendering**: Custom render module (render/)

## File Dependencies
- `assets/axis.obj`: 3D axis model for rendering
- `assets/axis.png`: Texture for 3D axis
- Model checkpoint: `croplargeEX2/dino_weight.pt` (downloaded from HF Hub) 
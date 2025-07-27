import rembg
import random
import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io
import math
from render import render, Model

# Global axis model for 3D rendering
axis_model = Model("./assets/axis.obj", texture_filename="./assets/axis.png")


def resize_foreground(image: Image, ratio: float) -> Image:
    """Resize foreground image with padding to maintain aspect ratio"""
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )

    # Crop foreground
    fg = image[y1:y2, x1:x2]

    # Pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # Compute padding according to ratio
    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return Image.fromarray(new_image)


def remove_background(image: Image, rembg_session=None, force: bool = False, **rembg_kwargs) -> Image:
    """Remove background from image using rembg"""
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def random_crop(image, crop_scale=(0.8, 0.95)):
    """Random crop image within specified scale range"""
    assert isinstance(image, Image.Image), "Input must be PIL.Image.Image"
    assert len(crop_scale) == 2 and 0 < crop_scale[0] <= crop_scale[1] <= 1

    width, height = image.size
    crop_width = random.randint(int(width * crop_scale[0]), int(width * crop_scale[1]))
    crop_height = random.randint(int(height * crop_scale[0]), int(height * crop_scale[1]))

    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)

    return image.crop((left, top, left + crop_width, top + crop_height))


def get_crop_images(img, num=3):
    """Get multiple randomly cropped versions of image"""
    return [random_crop(img) for _ in range(num)]


def background_preprocess(input_image, do_remove_background):
    """Preprocess image with optional background removal"""
    rembg_session = rembg.new_session() if do_remove_background else None

    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def remove_outliers_and_average(tensor, threshold=1.5):
    """Remove outliers using IQR method and return mean"""
    assert tensor.dim() == 1, "Input tensor must be 1-dimensional"

    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    non_outliers = tensor[(tensor >= lower_bound) & (tensor <= upper_bound)]

    if len(non_outliers) == 0:
        return tensor.mean().item()

    return non_outliers.mean().item()


def remove_outliers_and_average_circular(tensor, threshold=1.5):
    """Remove outliers from circular data (angles) and return mean angle"""
    assert tensor.dim() == 1, "Input tensor must be 1-dimensional"

    # Convert angles to 2D points
    radians = tensor * torch.pi / 180.0
    x_coords = torch.cos(radians)
    y_coords = torch.sin(radians)

    # Calculate mean vector
    mean_x = torch.mean(x_coords)
    mean_y = torch.mean(y_coords)

    differences = torch.sqrt((x_coords - mean_x) * (x_coords - mean_x) +
                             (y_coords - mean_y) * (y_coords - mean_y))

    # Calculate quartiles and IQR
    q1 = torch.quantile(differences, 0.25)
    q3 = torch.quantile(differences, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    # Filter non-outliers
    non_outliers = tensor[(differences >= lower_bound) & (differences <= upper_bound)]

    if len(non_outliers) == 0:
        mean_angle = torch.atan2(mean_y, mean_x) * 180.0 / torch.pi
        mean_angle = (mean_angle + 360) % 360
        return mean_angle

    # Recalculate mean vector for non-outliers
    radians = non_outliers * torch.pi / 180.0
    x_coords = torch.cos(radians)
    y_coords = torch.sin(radians)

    mean_x = torch.mean(x_coords)
    mean_y = torch.mean(y_coords)

    mean_angle = torch.atan2(mean_y, mean_x) * 180.0 / torch.pi
    mean_angle = (mean_angle + 360) % 360

    return mean_angle


def scale(x):
    """Scale vector by factor of 3"""
    return x * 3


def get_proj2D_XYZ(phi, theta, gamma):
    """Get 2D projection of 3D coordinate axes"""
    x = np.array([-1 * np.sin(phi) * np.cos(gamma) - np.cos(phi) * np.sin(theta) * np.sin(gamma),
                  np.sin(phi) * np.sin(gamma) - np.cos(phi) * np.sin(theta) * np.cos(gamma)])
    y = np.array([-1 * np.cos(phi) * np.cos(gamma) + np.sin(phi) * np.sin(theta) * np.sin(gamma),
                  np.cos(phi) * np.sin(gamma) + np.sin(phi) * np.sin(theta) * np.cos(gamma)])
    z = np.array([np.cos(theta) * np.sin(gamma), np.cos(theta) * np.cos(gamma)])

    return scale(x), scale(y), scale(z)


def draw_axis(ax, origin, vector, color, label=None):
    """Draw axis arrow on matplotlib plot"""
    ax.quiver(origin[0], origin[1], vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color)
    if label:
        ax.text(origin[0] + vector[0] * 1.1, origin[1] + vector[1] * 1.1, label, color=color, fontsize=12)


def matplotlib_2D_arrow(angles, rm_bkg_img):
    """Create 2D arrow visualization using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 8))

    phi = np.radians(angles[0])
    theta = np.radians(angles[1])
    gamma = np.radians(-1 * angles[2])

    w, h = rm_bkg_img.size
    if h > w:
        extent = [-5 * w / h, 5 * w / h, -5, 5]
    else:
        extent = [-5, 5, -5 * h / w, 5 * h / w]
    ax.imshow(rm_bkg_img, extent=extent, zorder=0, aspect='auto')

    origin = np.array([0, 0])
    rot_x, rot_y, rot_z = get_proj2D_XYZ(phi, theta, gamma)

    arrow_attr = [{'point': rot_x, 'color': 'r', 'label': 'front'},
                  {'point': rot_y, 'color': 'g', 'label': 'right'},
                  {'point': rot_z, 'color': 'b', 'label': 'top'}]

    if phi > 45 and phi <= 225:
        order = [0, 1, 2]
    elif phi > 225 and phi < 315:
        order = [2, 0, 1]
    else:
        order = [2, 1, 0]

    for i in range(3):
        draw_axis(ax, origin, arrow_attr[order[i]]['point'], arrow_attr[order[i]]['color'], arrow_attr[order[i]]['label'])

    ax.set_axis_off()
    ax.grid(False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)


def figure_to_img(fig):
    """Convert matplotlib figure to PIL Image"""
    with io.BytesIO() as buf:
        fig.savefig(buf, format='JPG', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
    return image


def render_3D_axis(phi, theta, gamma):
    """Render 3D coordinate axes using custom renderer"""
    radius = 240
    camera_location = [-1 * radius * math.cos(phi), -1 * radius * math.tan(theta), radius * math.sin(phi)]
    img = render(
        axis_model,
        height=512,
        width=512,
        filename="tmp_render.png",
        cam_loc=camera_location
    )
    return img.rotate(gamma)


def overlay_images_with_scaling(center_image: Image.Image, background_image, target_size=(512, 512)):
    """Overlay foreground image on background with scaling"""
    if center_image.mode != "RGBA":
        center_image = center_image.convert("RGBA")
    if background_image.mode != "RGBA":
        background_image = background_image.convert("RGBA")

    center_image = center_image.resize(target_size)
    bg_width, bg_height = background_image.size

    scale = target_size[0] / max(bg_width, bg_height)
    new_width = int(bg_width * scale)
    new_height = int(bg_height * scale)
    resized_background = background_image.resize((new_width, new_height))

    pad_width = target_size[0] - new_width
    pad_height = target_size[0] - new_height

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    resized_background = ImageOps.expand(resized_background, border=(left, top, right, bottom), fill=(255, 255, 255, 255))

    result = resized_background.copy()
    result.paste(center_image, (0, 0), mask=center_image)

    return result

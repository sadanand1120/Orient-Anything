"""
Orient-Anything: Object Orientation Estimation
Replicates Hugging Face Space demo exactly: https://huggingface.co/spaces/Viglong/Orient-Anything
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageColor
from transformers import AutoImageProcessor
import os
import math
import rembg
from math import sqrt
from functools import partial
from copy import deepcopy

from vision_tower import DINOv2_MLP


# ============================================================================
# MINIMAL 3D RENDERING IMPLEMENTATION (Independent of utils.py and render/)
# ============================================================================

class Vec2d:
    __slots__ = "x", "y", "arr"

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Vec3d):
            self.arr = Vec3d.narr
        else:
            assert len(args) == 2
            self.arr = list(args)
        self.x, self.y = [d if isinstance(d, int) else int(d + 0.5) for d in self.arr]

    def __repr__(self):
        return f"Vec2d({self.x}, {self.y})"

    def __truediv__(self, other):
        return (self.y - other.y) / (self.x - other.x)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Vec3d:
    __slots__ = "x", "y", "z", "arr"

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Vec4d):
            vec4 = args[0]
            arr_value = (vec4.x, vec4.y, vec4.z)
        else:
            assert len(args) == 3
            arr_value = args
        self.arr = np.array(arr_value, dtype=np.float64)
        self.x, self.y, self.z = self.arr

    def __repr__(self):
        return repr(f"Vec3d({','.join([repr(d) for d in self.arr])})")

    def __sub__(self, other):
        return self.__class__(*[ds - do for ds, do in zip(self.arr, other.arr)])

    def __bool__(self):
        return any(self.arr)


class Mat4d:
    def __init__(self, narr=None, value=None):
        self.value = np.matrix(narr) if value is None else value

    def __repr__(self):
        return repr(self.value)

    def __mul__(self, other):
        return self.__class__(value=self.value * other.value)


class Vec4d(Mat4d):
    def __init__(self, *narr, value=None):
        if value is not None:
            self.value = value
        elif len(narr) == 1 and isinstance(narr[0], Mat4d):
            self.value = narr[0].value
        else:
            assert len(narr) == 4
            self.value = np.matrix([[d] for d in narr])

        self.x, self.y, self.z, self.w = (
            self.value[0, 0], self.value[1, 0], self.value[2, 0], self.value[3, 0]
        )
        self.arr = self.value.reshape((1, 4))


class Canvas:
    def __init__(self, filename=None, height=500, width=500):
        self.filename = filename
        self.height, self.width = height, width
        self.img = Image.new("RGBA", (self.height, self.width), (0, 0, 0, 0))

    def draw(self, dots, color):
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        if isinstance(dots, tuple):
            dots = [dots]
        for dot in dots:
            if dot[0] >= self.height or dot[1] >= self.width or dot[0] < 0 or dot[1] < 0:
                continue
            self.img.putpixel(dot, color + (255,))

    def add_white_border(self, border_size=5):
        if self.img.mode != "RGBA":
            self.img = self.img.convert("RGBA")
        alpha = self.img.getchannel("A")
        dilated_alpha = alpha.filter(ImageFilter.MaxFilter(size=5))
        white_area = Image.new("RGBA", self.img.size, (255, 255, 255, 255))
        white_area.putalpha(dilated_alpha)
        result = Image.alpha_composite(white_area, self.img)
        return result

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.filename:
            self.img.save(self.filename)


class Model:
    def __init__(self, filename, texture_filename):
        self.vertices = []
        self.uv_vertices = []
        self.uv_indices = []
        self.indices = []

        texture = Image.open(texture_filename)
        self.texture_array = np.array(texture)
        self.texture_width, self.texture_height = texture.size

        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")]
                    self.vertices.append(Vec4d(x, y, z, 1))
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")]
                    self.uv_vertices.append([u, v])
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0]) for d in facet])
                    self.uv_indices.append([int(d[1]) for d in facet])


def normalize(x, y, z):
    unit = sqrt(x * x + y * y + z * z)
    if unit == 0:
        return 0, 0, 0
    return x / unit, y / unit, z / unit


def dot_product(a0, a1, a2, b0, b1, b2):
    return a0 * b0 + a1 * b1 + a2 * b2


def cross_product(a0, a1, a2, b0, b1, b2):
    x = a1 * b2 - a2 * b1
    y = a2 * b0 - a0 * b2
    z = a0 * b1 - a1 * b0
    return x, y, z


def get_min_max(a, b, c):
    min_val = a
    max_val = a
    if min_val > b:
        min_val = b
    if min_val > c:
        min_val = c
    if max_val < b:
        max_val = b
    if max_val < c:
        max_val = c
    return int(min_val), int(max_val)


def generate_faces(triangles, width, height):
    i, j, k, length = 0, 0, 0, 0
    bcy, bcz, x, y, z = 0., 0., 0., 0., 0.
    a, b, c = [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]
    m, bc = [0., 0., 0.], [0., 0., 0.]
    uva, uvb, uvc = [0., 0.], [0., 0.], [0., 0.]
    minx, maxx, miny, maxy = 0, 0, 0, 0
    length = triangles.shape[0]
    zbuffer = {}
    faces = []

    for i in range(length):
        a = triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2]
        b = triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2]
        c = triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2]
        uva = triangles[i, 0, 3], triangles[i, 0, 4]
        uvb = triangles[i, 1, 3], triangles[i, 1, 4]
        uvc = triangles[i, 2, 3], triangles[i, 2, 4]
        minx, maxx = get_min_max(a[0], b[0], c[0])
        miny, maxy = get_min_max(a[1], b[1], c[1])
        pixels = []
        for j in range(minx, maxx + 2):
            for k in range(miny - 1, maxy + 2):
                x = j
                y = k
                m[0], m[1], m[2] = cross_product(c[0] - a[0], b[0] - a[0], a[0] - x, c[1] - a[1], b[1] - a[1], a[1] - y)
                if abs(m[2]) > 0:
                    bcy = m[1] / m[2]
                    bcz = m[0] / m[2]
                    bc = (1 - bcy - bcz, bcy, bcz)
                else:
                    continue
                if bc[0] < -0.00001 or bc[1] < -0.00001 or bc[2] < -0.00001:
                    continue
                z = 1 / (bc[0] / a[2] + bc[1] / b[2] + bc[2] / c[2])
                v = (uva[0] * bc[0] / a[2] + uvb[0] * bc[1] / b[2] + uvc[0] * bc[2] / c[2]) * z * width
                u = height - (uva[1] * bc[0] / a[2] + uvb[1] * bc[1] / b[2] + uvc[1] * bc[2] / c[2]) * z * height
                idx = ((x + y) * (x + y + 1) + y) / 2
                if zbuffer.get(idx) is None or zbuffer[idx] < z:
                    zbuffer[idx] = z
                    pixels.append((i, j, k, int(u) - 1, int(v) - 1))
        faces.append(pixels)
    return faces


def draw_line(v1, v2, canvas, color="white"):
    v1, v2 = deepcopy(v1), deepcopy(v2)
    if v1 == v2:
        canvas.draw((v1.x, v1.y), color=color)
        return

    steep = abs(v1.y - v2.y) > abs(v1.x - v2.x)
    if steep:
        v1.x, v1.y = v1.y, v1.x
        v2.x, v2.y = v2.y, v2.x
    v1, v2 = (v1, v2) if v1.x < v2.x else (v2, v1)
    slope = abs((v1.y - v2.y) / (v1.x - v2.x))
    y = v1.y
    error = 0
    incr = 1 if v1.y < v2.y else -1
    dots = []
    for x in range(int(v1.x), int(v2.x + 0.5)):
        dots.append((int(y), x) if steep else (x, int(y)))
        error += slope
        if abs(error) >= 0.5:
            y += incr
            error -= 1
    canvas.draw(dots, color=color)


def draw_triangle(v1, v2, v3, canvas, color, wireframe=False):
    _draw_line = partial(draw_line, canvas=canvas, color=color)

    if wireframe:
        _draw_line(v1, v2)
        _draw_line(v2, v3)
        _draw_line(v1, v3)
        return

    def sort_vertices_asc_by_y(vertices):
        return sorted(vertices, key=lambda v: v.y)

    def fill_bottom_flat_triangle(v1, v2, v3):
        invslope1 = (v2.x - v1.x) / (v2.y - v1.y)
        invslope2 = (v3.x - v1.x) / (v3.y - v1.y)
        x1 = x2 = v1.x
        y = v1.y
        while y <= v2.y:
            _draw_line(Vec2d(x1, y), Vec2d(x2, y))
            x1 += invslope1
            x2 += invslope2
            y += 1

    def fill_top_flat_triangle(v1, v2, v3):
        invslope1 = (v3.x - v1.x) / (v3.y - v1.y)
        invslope2 = (v3.x - v2.x) / (v3.y - v2.y)
        x1 = x2 = v3.x
        y = v3.y
        while y > v2.y:
            _draw_line(Vec2d(x1, y), Vec2d(x2, y))
            x1 -= invslope1
            x2 -= invslope2
            y -= 1

    v1, v2, v3 = sort_vertices_asc_by_y((v1, v2, v3))

    if v1.y == v2.y == v3.y:
        pass
    elif v2.y == v3.y:
        fill_bottom_flat_triangle(v1, v2, v3)
    elif v1.y == v2.y:
        fill_top_flat_triangle(v1, v2, v3)
    else:
        v4 = Vec2d(int(v1.x + (v2.y - v1.y) / (v3.y - v1.y) * (v3.x - v1.x)), v2.y)
        fill_bottom_flat_triangle(v1, v2, v4)
        fill_top_flat_triangle(v2, v4, v3)


def normalize_vec3d(v):
    return Vec3d(*normalize(*v.arr))


def dot_product_vec3d(a, b):
    return dot_product(*a.arr, *b.arr)


def cross_product_vec3d(a, b):
    return Vec3d(*cross_product(*a.arr, *b.arr))


BASE_LIGHT = 0.9


def get_light_intensity(face):
    lights = [Vec3d(-2, 4, -10)]
    v1, v2, v3 = face
    up = normalize_vec3d(cross_product_vec3d(v2 - v1, v3 - v1))
    intensity = BASE_LIGHT
    for light in lights:
        intensity += dot_product_vec3d(up, normalize_vec3d(light)) * 0.2
    return intensity


def look_at(eye, target, up=Vec3d(0, -1, 0)):
    f = normalize_vec3d(eye - target)
    l = normalize_vec3d(cross_product_vec3d(up, f))
    u = cross_product_vec3d(f, l)

    rotate_matrix = Mat4d([[l.x, l.y, l.z, 0], [u.x, u.y, u.z, 0], [f.x, f.y, f.z, 0], [0, 0, 0, 1.0]])
    translate_matrix = Mat4d([[1, 0, 0, -eye.x], [0, 1, 0, -eye.y], [0, 0, 1, -eye.z], [0, 0, 0, 1.0]])

    return Mat4d(value=(rotate_matrix * translate_matrix).value)


def perspective_project(r, t, n, f, b=None, l=None):
    return Mat4d([[n / r, 0, 0, 0], [0, n / t, 0, 0], [0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)], [0, 0, -1, 0]])


def draw_with_z_buffer(screen_vertices, world_vertices, model, canvas):
    intensities = []
    triangles = []
    for i, triangle_indices in enumerate(model.indices):
        screen_triangle = [screen_vertices[idx - 1] for idx in triangle_indices]
        uv_triangle = [model.uv_vertices[idx - 1] for idx in model.uv_indices[i]]
        world_triangle = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        intensities.append(abs(get_light_intensity(world_triangle)))
        triangles.append([np.append(screen_triangle[i].arr, uv_triangle[i]) for i in range(3)])

    faces = generate_faces(np.array(triangles, dtype=np.float64), model.texture_width, model.texture_height)
    for face_dots in faces:
        for dot in face_dots:
            intensity = intensities[dot[0]]
            u, v = dot[3], dot[4]
            color = model.texture_array[u, v]
            canvas.draw((dot[1], dot[2]), tuple(int(c * intensity) for c in color[:3]))


def render_3d_model(model, height, width, filename, cam_loc, wireframe=False):
    model_matrix = Mat4d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    view_matrix = look_at(Vec3d(cam_loc[0], cam_loc[1], cam_loc[2]), Vec3d(0, 0, 0))
    projection_matrix = perspective_project(0.5, 0.5, 3, 1000)

    world_vertices = []

    def mvp(v):
        world_vertex = model_matrix * v
        world_vertices.append(Vec4d(world_vertex))
        return projection_matrix * view_matrix * world_vertex

    def ndc(v):
        v = v.value
        w = v[3, 0]
        x, y, z = v[0, 0] / w, v[1, 0] / w, v[2, 0] / w
        return Mat4d([[x], [y], [z], [1 / w]])

    def viewport(v):
        x = y = 0
        w, h = width, height
        n, f = 0.3, 1000
        return Vec3d(
            w * 0.5 * v.value[0, 0] + x + w * 0.5,
            h * 0.5 * v.value[1, 0] + y + h * 0.5,
            0.5 * (f - n) * v.value[2, 0] + 0.5 * (f + n),
        )

    screen_vertices = [viewport(ndc(mvp(v))) for v in model.vertices]

    with Canvas(filename, height, width) as canvas:
        draw_with_z_buffer(screen_vertices, world_vertices, model, canvas)
        render_img = canvas.add_white_border().copy()
    return render_img


# ============================================================================
# ORIENT-ANYTHING CLASS
# ============================================================================

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

        # Load 3D axis model for rendering
        self.axis_model = Model("./assets/axis.obj", texture_filename="./assets/axis.png")

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
        rm_bkg_img = self._background_preprocess_detailed(origin_img, remove_bg)

        # Get angles
        angles = self._get_3angle(rm_bkg_img)

        # Convert to radians for rendering
        phi = np.radians(angles[0])
        theta = np.radians(angles[1])
        gamma = angles[2]
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

    def _render_3D_axis(self, phi, theta, gamma):
        """Render 3D coordinate axes using custom renderer"""
        radius = 240
        camera_location = [-1 * radius * math.cos(phi), -1 * radius * math.tan(theta), radius * math.sin(phi)]
        img = render_3d_model(
            self.axis_model,
            height=512,
            width=512,
            filename="tmp_render.png",
            cam_loc=camera_location
        )
        return img.rotate(gamma)

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
    result = orient_any.predict("tt2.png")

    print(f"Checkpoint: {orient_any.ckpt_path}")
    print(f"Model: {orient_any.model_name}")
    print(f"Azimuth: {result['azimuth']}°")
    print(f"Polar: {result['polar']}°")
    print(f"Rotation: {result['rotation']}°")
    print(f"Confidence: {result['confidence']}")

    result['result_image'].save("output.png")

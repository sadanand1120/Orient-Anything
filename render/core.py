import typing as t
from functools import partial
import numpy as np
from copy import deepcopy
from .canvas import Canvas
from . import speedup


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


def draw_line(v1: Vec2d, v2: Vec2d, canvas: Canvas, color: t.Union[tuple, str] = "white"):
    """Draw line using Bresenham's algorithm"""
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
    error: float = 0
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
    """Draw triangle with 3 ordered vertices"""
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
        """False for zero vector (0, 0, 0)"""
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
            self.value[0, 0],
            self.value[1, 0],
            self.value[2, 0],
            self.value[3, 0],
        )
        self.arr = self.value.reshape((1, 4))


def normalize(v: Vec3d):
    """Normalize 3D vector"""
    return Vec3d(*speedup.normalize(*v.arr))


def dot_product(a: Vec3d, b: Vec3d):
    """Calculate dot product of two 3D vectors"""
    return speedup.dot_product(*a.arr, *b.arr)


def cross_product(a: Vec3d, b: Vec3d):
    """Calculate cross product of two 3D vectors"""
    return Vec3d(*speedup.cross_product(*a.arr, *b.arr))


BASE_LIGHT = 0.9


def get_light_intensity(face) -> float:
    """Calculate lighting intensity for face"""
    lights = [Vec3d(-2, 4, -10)]

    v1, v2, v3 = face
    up = normalize(cross_product(v2 - v1, v3 - v1))
    intensity = BASE_LIGHT
    for light in lights:
        intensity += dot_product(up, normalize(light)) * 0.2
    return intensity


def look_at(eye: Vec3d, target: Vec3d, up: Vec3d = Vec3d(0, -1, 0)) -> Mat4d:
    """Create look-at view matrix"""
    f = normalize(eye - target)
    l = normalize(cross_product(up, f))
    u = cross_product(f, l)

    rotate_matrix = Mat4d(
        [[l.x, l.y, l.z, 0], [u.x, u.y, u.z, 0], [f.x, f.y, f.z, 0], [0, 0, 0, 1.0]]
    )
    translate_matrix = Mat4d(
        [[1, 0, 0, -eye.x], [0, 1, 0, -eye.y], [0, 0, 1, -eye.z], [0, 0, 0, 1.0]]
    )

    return Mat4d(value=(rotate_matrix * translate_matrix).value)


def perspective_project(r, t, n, f, b=None, l=None):
    """Create perspective projection matrix"""
    return Mat4d(
        [
            [n / r, 0, 0, 0],
            [0, n / t, 0, 0],
            [0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]
    )


def draw(screen_vertices, world_vertices, model, canvas, wireframe=True):
    """Draw model using standard algorithm"""
    for triangle_indices in model.indices:
        vertex_group = [screen_vertices[idx - 1] for idx in triangle_indices]
        face = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        if wireframe:
            draw_triangle(*vertex_group, canvas=canvas, color="black", wireframe=True)
        else:
            intensity = get_light_intensity(face)
            if intensity > 0:
                draw_triangle(
                    *vertex_group, canvas=canvas, color=(int(intensity * 255),) * 3
                )


def draw_with_z_buffer(screen_vertices, world_vertices, model, canvas):
    """Draw model using z-buffer algorithm"""
    intensities = []
    triangles = []
    for i, triangle_indices in enumerate(model.indices):
        screen_triangle = [screen_vertices[idx - 1] for idx in triangle_indices]
        uv_triangle = [model.uv_vertices[idx - 1] for idx in model.uv_indices[i]]
        world_triangle = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        intensities.append(abs(get_light_intensity(world_triangle)))
        triangles.append(
            [np.append(screen_triangle[i].arr, uv_triangle[i]) for i in range(3)]
        )

    faces = speedup.generate_faces(
        np.array(triangles, dtype=np.float64), model.texture_width, model.texture_height
    )
    for face_dots in faces:
        for dot in face_dots:
            intensity = intensities[dot[0]]
            u, v = dot[3], dot[4]
            color = model.texture_array[u, v]
            canvas.draw((dot[1], dot[2]), tuple(int(c * intensity) for c in color[:3]))


def render(model, height, width, filename, cam_loc, wireframe=False):
    """Render 3D model to image"""
    model_matrix = Mat4d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    view_matrix = look_at(Vec3d(cam_loc[0], cam_loc[1], cam_loc[2]), Vec3d(0, 0, 0))
    projection_matrix = perspective_project(0.5, 0.5, 3, 1000)

    world_vertices = []

    def mvp(v):
        world_vertex = model_matrix * v
        world_vertices.append(Vec4d(world_vertex))
        return projection_matrix * view_matrix * world_vertex

    def ndc(v):
        """Convert to normalized device coordinates"""
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
        if wireframe:
            draw(screen_vertices, world_vertices, model, canvas)
        else:
            draw_with_z_buffer(screen_vertices, world_vertices, model, canvas)

        render_img = canvas.add_white_border().copy()
    return render_img

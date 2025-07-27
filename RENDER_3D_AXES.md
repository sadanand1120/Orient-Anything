# 3D Axis Rendering Pipeline Documentation

## Overview

This document explains the complete pipeline from the model's 3D orientation prediction (3 angles) to the final rendered 3D coordinate axes overlaid on the input image.

## Input: Model Predictions

The model outputs 3 angles representing object orientation:

- **φ (phi)**: Azimuth angle [0°, 360°] - rotation around Y-axis
- **θ (theta)**: Polar angle [-90°, 90°] - elevation from horizontal plane  
- **γ (gamma)**: Roll angle [varies by model] - rotation around Z-axis

## Pipeline Steps

### 1. Camera Positioning

**Purpose**: Position virtual camera to view the 3D axis model from the predicted orientation.

**Mathematical Formula**:
```
radius = 240
camera_location = [
    -radius × cos(φ),
    -radius × tan(θ), 
    radius × sin(φ)
]
```

**Explanation**:
- Camera orbits around origin at fixed radius (240 units)
- φ controls horizontal position (X-Z plane)
- θ controls vertical position (Y-axis)
- Camera always looks toward origin (0,0,0)

### 2. 3D Model Loading

**Assets**:
- `axis.obj`: 3D mesh file containing coordinate axes geometry
- `axis.png`: Texture file for coloring the axes

**Model Structure**:
- **Vertices**: 3D points defining axis geometry
- **UV Coordinates**: Texture mapping coordinates
- **Faces**: Triangular polygons with vertex indices
- **Texture**: RGB image for axis coloring

### 3. 3D Graphics Pipeline

#### 3.1 Model-View-Projection (MVP) Transform

**Model Matrix (M)**:
```
M = Identity Matrix (no model transformation)
```

**View Matrix (V)**:
```
eye = camera_location
target = (0, 0, 0)
up = (0, -1, 0)

f = normalize(eye - target)     // Forward vector
l = normalize(up × f)           // Left vector  
u = f × l                       // Up vector

V = [l.x  l.y  l.z  0] [1  0  0  -eye.x]
    [u.x  u.y  u.z  0] [0  1  0  -eye.y] 
    [f.x  f.y  f.z  0] [0  0  1  -eye.z]
    [0    0    0    1] [0  0  0    1  ]
```

**Projection Matrix (P)**:
```
r = 0.5, t = 0.5, n = 3, f = 1000

P = [n/r   0     0             0    ]
    [0    n/t    0             0    ]
    [0     0   -(f+n)/(f-n)  -2fn/(f-n)]
    [0     0     -1            0    ]
```

**MVP Transform**:
```
vertex_clip = P × V × M × vertex_world
```

#### 3.2 Perspective Division

**Normalized Device Coordinates (NDC)**:
```
vertex_ndc = vertex_clip / vertex_clip.w
```

#### 3.3 Viewport Transform

**Screen Coordinates**:
```
width = 512, height = 512
x_screen = (x_ndc + 1) × width/2
y_screen = (y_ndc + 1) × height/2  
z_screen = (z_ndc + 1) × (far-near)/2 + (far+near)/2
```

### 4. Triangle Rasterization

#### 4.1 Triangle Setup

For each triangle face:
1. **Screen Triangle**: Projected vertices in screen space
2. **World Triangle**: Original 3D vertices for lighting
3. **UV Triangle**: Texture coordinates for each vertex

#### 4.2 Barycentric Coordinates

**Point-in-Triangle Test**:
```
Given point P and triangle vertices A, B, C:

v0 = C - A
v1 = B - A  
v2 = P - A

d00 = v0 · v0
d01 = v0 · v1
d11 = v1 · v1
d20 = v2 · v0
d21 = v2 · v1

denom = d00 × d11 - d01 × d01
v = (d11 × d20 - d01 × d21) / denom
w = (d00 × d21 - d01 × d20) / denom
u = 1.0 - v - w

Point P is inside triangle if: u ≥ 0, v ≥ 0, w ≥ 0
```

#### 4.3 Z-Buffer Algorithm

**Depth Testing**:
```
For each pixel (x, y):
    z_interpolated = u×z_A + v×z_B + w×z_C
    
    if z_interpolated > z_buffer[x, y]:
        z_buffer[x, y] = z_interpolated
        render_pixel(x, y)
```

### 5. Lighting and Texturing

#### 5.1 Lighting Calculation

**Light Sources**:
```
lights = [(-2, 4, -10)]  // Single directional light
base_light = 0.9
```

**Per-Face Lighting**:
```
v1, v2, v3 = triangle_vertices
normal = normalize((v2 - v1) × (v3 - v1))

intensity = base_light
for light in lights:
    light_dir = normalize(light)
    intensity += dot(normal, light_dir) × 0.2
```

#### 5.2 Texture Mapping

**UV Interpolation**:
```
u_interpolated = u×u_A + v×u_B + w×u_C
v_interpolated = u×v_A + v×v_B + w×v_C
```

**Perspective Correction**:
```
u_corrected = (u_A/z_A + v×u_B/z_B + w×u_C/z_C) × z_interpolated
v_corrected = (v_A/z_A + v×v_B/z_B + w×v_C/z_C) × z_interpolated
```

**Final Color**:
```
texture_color = sample_texture(u_corrected, v_corrected)
final_color = texture_color × intensity
```

### 6. Canvas Rendering

#### 6.1 Pixel Drawing

**Bresenham's Line Algorithm** (for wireframe mode):
```
Given line from (x1, y1) to (x2, y2):
    dx = |x2 - x1|
    dy = |y2 - y1|
    steep = dy > dx
    
    if steep:
        swap(x1, y1), swap(x2, y2)
    
    if x1 > x2:
        swap(x1, x2), swap(y1, y2)
    
    y = y1
    error = 0
    y_step = 1 if y1 < y2 else -1
    
    for x in range(x1, x2+1):
        draw_pixel(y, x) if steep else draw_pixel(x, y)
        error += dy/dx
        if error >= 0.5:
            y += y_step
            error -= 1
```

#### 6.2 Triangle Filling

**Scanline Algorithm**:
1. Sort vertices by Y-coordinate
2. Fill bottom flat triangle
3. Fill top flat triangle
4. Use linear interpolation for X-coordinates

### 7. Post-Processing

#### 7.1 White Border Addition

**Alpha Channel Dilation**:
```
dilated_alpha = alpha_channel.filter(MaxFilter(size=5))
white_border = Image.new("RGBA", size, (255,255,255,255))
white_border.putalpha(dilated_alpha)
result = alpha_composite(white_border, rendered_image)
```

#### 7.2 Final Rotation

**Image Rotation**:
```
final_image = rendered_image.rotate(γ)
```

### 8. Image Overlay

#### 8.1 Background Scaling

**Aspect Ratio Preservation**:
```
scale = 512 / max(background_width, background_height)
new_width = background_width × scale
new_height = background_height × scale
```

#### 8.2 Centering and Padding

**Padding Calculation**:
```
pad_width = 512 - new_width
pad_height = 512 - new_height
left = pad_width // 2
top = pad_height // 2
right = pad_width - left
bottom = pad_height - top
```

#### 8.3 Final Composition

**Alpha Blending**:
```
result = background_image.copy()
result.paste(axis_image, (0, 0), mask=axis_image)
```

## Mathematical Summary

The complete pipeline transforms 3 angles (φ, θ, γ) into a 3D coordinate visualization through:

1. **Camera Positioning**: Spherical coordinates → Cartesian camera position
2. **3D Transformations**: World → View → Projection → Screen coordinates  
3. **Rasterization**: 3D triangles → 2D pixels with depth testing
4. **Lighting & Texturing**: Surface normals → illumination → texture sampling
5. **Compositing**: 3D render → 2D overlay on input image

The result is a photorealistic 3D coordinate system that accurately represents the predicted object orientation. 
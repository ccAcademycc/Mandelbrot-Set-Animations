import os
import numpy as np
from PIL import Image
from numba import cuda
from scipy.interpolate import interp1d

# Gradient creation function
def make_gradient(colors, interpolation):
    """
    Creates a gradient function based on specified colors and interpolation type.
    Colors should be given in the 0-255 RGB range.
    """
    X = [i / (len(colors) - 1) for i in range(len(colors))]
    Y = [[color[i] for color in colors] for i in range(3)]
    channels = [interp1d(X, y, kind=interpolation) for y in Y]
    return lambda x: [int(np.clip(channel(x), 0, 255)) for channel in channels]

# Define colors in the 0-255 RGB range
black = (0, 0, 0)
white = (255, 255, 255)
purple = (204, 179, 255)

# Define color stops for the gradient
colors = [white, purple, black, black]
gradient = make_gradient(colors, interpolation="cubic")

# Create a palette of 256 colors
num_colors = 256
palette = [gradient(i / num_colors) for i in range(num_colors)]
palette_array = np.array(palette, dtype=np.uint8)

# GPU Kernel to calculate Mandelbrot set for each pixel
@cuda.jit
def mandelbrot_kernel(center_real, center_imag, scale, width, height, max_iterations, image, palette):
    """
    Kernel function to compute Mandelbrot set values for each pixel in parallel.
    Colors each pixel according to the number of iterations to escape or reaches max_iterations.
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        c_real = center_real + scale * (x - width / 2)
        c_imag = center_imag + scale * (height / 2 - y)
        z_real = 0.0
        z_imag = 0.0
        iteration = 0
        while z_real * z_real + z_imag * z_imag <= 4.0 and iteration < max_iterations:
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag = 2.0 * z_real * z_imag + c_imag
            z_real = z_real_new
            iteration += 1
        if iteration == max_iterations:
            # Point is in the Mandelbrot set (black)
            image[y, x, 0] = 0  # R
            image[y, x, 1] = 0  # G
            image[y, x, 2] = 0  # B
        else:
            # Color based on iteration count
            color = palette[int(iteration % 256)-1]
            image[y, x, 0] = color[0]  # R
            image[y, x, 1] = color[1]  # G
            image[y, x, 2] = color[2]  # B

# Generate a single frame
def generate_frame(center: complex, scale: float, width: int, height: int, max_iterations: int, filename: str):
    """
    Generates a Mandelbrot set image frame and saves it to the specified filename.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for RGB
    blockdim = (16, 16)
    griddim = (width // blockdim[0] + 1, height // blockdim[1] + 1)
    mandelbrot_kernel[griddim, blockdim](center.real, center.imag, scale, width, height, max_iterations, image, palette_array)
    image = Image.fromarray(image)
    image.save(filename)

# Parameters for generating frames
# Starting center coordinates for the Mandelbrot set
start_center = complex(-0.226266648, 1.11617444)
# Fixed center for this rendering sequence
end_center = start_center

# Zoom scales (initial and final) for the transformation
initial_scale = 9e-10
final_scale = 2e-10

# Number of frames in the animation
num_frames = 600

# Frame resolution (8K) — adjust if rendering performance is slow (e.g., 1920, 1080 for Full HD)
width, height = 7680, 4320

# Prepare the output directory
output_folder = "mandelbrot_increase_iterations_2"
os.makedirs(output_folder, exist_ok=True)

# Interpolate scales and center positions for smooth transformation
scales = np.geomspace(initial_scale, final_scale, num_frames)
centers_real = np.linspace(start_center.real, end_center.real, num_frames)
centers_imag = np.linspace(start_center.imag, end_center.imag, num_frames)

# Resume frame generation from a specific start frame, if needed
start_frame = 0

# Generate each frame with a fixed center and interpolated scale
for i in range(start_frame, num_frames):
    scale = scales[i]
    center = complex(centers_real[i], centers_imag[i])
    filename = os.path.join(output_folder, f"{i:05d}.png")
    generate_frame(center, scale, width, height, i, filename)
    print(f"Generated {filename} at center {center} with scale {scale}")

print("All frames generated.")

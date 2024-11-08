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
red = (255, 0, 0)
white = (255, 255, 255)

# Define color stops for the gradient
colors = [black, red, white, black]
gradient = make_gradient(colors, interpolation="cubic")

# Create a palette of 256 colors
num_colors = 256
palette = [gradient(i / num_colors) for i in range(num_colors)]
palette_array = np.array(palette, dtype=np.uint8)

# GPU Kernel for Julia set calculation
@cuda.jit
def julia_kernel(c_real, c_imag, scale, width, height, max_iterations, image, palette):
    """
    Kernel function to compute Julia set values for each pixel in parallel.
    Colors each pixel according to the number of iterations to escape or reach max_iterations.
    """
    x, y = cuda.grid(2)
    if x < width and y < height:
        z_real = scale * (x - width / 2)
        z_imag = scale * (height / 2 - y)
        iteration = 0
        while z_real * z_real + z_imag * z_imag <= 4.0 and iteration < max_iterations:
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag = 2.0 * z_real * z_imag + c_imag
            z_real = z_real_new
            iteration += 1
        if iteration == max_iterations:
            image[y, x, 0] = 255  # R
            image[y, x, 1] = 255  # G
            image[y, x, 2] = 255  # B
        else:
            color = palette[int(iteration % 256)]
            image[y, x, 0] = color[0]  # R
            image[y, x, 1] = color[1]  # G
            image[y, x, 2] = color[2]  # B

# Generate a single frame for the Julia set
def generate_frame(c, scale: float, width: int, height: int, max_iterations: int, filename: str):
    """
    Generates a Julia set image frame and saves the image.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for RGB
    blockdim = (16, 16)
    griddim = (width // blockdim[0] + 1, height // blockdim[1] + 1)
    julia_kernel[griddim, blockdim](c.real, c.imag, scale, width, height, max_iterations, image, palette_array)
    image = Image.fromarray(image).convert("RGBA")  # Convert to RGBA for transparency support
    image = image.convert("RGB")  # Convert back to RGB before saving
    image.save(filename)

# Parameters
# Maximum iterations for Julia set calculation
max_iterations = 6000

# Image resolution
width, height = 7680, 4320

# Fixed scale for Julia set
scale = 0.001

# Prepare the output directory
output_folder = "julia_fixed_point"
os.makedirs(output_folder, exist_ok=True)

# Generate frames for specific values of 'c' in the Julia set
frame_count = 0

# Fixed complex number for Julia set generation
c = complex( -0.549047586, -0.562183818)

filename = os.path.join(output_folder, f"{frame_count:05d}.png")
generate_frame(c, scale, width, height, max_iterations, filename)
frame_count += 1
print(f"Generated frame {frame_count} for c = {c}")

print("All frames generated.")

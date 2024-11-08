import os
import numpy as np
from PIL import Image
from numba import cuda
from scipy.interpolate import interp1d

# Gradient creation function with color values in the 0-255 RGB range
def make_gradient(colors, interpolation):
    """
    Creates a gradient function based on specified colors and interpolation type.
    Colors should be provided in the 0-255 RGB range.
    """
    X = [i / (len(colors) - 1) for i in range(len(colors))]
    Y = [[color[i] for color in colors] for i in range(3)]
    channels = [interp1d(X, y, kind=interpolation) for y in Y]
    return lambda x: [int(np.clip(channel(x), 0, 255)) for channel in channels]

# Define colors in the 0-255 RGB range
black = (0, 0, 0)
red = (255, 0, 0)
white = (255, 255, 255)
purple = (204, 179, 255)

# Create a gradient that goes from black to purple, red, and back to black
colors = [black, purple, red, black]
gradient = make_gradient(colors, interpolation="cubic")

# Create a palette of 256 colors
num_colors = 256
palette = [gradient(i / num_colors) for i in range(num_colors)]
palette_array = np.array(palette, dtype=np.uint8)

# Copy the palette array to the GPU
palette_array_gpu = cuda.to_device(palette_array)

# GPU Kernel for Julia set
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

# Generate a single Julia set image (used to create each tile)
def generate_julia_tile(c, scale: float, tile_size: int, max_iterations: int):
    """
    Generates a Julia set tile for a specific value of 'c' and returns it as an image.
    """
    image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)  # 3 channels for RGB
    blockdim = (16, 16)
    griddim = (tile_size // blockdim[0] + 1, tile_size // blockdim[1] + 1)
    julia_kernel[griddim, blockdim](c.real, c.imag, scale, tile_size, tile_size, max_iterations, image, palette_array_gpu)
    return Image.fromarray(image).convert("RGB")

# Function to generate the grid of c-values (center of each grid cell)
def generate_c_values(grid_size, real_range=(-2, 2), imag_range=(-2, 2)):
    """
    Generates complex values 'c' at the center of each grid cell within the given range.
    """
    real_step = (real_range[1] - real_range[0]) / grid_size
    imag_step = (imag_range[1] - imag_range[0]) / grid_size
    real_values = np.linspace(real_range[0] + real_step / 2, real_range[1] - real_step / 2, grid_size)
    imag_values = np.linspace(imag_range[1] - imag_step / 2, imag_range[0] + imag_step / 2, grid_size)  # Invert the imaginary axis
    c_values = [complex(r, i) for i in imag_values for r in real_values]
    return c_values

# Parameters
max_iterations = 6000
output_width = 2160
output_height = 2160

# Start the numbering from 0
image_counter = 0

grid_size = 31

s = 0.975
# Loop to generate frames
for frame in range(1, 300):
    # Calculate the size of each tile based on the grid size and the 2160x2160 resolution
    scale_factor = 0.99  # Scaling the grid to be 1% smaller
    tile_size = (output_width / grid_size) * scale_factor  # Apply scale factor for the grid

    # Calculate the margin to center the scaled-down grid
    x_margin = (output_width - tile_size * grid_size) / 2
    y_margin = (output_height - tile_size * grid_size) / 2

    # Calculate the appropriate scale for the Julia sets to fit exactly in each tile
    scale = 4.0 / tile_size * s  # Ensure the Julia set fits within the (-2, 2) range of the complex plane

    # Adjust the scaling factor `s` to zoom in further for the next frame
    s *= 0.975

    # Prepare the output image (2160x2160 image with all Julia sets)
    output_image = Image.new("RGB", (output_width, output_height))

    # Generate c-values for each grid point (center of each tile)
    c_values = generate_c_values(grid_size)

    # Loop through each position in the grid and generate the corresponding Julia set
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            c = c_values[idx]

            # Generate the Julia set tile for this specific c-value
            julia_tile = generate_julia_tile(c, scale=scale, tile_size=int(tile_size + 1), max_iterations=max_iterations)

            # Calculate the position with the margin applied
            x_pos = int(x_margin + j * tile_size)
            y_pos = int(y_margin + i * tile_size)

            # Paste the tile in the correct position in the output image
            output_image.paste(julia_tile, (x_pos, y_pos))

    # Create output folder if it doesn't exist
    output_folder = "julia_sets_collection_zoom"
    os.makedirs(output_folder, exist_ok=True)

    # Save the final image in the specified folder with the appropriate filename (zero-padded)
    file_name = f"{output_folder}/{str(image_counter).zfill(5)}.png"
    output_image.save(file_name)
    print(f"Image saved: {file_name}")

    # Increment the counter for the next image
    image_counter += 1

print("All Julia set images generated and saved.")

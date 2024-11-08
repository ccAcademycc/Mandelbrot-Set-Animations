# Mandelbrot-Set-Animations

The provided Python scripts (currently 9) can be used to create your own animations of the Mandelbrot set or Julia sets.
Each folder contains a Python script and a video showcasing the animation that can be generated with that script.

All scripts use the Numba package with CUDA, which requires an NVIDIA graphics card. 
This enables much faster computation of frames.

The scripts serve different purposes. Here is a brief overview:

  1. mandelbrot_increase_iterations_0.py 
  2. mandelbrot_increase_iterations_1.py
  3. mandelbrot_increase_iterations_2.py
  4. mandelbrot_increase_iterations_3.py
     
These scripts create frames of the Mandelbrot set, 
increasing the maximum number of iterations used to compute the Mandelbrot sequence by 1 with each frame.

  5. mandelbrot_zoom.py
     
This script creates a Mandelbrot zoom into the specified point, 'center'.

  6. julia_fixed_point.py
     
This script creates an image of the Julia set for a fixed complex number, 'c'.

  7. julia_change_c_animation.py
      
This script generates frames of Julia sets while changing the value of c across frames.

  8. julia_sets_collection.py
      
This script creates frames displaying a grid of Julia sets, with the grid resolution increasing progressively from frame to frame.

  9. julia_sets_collection_zoom.py
      
This script generates frames of a grid of Julia sets with a fixed grid size, while scaling each individual Julia set across frames.

Feel free to adjust the following parameters as needed:
  - Resolution
  - Colors
  - Number of colors
  - Maximum number of iterations
  - Center point
  - etc.


####
Important: 
  - Each script generates a folder containing only the corresponding frames.
  - All images are rendered in 8K by default.
The reason for rendering in 8K is that it provides (in my opinion) the best balance of quality and file size, resulting in a high-quality 4K video when downscaled.

Here is how you can achieve this: 

I.    Pick a specific file and open it in PyCharm or another IDE.

II.   Run the file to create the frames.

III.  Install FFmpeg and run the corresponding command in the terminal (depending on which frames you rendered):

-----------------------------------------------------------------------------------------------       
       1.
       2.
       3. 
       4.         
       5.
       6.
       7.
       8.
       9. 
-----------------------------------------------------------------------------------------------       


# Animations for the Mandelbrot Set and Julia Sets

The provided Python scripts (currently 9) can be used to create your own animations of the Mandelbrot set or Julia sets.
All scripts use the 'Numba' package with 'CUDA'. This enables much faster computation of frames.

The scripts serve different purposes. Here is a brief overview:

-----------------------------------------------------------------------------------------------       
  # 1. mandelbrot_increase_iterations_0.py

![mandelbrot_increase_iterations_0](https://github.com/user-attachments/assets/3cae6733-70d7-4bc7-bd52-08a50b96ab25)

  # 2. mandelbrot_increase_iterations_1.py

![mandelbrot_increase_iterations_1](https://github.com/user-attachments/assets/0ac6025d-1c43-4456-b7a2-808db344321a)
  
 # 3. mandelbrot_increase_iterations_2.py

![mandelbrot_increase_iterations_2](https://github.com/user-attachments/assets/97bd9225-0ed9-4ca1-a840-1d6f66dc98bf)

 # 4. mandelbrot_increase_iterations_3.py
 
 ![mandelbrot_increase_iterations_3](https://github.com/user-attachments/assets/0c54cfef-e7cd-49d8-8bd9-a2b7dbef2af4)

These four scripts create frames of the Mandelbrot set, where the maximum number of iterations used to compute the Mandelbrot sequence is increased by 1 with each frame.

-----------------------------------------------------------------------------------------------       
 # 5. mandelbrot_zoom.py

 ![mandelbrot_zoom](https://github.com/user-attachments/assets/91ef08c2-bea0-4036-85c6-3d0ae5e71939)
     
This script creates a Mandelbrot zoom into the specified point, called 'center'.

-----------------------------------------------------------------------------------------------       

 # 6. julia_fixed_point.py

 ![julia_fixed_point](https://github.com/user-attachments/assets/229c6ba1-0f8e-4ba6-9c15-8a430cae382a)
     
This script creates an image of the Julia set for a fixed complex number, 'c'.

-----------------------------------------------------------------------------------------------       

 # 7. julia_change_c_animation.py

 ![julia_change_c_animation](https://github.com/user-attachments/assets/62893236-7821-44c2-9a8f-675eb81601d7)
      
This script generates frames of Julia sets while changing the value of c across frames.

-----------------------------------------------------------------------------------------------       

 # 8. julia_sets_collection.py

 ![julia_sets_collection](https://github.com/user-attachments/assets/fedf2a8e-9100-4530-9a4b-a02318fdc744)
      
This script creates frames displaying a grid of Julia sets, with the grid resolution increasing progressively from frame to frame.

-----------------------------------------------------------------------------------------------       

  9. julia_sets_collection_zoom.py
      
This script generates frames of a grid of Julia sets with a fixed grid size, while scaling each individual Julia set across frames.

-----------------------------------------------------------------------------------------------       
      
### Important: 
  - Each script generates a folder containing only the corresponding frames.
  - All images are rendered in 8K by default.
It provides (in my opinion) the best balance of quality and file size when the frames are rendered in 8K and then downscaled to a 4K video.

Here is how you can achieve this:

I.    Pick a specific file and open it in PyCharm or another IDE.

II.   Run the file to create the frames.

III.  Install FFmpeg and run the corresponding command in the terminal (depending on which frames you rendered):

-----------------------------------------------------------------------------------------------       
       1.  ffmpeg -framerate 10 -i mandelbrot_increase_iterations_0/%05d.png -vf "scale=3840:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p mandelbrot_increase_iterations_0.mp4
       2.  ffmpeg -framerate 30 -i mandelbrot_increase_iterations_1/%05d.png -vf "scale=3840:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p mandelbrot_increase_iterations_1.mp4
       3.  ffmpeg -framerate 50 -i mandelbrot_increase_iterations_2/%05d.png -vf "scale=3840:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p mandelbrot_increase_iterations_2.mp4
       4.  ffmpeg -framerate 40 -i mandelbrot_increase_iterations_3/%05d.png -vf "scale=3840:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p mandelbrot_increase_iterations_3.mp4      
       5.  ffmpeg -framerate 60 -i mandelbrot_zoom/%05d.png -vf "scale=3840:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p mandelbrot_zoom.mp4
       6.  +++ Only one frame +++
       7.  ffmpeg -framerate 60 -i julia_change_c_animation/%05d.png -vf "scale=3840:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p julia_change_c_animation.mp4
       8.  ffmpeg -framerate 1 -i julia_sets_collection/%05d.png -vf "scale=2160:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p julia_sets_collection.mp4
       9.  ffmpeg -framerate 40 -i julia_sets_collection_zoom/%05d.png -vf "scale=2160:2160" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p julia_sets_collection_zoom.mp4
-----------------------------------------------------------------------------------------------       


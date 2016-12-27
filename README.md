# CUDA-Fractal-Flames
An implementation of the fractal flame algorithm in C and Nvidia CUDA.  Targeted for a Linux client.

# INTRODUCTION
This project intends to leverage the Nvida CUDA API in the generation of fractal flame images.  The transforms and procedures implemented are described in a paper available at: flam3.com/flame.pdf.  More background on fractal flames is available on Wikipedia (https://en.wikipedia.org/wiki/Fractal_flame).

# PROJECT OVERVIEW
The program produces a TIFF image of dimensions "CUDA_N x CUDA_N," where CUDA_N is a defined constant in fractal_cuda.cu, and is related to the number of threads that will run on the GPU.  Users input Cartesian grid coordinates related to the minimum and maximum x and y values.  Since the output image is a square, the range covered by the x and y axis should normally be symmetrical to avoid skewing.  The coordinates do not need to be centred around 0,0.

Users can randomly generate (or read in from a file) weighting coefficients and colors, and select which transforms should be used.

Memory is allocated on the GPU for CUDA_N x CUDA_N pixels.  Each GPU pixel data structure contains a random seed, a global hit counter, and a histogram with NUMV many elements.  The chaos game starts at a random pixel location, applies a set of randomly selected coefficients to a transform function, and will produce a new point.  At the new point, the appropriate histogram counter is incremented (each coefficient set is tied to a hit counter) because this will determine the pixel's final color.

Once a determined number of samples have been run, several steps occur to produce the final image.  A pixel's global hit total is determined, and this values of the histogram are converted to log form.  The logarithmic values are then normalized (using the log vice the linear normalize value produces the smoothest overall color, since some pixels acquire exponentially more points then their neighbors.)  Data is transferred back to the host, and a helper function writes out a TIFF format picture.

# USAGE

An example usage of this program would be:

./fractal -n 10 -c test.coeff -m -10 -M 10 -l -10 -L 10

Flags:

-n flag specifies the number of color/coefficient pairs to use
-c specifics an input file for affine transform coefficients (space separated)
-p  specifies an r g b color palette (space separated)
- S int (int specifies number of rotational axis for symmetry)
-m minimum value on the X axis
-M maximum value on the X axis
-l minimum value on the Y axis
-L maximum value on the Y axis
-f output file name (defaults to /tmp/fractal.tif)
-v use appropriate transform (if none is specified, the linear transform is used).  The number of times a transform is listed will influence its weight.  i.e. -v 0 -v 1 will be a 50/50 split.  -v 0 -v 0 -v 1 will be a 66/33 split.

Values for v include:
        0                      Linear
        1                      Sinusoidal
        2                      Spherical
        3                      Swirl
        4                      Horseshoe
        5                      Polar
        6                      Handkerchief
        7                      Heart
        8                      Disk
        9                      Spiral
        10                     Hyperbolic
        11                     Diamond
        12                     Ex
        13                     Julia
        14                     Bent
        15                     Waves
        16                     Fisheye
        17                     Popcorn
        18                     Exponential
        19                     Power
        20                     Cosine
        21                     Rings
        22                     Fan
        23                     Eyefish
        24                     Bubble
        25                     Cylinder
        26                     Tangent
        27                     Cross
        28                     Collatz

# CURRENT LIMITATIONS
Keeping a histogram of hit counters is a memory intensive practice.  Using histograms allows for easy use of atomic operations (we can increment a counter atomically.)  In the pure C version of the program, a pixel structure only hold a r, b, and b value.  Updates involve locking a pixels, computing an average of two colors, and writing back that value.  There is not a good atomic locking solution in CUDA to allow me to use this approach.

CUDA likes a flat memory model.  Multidimensional arrays and multiple calls to CUDA's malloc function are possible, but tricky and costly.  I have used a hard coded value of NUMV, which will limit the number of coefficient sets that can be used in a render without a re-compile.  Large values of NUMV require lots of GPU memory (see reasons above) and especially more of CUDA_N is a large value.

There is no good entropy source available on the GPU, so we rely on the host for random numbers.  I've used the host to seed starting values, which are then passed to a GPU side 16 bit linear shift register random number generator.  Re-seeding is necessary because otherwise cycles start of occur, and this manifests as visual artifacts.

# TODO LIST / WISH LIST

- See if speed can be gained by spawning a p_thread to populate a buffer with random numbers while the GPU code runs concurrently.
- Abuse pointer arithmetic to allow for a flexible array member to be used to allocate the histogram counters (NUMV) dynamically.  This is not a supported C feature, and may break.
- Implement a "sliding widow" to allow very large images to be generated by stitching together multiple several small grids.  This can be done manually, and it works, assuming the same random seed is used at the start of each run.
- Try only generating only half the required random seeds, and using bit shifts / masks to populate the values, since we only use 16 bits on the GPU anyway.
- Output formats other than TIFF
- Quantify number of iterations that can be run before reseeding is needed to avoid reseeding too often.
- Visual front end?

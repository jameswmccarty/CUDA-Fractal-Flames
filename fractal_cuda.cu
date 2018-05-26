#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <tiffio.h>
#include <string.h>
#include <limits.h>

//#define CUDA_N 2560
#define CUDA_N 1024
#define RANDR(lo,hi) ((lo) + (((hi)-(lo)) * drand48()))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define ERR_CUDA(x)  err = ((x)); if(err != cudaSuccess) { printf("%s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
#define MAX_INC 4294967290
#define NUMV 15

/* globals */
cudaError_t err;

typedef struct
{
  double a, b, c, d, e, f;
  double pa1, pa2, pa3, pa4;
} affine;

typedef struct
{
  FILE *pal;
  FILE *co;
  double xmin, xmax, ymin, ymax;
  int seed;
  int symmetry;
  int choices;
  int choice[];
} params;

typedef struct
{
  unsigned char r, g, b;
} color;

typedef struct
{
  //unsigned int overflow;
  union
  {
    unsigned int hits;
    double normal;
  } val;
} weight;

typedef struct
{
  unsigned int seed;
  double total_hits;
  weight color_hstgm[NUMV];
} d_pixel;

typedef struct
{
  unsigned char r, g, b;
  double color_weight[NUMV];
} h_pixel;

typedef struct
{
  int h_xres, h_yres, h_n;
  double h_xmin, h_xmax, h_ymin, h_ymax;
  unsigned int *h_chaos;
  unsigned int *d_chaos;
  params *d_params;
  params *h_params;
  affine *d_coeff;
  affine *h_coeff;
  color *pallete;
  d_pixel *d_pixels;
  d_pixel *h_dpixels;
  h_pixel *h_hpixels;
} fractal;

int
random_bit (void)
{
  return random () & 01;
}

/* from Paul Bourke */
void
contractive_mapping (affine * coeff)
{
  double a, b, d, e;

  do
    {
      do
	{
	  a = drand48 ();
	  d = RANDR (a * a, 1.);
	  if (random_bit ())
	    d = -d;
	}
      while ((a * a + d * d) > 1.);
      do
	{
	  b = drand48 ();
	  e = RANDR (b * b, 1.);
	  if (random_bit ())
	    e = -e;
	}
      while ((b * b + e * e) > 1.);
    }
  while ((a * a + b * b + d * d + e * e) >
	 (1. + (a * e - d * b) * (a * e - d * b)));

  coeff->a = a;
  coeff->b = b;
  coeff->c = RANDR (-2., 2.);
  coeff->d = d;
  coeff->e = e;
  coeff->f = RANDR (-2., 2.);
}

/* initialize the coefficient values */
void
coeff_init (affine * coeff)
{
  if (random_bit ())
    contractive_mapping (coeff);
  else
    {
      coeff->a = RANDR (-1.5, 1.5);
      coeff->b = RANDR (-1.5, 1.5);
      coeff->c = RANDR (-1.5, 1.5);
      coeff->d = RANDR (-1.5, 1.5);
      coeff->e = RANDR (-1.5, 1.5);
      coeff->f = RANDR (-1.5, 1.5);
    }
  coeff->pa4 = RANDR (-2, 2);
  coeff->pa3 = RANDR (-2, 2);
  coeff->pa2 = RANDR (-2, 2);
  coeff->pa1 = RANDR (-2, 2);

  printf ("%f, %f, %f, %f, %f, %f\n", coeff->a, coeff->b, coeff->c, coeff->d,
	  coeff->e, coeff->f);
}

void
write_to_tiff (fractal * fract)
{
  int row, col, idx;
  TIFF *output;
  char *raster;
  h_pixel *img = (*fract).h_hpixels;

  printf ("Writing to file.\n");
  /* Open the output image */
  if ((output = TIFFOpen ("/tmp/fractal.tif", "w")) == NULL)
    {
      fprintf (stderr, "Could not open outgoing image.\n");
      exit (EXIT_FAILURE);
    }

  /* malloc space for the image lines */
  raster = (char *) malloc (CUDA_N * 3 * sizeof (char));
  if (raster == NULL)
    {
      printf ("malloc() failed in write_to_tiff.\n");
      exit (EXIT_FAILURE);
    }

  /* Write the tiff tags to the file */

  TIFFSetField (output, TIFFTAG_IMAGEWIDTH, CUDA_N);
  TIFFSetField (output, TIFFTAG_IMAGELENGTH, CUDA_N);
  TIFFSetField (output, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField (output, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
  TIFFSetField (output, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField (output, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField (output, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField (output, TIFFTAG_SAMPLESPERPIXEL, 3);
  for (row = 0; row < CUDA_N; row++)
    {
      for (col = 0; col < CUDA_N; col++)
	{
	  idx = row * CUDA_N + col;
	  raster[col * 3] = img[idx].r;
	  raster[col * 3 + 1] = img[idx].g;
	  raster[col * 3 + 2] = img[idx].b;
	}
      if (TIFFWriteScanline (output, raster, row, CUDA_N * 3) != 1)
	{
	  fprintf (stderr, "Could not write image\n");
	  exit (EXIT_FAILURE);
	}
    }

  free (raster);
  /* close the file */
  TIFFClose (output);
}

/* apply color, gamma & log correction */
void
color_process (fractal * fract, double gamma)
{

  double max_count = 0.;
  double r_acc, g_acc, b_acc;
  int idx, i;

  /* find the largest hit counter (log value) */
  /* store as max_count. Move a double of value at */
  for (idx = 0; idx < CUDA_N * CUDA_N; idx++)
    {
      max_count = MAX (fract->h_dpixels[idx].total_hits, max_count);
    }

  /* sanity check */
  if (max_count == 0.)
    {
      max_count = 1.0;
    }
  printf ("Max hits was: %f\n", max_count);

  /* build normalized pixel values */
  /* determine final pixel values from palette */
  for (idx = 0; idx < CUDA_N * CUDA_N; idx++)
    {
      fract->h_dpixels[idx].total_hits /= max_count;
      r_acc = 0.;
      g_acc = 0.;
      b_acc = 0.;
      for (i = 0; i < fract->h_n; i++)
	{
	  r_acc +=
	    fract->h_hpixels[idx].color_weight[i] * fract->pallete[i].r;
	  g_acc +=
	    fract->h_hpixels[idx].color_weight[i] * fract->pallete[i].g;
	  b_acc +=
	    fract->h_hpixels[idx].color_weight[i] * fract->pallete[i].b;
	}			/* sum all sub values */
      r_acc *= fract->h_dpixels[idx].total_hits;
      g_acc *= fract->h_dpixels[idx].total_hits;
      b_acc *= fract->h_dpixels[idx].total_hits;
      /* apply a global correction factor */
      r_acc *= pow (fract->h_dpixels[idx].total_hits, (1.0 / gamma));
      g_acc *= pow (fract->h_dpixels[idx].total_hits, (1.0 / gamma));
      b_acc *= pow (fract->h_dpixels[idx].total_hits, (1.0 / gamma));
      /* complete gamma compensation per channel */
      fract->h_hpixels[idx].r = (unsigned char) r_acc;
      fract->h_hpixels[idx].g = (unsigned char) g_acc;
      fract->h_hpixels[idx].b = (unsigned char) b_acc;
      /* store results back to matrix */
    }

}				/* end of color_process */

__device__ double
modulus (double a, double b)
{
  int cast;

  cast = (int) (a / b);
  return a - ((double) cast * b);
}				/* end of modulus */

__device__ void
cuda_rnd (unsigned int *seed)
{
	const unsigned int m = 2147483648;
	const unsigned int a = 161664525;
	const unsigned int c = 12345;
	unsigned int s = *seed;
	s = (a*s+c)%m;
	*seed = s;
}

__global__ void
pre_process (d_pixel * pxls, unsigned int *rnds, int d_n)
{
  int i, j, buf_idx, co_idx;

  /* compute x (i) and y (j) index from Block and Thread */
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(i < CUDA_N && j < CUDA_N))
    return;			/* verify inbounds of image */
  buf_idx = i + j * CUDA_N;

  /* destroy everything */
  for (co_idx = 0; co_idx < d_n; co_idx++)
    {
      pxls[buf_idx].color_hstgm[co_idx].val.hits = 0;
      //pxls[buf_idx].color_hstgm[co_idx].overflow = 0;
    }
  pxls[buf_idx].total_hits = 0.;

}				/* end of pre-process */

__global__ void
resow (d_pixel * pxls, unsigned int *rnds)
{
  int i, j, buf_idx;

  /* compute x (i) and y (j) index from Block and Thread */
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(i < CUDA_N && j < CUDA_N))
    return;			/* verify inbounds of image */
  buf_idx = i + j * CUDA_N;

  /* transfer random seeds */
  pxls[buf_idx].seed = rnds[buf_idx];

}				/* end of resow */

__global__ void
post_process (d_pixel * pxls, int d_n)
{
  int i, j, buf_idx, co_idx;
  double tot_hits;
  unsigned long long int hits_acc = 0;

  /* compute x (i) and y (j) index from Block and Thread */
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(i < CUDA_N && j < CUDA_N))
    return;			/* verify inbounds of image */
  buf_idx = i + j * CUDA_N;

  for (co_idx = 0; co_idx < d_n; co_idx++)
    {
      //if(pxls[buf_idx].color_hstgm[co_idx].overflow) {
      //      hits_acc += (unsigned long long int) MAX_INC * 
      //              (unsigned long long int) pxls[buf_idx].color_hstgm[co_idx].overflow;
      //      printf("Overflow x%d for %d at %d.\n",pxls[buf_idx].color_hstgm[co_idx].overflow, buf_idx, co_idx);
      //}
      hits_acc +=
	(unsigned long long int) pxls[buf_idx].color_hstgm[co_idx].val.hits;
    }

  /* normalize the color hit counter and store log10(total) */
  if (hits_acc)
    {
      tot_hits = (double) hits_acc;
      for (co_idx = 0; co_idx < d_n; co_idx++)
	{
	  pxls[buf_idx].color_hstgm[co_idx].val.normal =
	    ((double) pxls[buf_idx].color_hstgm[co_idx].val.hits) / tot_hits;
	}
      pxls[buf_idx].total_hits = log2f (tot_hits);
    }
  else
    {
      pxls[buf_idx].total_hits = 0.;
      for (co_idx = 0; co_idx < d_n; co_idx++)
	{
	  pxls[buf_idx].color_hstgm[co_idx].val.normal = 0.;
	}
    }

}				/* end of post process */

__global__ void
render (d_pixel * pxls, affine * coarray, params * prms, int d_n,
	int iterations, double xmin, double xmax, double ymin, double ymax)
{
  int i, t, j, s, step, buf_idx;
  double newx, newy, x, y, theta, P0, P1;
  double theta2, r, xtmp, ytmp, prefix;


  /* compute x (i) and y (j) index from Block and Thread */
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(i < CUDA_N && j < CUDA_N))
    return;			/* verify inbounds of image */

  xtmp = xmax - (((double) i / (double) CUDA_N) * (xmax - xmin));
  ytmp = ymax - (((double) j / (double) CUDA_N) * (ymax - ymin));

  for (step = -35; step < iterations; step++)
    {
      cuda_rnd (&pxls->seed);
      t = pxls->seed % d_n;
      x = coarray[t].a * xtmp + coarray[t].b * ytmp + coarray[t].c;
      y = coarray[t].d * xtmp + coarray[t].e * ytmp + coarray[t].f;
      cuda_rnd (&pxls->seed);

      switch (prms->choice[pxls->seed % prms->choices])
	{
	case 0:		/* Linear */
	  xtmp = x;
	  ytmp = y;
	  break;
	case 1:		/* Sinusoidal */
	  xtmp = sin (x);
	  ytmp = sin (y);
	  break;
	case 2:		/* Spherical */
	  r = 1.0 / (x * x + y * y);
	  xtmp = r * x;
	  ytmp = r * y;
	  break;
	case 3:		/* Swirl */
	  r = x * x + y * y;
	  xtmp = x * sin (r) - y * cos (r);
	  ytmp = x * cos (r) + y * sin (r);
	  break;
	case 4:		/* Horseshoe */
	  r = 1.0 / sqrt (x * x + y * y);
	  xtmp = r * (x - y) * (x + y);
	  ytmp = r * 2.0 * x * y;
	  break;
	case 5:		/* Polar */
	  xtmp = atan2 (y, x) / M_PI;
	  ytmp = sqrt (x * x + y * y) - 1.0;
	  break;
	case 6:		/* Handkerchief */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp = r * sin (theta + r);
	  ytmp = r * cos (theta - r);
	  break;
	case 7:		/* Heart */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp = r * sin (theta * r);
	  ytmp = -r * cos (theta * r);
	  break;
	case 8:		/* Disk */
	  r = sqrt (x * x + y * y) * M_PI;
	  theta = atan2 (y, x) / M_PI;
	  xtmp = theta * sin (r);
	  ytmp = theta * cos (r);
	  break;
	case 9:		/* Spiral */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp = (1.0 / r) * (cos (theta) + sin (r));
	  ytmp = (1.0 / r) * (sin (theta) - cos (r));
	  break;
	case 10:		/* Hyperbolic */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp = sin (theta) / r;
	  ytmp = r * cos (theta);
	  break;
	case 11:		/* Diamond */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp = sin (theta) * cos (r);
	  ytmp = cos (theta) * sin (r);
	  break;
	case 12:		/* Ex */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  P0 = sin (theta + r);
	  P0 = P0 * P0 * P0;
	  P1 = cos (theta - r);
	  P1 = P1 * P1 * P1;
	  newx = r * (P0 + P1);
	  newy = r * (P0 - P1);
	  break;
	case 13:		/* Julia */
	  r = sqrt (sqrt (x * x + y * y));
	  theta = atan2 (y, x) * .5;
	  if (pxls->seed & 0x01)
	    theta += M_PI;
	  xtmp = r * cos (theta);
	  ytmp = r * sin (theta);
	  break;
	case 14:		/* Bent */
	  if (x >= 0.0 && y >= 0.0)
	    {
	      xtmp = x;
	      ytmp = y;
	    }
	  else if (x < 0.0 && y >= 0.0)
	    {
	      xtmp = 2.0 * x;
	      ytmp = y;
	    }
	  else if (x >= 0.0 && y < 0.0)
	    {
	      xtmp = x;
	      ytmp = y * .5;
	    }
	  else if (x < 0.0 && y < 0.0)
	    {
	      xtmp = 2.0 * x;
	      ytmp = y * .5;
	    }
	  break;
	case 15:		/* Waves */
	  xtmp =
	    x + coarray[t].pa1 * sin (y / (coarray[t].pa2 * coarray[t].pa2));
	  ytmp =
	    y + coarray[t].pa3 * sin (x / (coarray[t].pa4 * coarray[t].pa4));
	  break;
	case 16:		/* Fisheye */
	  r = 2.0 / (1. + sqrt (x * x + y * y));
	  xtmp = r * y;
	  ytmp = r * x;
	  break;
	case 17:		/* Popcorn */
	  xtmp = x + coarray[t].c * sin (tan (3.0 * y));
	  ytmp = y + coarray[t].f * sin (tan (3.0 * x));
	  break;
	case 18:		/* Exponential */
	  xtmp = exp (x - 1.0) * cos (M_PI * y);
	  ytmp = exp (x - 1.0) * sin (M_PI * y);
	  break;
	case 19:		/* Power */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp = pow (r, sin (theta)) * cos (theta);
	  ytmp = pow (r, sin (theta)) * sin (theta);
	  break;
	case 20:		/* Cosine */
	  xtmp = cos (M_PI * x) * cosh (y);
	  ytmp = -1. * sin (M_PI * x) * sinh (y);
	  break;
	case 21:		/* Rings */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  prefix =
	    modulus ((r + coarray[t].pa2 * coarray[t].pa2),
		     (2.0 * coarray[t].pa2 * coarray[t].pa2)) -
	    (coarray[t].pa2 * coarray[t].pa2) +
	    (r * (1.0 - coarray[t].pa2 * coarray[t].pa2));
	  xtmp = prefix * cos (theta);
	  ytmp = prefix * sin (theta);
	  break;
	case 22:		/* Fan */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  prefix = M_PI * coarray[t].c * coarray[t].c;
	  if (modulus (theta, prefix) > (prefix * .5))
	    {
	      xtmp = r * cos (theta - (prefix * .5));
	      ytmp = r * sin (theta - (prefix * .5));
	    }
	  else
	    {
	      xtmp = r * cos (theta + (prefix * .5));
	      ytmp = r * sin (theta + (prefix * .5));
	    }
	  break;
	case 23:		/* Eyefish */
	  r = 2.0 / (1. + sqrt (x * x + y * y));
	  xtmp = r * x;
	  ytmp = r * y;
	  break;
	case 24:		/* Bubble */
	  r = 4 + x * x + y * y;
	  xtmp = (4.0 * x) / r;
	  ytmp = (4.0 * y) / r;
	  break;
	case 25:		/* Cylinder */
	  xtmp = sin (x);
	  ytmp = y;
	  break;
	case 26:		/* Tangent */
	  xtmp = sin (x) / cos (y);
	  ytmp = tan (y);
	  break;
	case 27:		/* Cross */
	  r = sqrt (1.0 / ((x * x - y * y) * (x * x - y * y)));
	  xtmp = x * r;
	  ytmp = y * r;
	  break;
	case 28:		/* Collatz */
	  xtmp = .25 * (1.0 + 4.0 * x - (1.0 + 2.0 * x) * cos (M_PI * x));
	  ytmp = .25 * (1.0 + 4.0 * y - (1.0 + 2.0 * y) * cos (M_PI * y));
	  break;
	case 29:		/* Mobius */
	  prefix =
	    (coarray[t].pa3 * x + coarray[t].pa4) * (coarray[t].pa3 * x +
						     coarray[t].pa4) +
	    coarray[t].pa3 * y * coarray[t].pa3 * y;
	  xtmp =
	    ((coarray[t].pa1 * x + coarray[t].pa2) * (coarray[t].pa3 * x +
						      coarray[t].pa4) +
	     coarray[t].pa1 * coarray[t].pa3 * y * y) / prefix;
	  ytmp =
	    (coarray[t].pa1 * y * (coarray[t].pa3 * x + coarray[t].pa4) -
	     coarray[t].pa3 * y * (coarray[t].pa1 * x +
				   coarray[t].pa2)) / prefix;
	  break;
	case 30:		/* Blob */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  xtmp =
	    r * (coarray[t].pa2 +
		 0.5 * (coarray[t].pa1 -
			coarray[t].pa2) * (sin (coarray[t].pa3 * theta) +
					   1)) * cos (theta);
	  ytmp =
	    r * (coarray[t].pa2 +
		 0.5 * (coarray[t].pa1 -
			coarray[t].pa2) * (sin (coarray[t].pa3 * theta) +
					   1)) * sin (theta);
	  break;
	case 34:		/* Not Broken Waves */
	  xtmp = x + coarray[t].b * sin (y / pow (coarray[t].c, 2.0));
	  ytmp = y + coarray[t].e * sin (x / pow (coarray[t].f, 2.0));
	  break;
	case 35:		/* un-bork'd rings */
	  r = sqrt (x * x + y * y);
	  theta = atan2 (y, x);
	  prefix =
	    modulus ((r + coarray[t].c * coarray[t].c),
		     (2.0 * coarray[t].c * coarray[t].c)) -
	    coarray[t].c * coarray[t].c +
	    (r * (1 - coarray[t].c * coarray[t].c));
	  xtmp = prefix * cos (theta);
	  ytmp = prefix * sin (theta);
	  break;
	default:		/* Linear */
	  xtmp = x;
	  ytmp = y;
	  break;
	}			/* end switch */

      if (step > 0)
	{
	  theta2 = 0.0;
	  for (s = 0; s < prms->symmetry; s++)
	    {
	      theta2 += ((2 * M_PI) / (prms->symmetry));
	      newx = xtmp * cos (theta2) - ytmp * sin (theta2);
	      newy = xtmp * sin (theta2) + ytmp * cos (theta2);
	      if (newx >= xmin && newx <= xmax && newy >= ymin
		  && newy <= ymax)
		{
		  i =
		    CUDA_N - (int) (((xmax - newx) / (xmax - xmin)) * CUDA_N);
		  j =
		    CUDA_N - (int) (((ymax - newy) / (ymax - ymin)) * CUDA_N);
		  buf_idx = i + j * CUDA_N;

		  if (i >= 0 && j >= 0 && buf_idx < CUDA_N * CUDA_N
		      && buf_idx >= 0)
		    {
		      if (atomicInc
			  (&pxls[buf_idx].color_hstgm[t].val.hits,
			   MAX_INC) == MAX_INC)
			{
			  //atomicAdd(&pxls[buf_idx].color_hstgm[t].overflow, 1);
			  printf ("Block %d overflowed at %d histogram.\n",
				  buf_idx, t);
			}
		    }		/* end if CUDA bounds */
		}		/* end if bounds check */
	    }			/* end for s */
	}			/* end step if */
    }

}				/* end of render */

void
data_transfer (fractal * flame)
{

  int idx, co_idx;

  //ssize_t d_size = CUDA_N*CUDA_N*(sizeof(d_pixel) + sizeof(weight)*flame->h_n);
  ssize_t d_size = CUDA_N * CUDA_N * (sizeof (d_pixel));

  /* move results from GPU memory to HOST copy */
  ERR_CUDA (cudaMemcpy
	    (flame->h_dpixels, flame->d_pixels, d_size,
	     cudaMemcpyDeviceToHost))
    /* do a local structure to structure transfer */
    for (idx = 0; idx < CUDA_N * CUDA_N; idx++)
    {
      for (co_idx = 0; co_idx < flame->h_n; co_idx++)
	{
	  flame->h_hpixels[idx].color_weight[co_idx] =
	    flame->h_dpixels[idx].color_hstgm[co_idx].val.normal;
	}
    }
}				/* end data_transfer */

void
colors_setup (fractal * flame, FILE * filename)
{
  int idx = 0;

  for (idx = 0; idx < flame->h_n; idx++)
    {
      flame->pallete[idx].r = (unsigned char) RANDR (64, 256);
      flame->pallete[idx].g = (unsigned char) RANDR (64, 256);
      flame->pallete[idx].b = (unsigned char) RANDR (64, 256);
      printf ("%d %d %d.\n", flame->pallete[idx].r, flame->pallete[idx].g,
	      flame->pallete[idx].b);
    }

  if (filename != NULL)
    {
      int f_r, f_g, f_b;

      idx = 0;
      while ((fscanf
	      (filename, "%d %d %d\n", &f_r, &f_g, &f_b) != EOF)
	     && idx < flame->h_n)
	{
	  flame->pallete[idx].r = (unsigned char) f_r;
	  flame->pallete[idx].g = (unsigned char) f_g;
	  flame->pallete[idx].b = (unsigned char) f_b;
	  printf ("Setting index %d to %d,%d,%d.\n", idx, f_r, f_g, f_b);
	  idx++;
	}
      (void) fclose (filename);
    }

}				/* end of colors_setup */

/* This function reads from file, or randomizes coefficients
 * and copies them to the DEVICE.  Also moves the transforms
 * list.  NOTE: THIS FUNCTION CALLS CUDA FREE AND CUDA
 * MALLOC                                                    */
void
coeff_xfrm_setup (fractal * flame, FILE * filename)
{
  int idx;
  ssize_t coeff_size = flame->h_n * sizeof (affine);
  double f_a, f_b, f_c, f_d, f_e, f_f;

  /* allocate a set of coefficients for HOST */
  flame->h_coeff = (affine *) malloc (coeff_size);
  if (flame->h_coeff == NULL)
    {
      printf ("malloc() failed in c.\n");
      exit (EXIT_FAILURE);
    }
  memset (flame->h_coeff, '\0', coeff_size);

  /* initialize the values */
  for (idx = 0; idx < flame->h_n; idx++)
    {
      coeff_init (&(flame->h_coeff[idx]));
    }

  if (filename != NULL)
    {
      idx = 0;
      while ((fscanf (filename, "%lf %lf %lf %lf %lf %lf\n",
		      &f_a, &f_b, &f_c, &f_d, &f_e, &f_f) != EOF)
	     && idx < flame->h_n)
	{
	  flame->h_coeff[idx].a = f_a;
	  flame->h_coeff[idx].b = f_b;
	  flame->h_coeff[idx].c = f_c;
	  flame->h_coeff[idx].d = f_d;
	  flame->h_coeff[idx].e = f_e;
	  flame->h_coeff[idx].f = f_f;
	  printf ("Setting index coeffs at index %d\n", idx);
	  idx++;
	}
      (void) fclose (filename);
    }

  /* create a CUDA buffer for the values and ZERO */
  ERR_CUDA (cudaMalloc (&flame->d_coeff, coeff_size))
    ERR_CUDA (cudaMemset (flame->d_coeff, '\0', coeff_size))
    /* and move values to the DEVICE */
    ERR_CUDA (cudaMemcpy
	      (flame->d_coeff, flame->h_coeff, coeff_size,
	       cudaMemcpyHostToDevice))
    /* determine if we need to resize the DEVICE memory */
    /* Is true if we have > 1 Transform                 */
    if (flame->h_params->choices > 1)
    {
      ERR_CUDA (cudaFree (flame->d_params))
	ERR_CUDA (cudaMalloc (&flame->d_params,
			      sizeof (params) +
			      sizeof (int) * flame->h_params->choices));
      ERR_CUDA (cudaMemset (flame->d_params, '\0', sizeof (params)));
    }
  /* then do the transfer */
ERR_CUDA (cudaMemcpy (flame->d_params, flame->h_params, sizeof (params) + sizeof (int) * flame->h_params->choices, cudaMemcpyHostToDevice))}	/* end of coeff_setup */

void
setup_fractal (fractal * flame)
{
  //ssize_t d_size = CUDA_N*CUDA_N*(sizeof(d_pixel) + sizeof(weight)*flame->h_n);
  //ssize_t h_size = CUDA_N*CUDA_N*(sizeof(h_pixel) + sizeof(double)*flame->h_n);
  ssize_t d_size = CUDA_N * CUDA_N * (sizeof (d_pixel));
  ssize_t h_size = CUDA_N * CUDA_N * (sizeof (h_pixel));
  ssize_t r_size = CUDA_N * CUDA_N * (sizeof (unsigned int));

  /* assign a CUDA memory buffer for the fractal, and ZERO */
  ERR_CUDA (cudaMalloc (&flame->d_pixels, d_size))
    ERR_CUDA (cudaMemset (flame->d_pixels, '\0', d_size))
    /* allocate a swap buffer and working buffer for HOST */
    flame->h_dpixels = (d_pixel *) malloc (d_size);
  if (flame->h_dpixels == NULL)
    {
      printf ("malloc() failed in setup_fractal.\n");
      exit (EXIT_FAILURE);
    }
  memset (flame->h_dpixels, '\0', d_size);
  flame->h_hpixels = (h_pixel *) malloc (h_size);
  if (flame->h_dpixels == NULL)
    {
      printf ("malloc() failed in setup_fractal.\n");
      exit (EXIT_FAILURE);
    }
  memset (flame->h_hpixels, '\0', h_size);

  /* allocate a color palette for HOST */
  flame->pallete = (color *) malloc (flame->h_n * sizeof (color));
  if (flame->pallete == NULL)
    {
      printf ("malloc() failed in setup_fractal.\n");
      exit (EXIT_FAILURE);
    }
  memset (flame->pallete, '\0', flame->h_n * sizeof (color));

  /* assigned a CUDA buffer for random vars */
  ERR_CUDA (cudaMalloc (&flame->d_chaos, r_size));
  ERR_CUDA (cudaMemset (flame->d_chaos, '\0', r_size));
  /* get HOST memory for random vars */
  flame->h_chaos = (unsigned int *) malloc (r_size);
  if (flame->h_chaos == NULL)
    {
      printf ("malloc() failed in setup_fractal.\n");
      exit (EXIT_FAILURE);
    }
  memset (flame->h_chaos, '\0', r_size);

  /* setup HOST memory for transform parameters */
  flame->h_params = (params *) malloc (sizeof (params) + sizeof (int));
  if (flame->h_params == NULL)
    {
      printf ("malloc() failed in setup_fractal.\n");
      exit (EXIT_FAILURE);
    }
  flame->h_params->pal = NULL;
  flame->h_params->co = NULL;
  flame->h_params->seed = 1;
  flame->h_params->symmetry = 1;
  flame->h_params->choices = 1;
  flame->h_params->xmin = -1.;
  flame->h_params->xmax = 1.;
  flame->h_params->ymin = -1.;
  flame->h_params->ymax = 1.;
  flame->h_params->choice[0] = 0;	/* initially 1 tranform, set to 0 */
  /* and initial allocation of DEVICE transform parameters */
  ERR_CUDA (cudaMalloc (&flame->d_params, sizeof (params) + sizeof (int)));
  ERR_CUDA (cudaMemset (flame->d_params, '\0', sizeof (params)));


}				/* end setup_fractal */

void
reseed (fractal * flame)
{
  ssize_t r_size = CUDA_N * CUDA_N * (sizeof (unsigned int));
  int idx;

  memset (flame->h_chaos, '\0', r_size);

  /* sow the seeds of chaos */
  for (idx = 0; idx < CUDA_N * CUDA_N; idx++)
    {
      flame->h_chaos[idx] = (unsigned int) rand ();
    }
  /* place in buffer for later access during resow */
ERR_CUDA (cudaMemcpy (flame->d_chaos, flame->h_chaos, r_size, cudaMemcpyHostToDevice))}	/* end of reseed */

void
teardown_fractal (fractal * flame)
{
  ERR_CUDA (cudaFree (flame->d_pixels))
    ERR_CUDA (cudaFree (flame->d_coeff))
    ERR_CUDA (cudaFree (flame->d_chaos))
    ERR_CUDA (cudaFree (flame->d_params)) free (flame->h_hpixels);
  free (flame->h_dpixels);
  free (flame->pallete);
  free (flame->h_coeff);
  free (flame->h_chaos);
  free (flame->h_params);
}				/* end teardown_fractal */

void
print_usage ()
{
  /* print program use */

  printf ("fractal usage:\n");
  printf ("fractal [-options ...]\n\n");
  printf ("options include:\n");

  printf ("\n\nValues for v include:\n");
  printf ("\t0\t\t\tLinear\n");
  printf ("\t1\t\t\tSinusoidal\n");
  printf ("\t2\t\t\tSpherical\n");
  printf ("\t3\t\t\tSwirl\n");
  printf ("\t4\t\t\tHorseshoe\n");
  printf ("\t5\t\t\tPolar\n");
  printf ("\t6\t\t\tHandkerchief\n");
  printf ("\t7\t\t\tHeart\n");
  printf ("\t8\t\t\tDisk\n");
  printf ("\t9\t\t\tSpiral\n");
  printf ("\t10\t\t\tHyperbolic\n");
  printf ("\t11\t\t\tDiamond\n");
  printf ("\t12\t\t\tEx\n");
  printf ("\t13\t\t\tJulia\n");
  printf ("\t14\t\t\tBent\n");
  printf ("\t15\t\t\tWaves\n");
  printf ("\t16\t\t\tFisheye\n");
  printf ("\t17\t\t\tPopcorn\n");
  printf ("\t18\t\t\tExponential\n");
  printf ("\t19\t\t\tPower\n");
  printf ("\t20\t\t\tCosine\n");
  printf ("\t21\t\t\tRings\n");
  printf ("\t22\t\t\tFan\n");
  printf ("\t23\t\t\tEyefish\n");
  printf ("\t24\t\t\tBubble\n");
  printf ("\t25\t\t\tCylinder\n");
  printf ("\t26\t\t\tTangent\n");
  printf ("\t27\t\t\tCross\n");
  printf ("\t28\t\t\tCollatz\n");
  fflush (stdout);
}				/* end of print usage */

void
parse_args (int argc, char **argv, fractal * fractal)
{
  int i = 1;
  int override = 0;

  while (i < argc)
    {
      if (!strcmp (argv[i], "-h"))
	{
	  print_usage ();
	  exit (EXIT_SUCCESS);
	}
      else if (!strcmp (argv[i], "-R"))
	{
	  fractal->h_params->seed = atoi (argv[i + 1]);
	  i += 2;
	}
      else if (!strcmp (argv[i], "-n"))
	{
	  fractal->h_n = atoi (argv[i + 1]);
	  if (fractal->h_n > NUMV || fractal->h_n < 1)
	    {
	      printf
		("Error: %d is > built in maximum of %d.\n  You will need to recompile the program with a larger version of NUMV.\n",
		 fractal->h_n, NUMV);
	      exit (EXIT_FAILURE);
	    }
	  i += 2;
	}
      else if (!strcmp (argv[i], "-S"))
	{
	  fractal->h_params->symmetry = atoi (argv[i + 1]);
	  i += 2;
	  if (fractal->h_params->symmetry <= 0)
	    fractal->h_params->symmetry = 1;
	}
      else if (!strcmp (argv[i], "-p"))
	{
	  if (((fractal->h_params->pal = fopen (argv[i + 1], "r")) == NULL))
	    {
	      printf ("Error reading input file %s.\n", argv[i + 1]);
	      exit (EXIT_FAILURE);
	    }
	  i += 2;
	}
      else if (!strcmp (argv[i], "-c"))
	{
	  if (((fractal->h_params->co = fopen (argv[i + 1], "r")) == NULL))
	    {
	      printf ("Error reading input file %s.\n", argv[i + 1]);
	      exit (EXIT_FAILURE);
	    }
	  i += 2;
	}
      else if (!strcmp (argv[i], "-m"))
	{
	  fractal->h_params->xmin = (double) atof (argv[i + 1]);
	  i += 2;
	}
      else if (!strcmp (argv[i], "-M"))
	{
	  fractal->h_params->xmax = (double) atof (argv[i + 1]);
	  i += 2;
	}
      else if (!strcmp (argv[i], "-l"))
	{
	  fractal->h_params->ymin = (double) atof (argv[i + 1]);
	  i += 2;
	}
      else if (!strcmp (argv[i], "-L"))
	{
	  fractal->h_params->ymax = (double) atof (argv[i + 1]);
	  i += 2;
	}
      else if (!strcmp (argv[i], "-v"))
	{
	  if (!override)
	    {			/* first user supplied param overrides default */
	      fractal->h_params->choice[0] = atoi (argv[i + 1]);
	      override = 1;
	    }
	  else
	    {
	      fractal->h_params =
		(params *) realloc (fractal->h_params,
				    sizeof (params) +
				    sizeof (int) *
				    (fractal->h_params->choices + 1));
	      if (fractal->h_params == NULL)
		{
		  printf ("Error: malloc() failed in parse_args.");
		  exit (EXIT_FAILURE);
		}
	      fractal->h_params->choice[fractal->h_params->choices] =
		atoi (argv[i + 1]);
	      fractal->h_params->choices++;
	    }
	  printf ("using trasformation number %d\n", atoi (argv[i + 1]));
	  i += 2;
	}
      else
	{
	  print_usage ();
	  exit (EXIT_FAILURE);
	}
    }
}				/* end of parse args */

int
main (int argc, char **argv)
{
  fractal flame;

  flame.h_n = NUMV;

  /* setup block sizes to allow for rendering in min number of blocks */
  ERR_CUDA (cudaDeviceReset ())
  dim3 threadsPerBlock (1, 1);
  dim3 numBlocks (CUDA_N / threadsPerBlock.x, CUDA_N / threadsPerBlock.y);

  /* memory setup and initial setup */
  setup_fractal (&flame);
  parse_args (argc, argv, &flame);
  /* seed the randomizer */
  srandom (flame.h_params->seed);
  srand48 (random ());
  colors_setup (&flame, flame.h_params->pal);
  coeff_xfrm_setup (&flame, flame.h_params->co);

  /* dispatch the CUDA process */
  pre_process <<< numBlocks, threadsPerBlock >>> (flame.d_pixels,
						  flame.d_chaos, flame.h_n);
  cudaDeviceSynchronize ();
  for (int salt = 0; salt < 10; salt++)
    {
      reseed (&flame);
      resow <<< numBlocks, threadsPerBlock >>> (flame.d_pixels,
						flame.d_chaos);
      render <<< numBlocks, threadsPerBlock >>> (flame.d_pixels,
						 flame.d_coeff,
						 flame.d_params, flame.h_n,
						 20, flame.h_params->xmin,
						 flame.h_params->xmax,
						 flame.h_params->ymin,
						 flame.h_params->ymax);
    }
  cudaDeviceSynchronize ();
  post_process <<< numBlocks, threadsPerBlock >>> (flame.d_pixels, flame.h_n);

  /* copy results from GPU to HOST */
  data_transfer (&flame);

  /* perform CPU post-processing */
  color_process (&flame, 1.0);

  /* write the image to file */
  write_to_tiff (&flame);
  /* then free the DEVICE and HOST memory */
  teardown_fractal (&flame);
  return 0;
}

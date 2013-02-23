#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>

#define PI	3.14159265
#define CACHE_LINE_SIZE 64
#define FLOATS_PER_CACHE_LINE 16

void cpu_fftx(float *real_image, float *imag_image, int size_x, int size_y, float *termsYreal, float *termsYimag)
{
  // Create some space for storing temporary values
  // float *realOutBuffer = new float[size_x];
  // float *imagOutBuffer = new float[size_x];
  // // Local values
  // float *fft_real = new float[size_y];
  // float *fft_imag = new float[size_y];

#pragma omp parallel for
  for(unsigned int x = 0; x < size_x; x++)
  {
    // Create some space for storing temporary values
    float *realOutBuffer = new float[size_x];
    float *imagOutBuffer = new float[size_x];
    // Local values
    // float *fft_real = new float[size_y];
    // float *fft_imag = new float[size_y];

    for(unsigned int y = 0; y < size_y; y++)
    {
      // Compute the frequencies for this index
 //      for(unsigned int n = 0; n < size_y; n++)
 //      {
	// float term = -2 * PI * y * n / size_y;
	// fft_real[n] = cos(term);
	// fft_imag[n] = sin(term);
 //      }

      // Compute the value for this index
      realOutBuffer[y] = 0.0f;
      imagOutBuffer[y] = 0.0f;
      for(unsigned int n = 0; n < size_y; n++)
      {
        int termIndex = (n * y) % size_y;
      	realOutBuffer[y] += (real_image[x*size_x + n] * termsYreal[termIndex]) - (imag_image[x*size_x + n] * termsYimag[termIndex]);
      	imagOutBuffer[y] += (imag_image[x*size_x + n] * termsYreal[termIndex]) + (real_image[x*size_x + n] * termsYimag[termIndex]);
        // realOutBuffer[y] += (real_image[x*size_x + n] * fft_real[n]) - (imag_image[x*size_x + n] * fft_imag[n]);
        // imagOutBuffer[y] += (imag_image[x*size_x + n] * fft_real[n]) + (real_image[x*size_x + n] * fft_imag[n]);
      }
    }
    // Write the buffer back to were the original values were
    for(unsigned int y = 0; y < size_y; y++)
    {
      real_image[x*size_x + y] = realOutBuffer[y];
      imag_image[x*size_x + y] = imagOutBuffer[y];
    }
    // Reclaim some memory
    delete [] realOutBuffer;
    delete [] imagOutBuffer;
    // delete [] fft_real;
    // delete [] fft_imag;
  }
  // // Reclaim some memory
  // delete [] realOutBuffer;
  // delete [] imagOutBuffer;
  // delete [] fft_real;
  // delete [] fft_imag;
}

// This is the same as the thing above, except it has a scaling factor added to it
void cpu_ifftx(float *real_image, float *imag_image, int size_x, int size_y, float *termsYreal, float *termsYimag)
{
  // // Create some space for storing temporary values
  // float *realOutBuffer = new float[size_x];
  // float *imagOutBuffer = new float[size_x];
  // float *fft_real = new float[size_y];
  // float *fft_imag = new float[size_y];

  #pragma omp parallel for
  for(unsigned int x = 0; x < size_x; x++)
  {
    // Create some space for storing temporary values
    float *realOutBuffer = new float[size_x];
    float *imagOutBuffer = new float[size_x];
    // float *fft_real = new float[size_y];
    // float *fft_imag = new float[size_y];

    for(unsigned int y = 0; y < size_y; y++)
    {
 //      for(unsigned int n = 0; n < size_y; n++)
 //      {
 //        // Compute the frequencies for this index
	// float term = 2 * PI * y * n / size_y;
	// fft_real[n] = cos(term);
	// fft_imag[n] = sin(term);
 //      }

      // Compute the value for this index
      realOutBuffer[y] = 0.0f;
      imagOutBuffer[y] = 0.0f;
      for(unsigned int n = 0; n < size_y; n++)
      {
        int termIndex = (n * y) % size_y;
        realOutBuffer[y] += (real_image[x*size_x + n] * termsYreal[termIndex]) - (imag_image[x*size_x + n] * termsYimag[termIndex]);
        imagOutBuffer[y] += (imag_image[x*size_x + n] * termsYreal[termIndex]) + (real_image[x*size_x + n] * termsYimag[termIndex]);
      	// realOutBuffer[y] += (real_image[x*size_x + n] * fft_real[n]) - (imag_image[x*size_x + n] * fft_imag[n]);
      	// imagOutBuffer[y] += (imag_image[x*size_x + n] * fft_real[n]) + (real_image[x*size_x + n] * fft_imag[n]);
      }

      // Incoporate the scaling factor here
      realOutBuffer[y] /= size_y;
      imagOutBuffer[y] /= size_y;
    }
    // Write the buffer back to were the original values were
    for(unsigned int y = 0; y < size_y; y++)
    {
      real_image[x*size_x + y] = realOutBuffer[y];
      imag_image[x*size_x + y] = imagOutBuffer[y];
    }
    // Reclaim some memory
    delete [] realOutBuffer;
    delete [] imagOutBuffer;
    // delete [] fft_real;
    // delete [] fft_imag;
  }
  // // Reclaim some memory
  // delete [] realOutBuffer;
  // delete [] imagOutBuffer;
  // delete [] fft_real;
  // delete [] fft_imag;
}

void cpu_ffty(float *real_image, float *imag_image, int size_x, int size_y, float *termsXreal, float *termsXimag)
{
  // // Allocate some space for temporary values
  // float *realOutBuffer = new float[size_y];
  // float *imagOutBuffer = new float[size_y];
  // float *fft_real = new float[size_x];
  // float *fft_imag = new float[size_x];

  #pragma omp parallel for schedule(static, 16)
  for(unsigned int y = 0; y < size_y; y++)
  {
    // Allocate some space for temporary values
    float *realOutBuffer = new float[size_y];
    float *imagOutBuffer = new float[size_y];
    float *fft_real = new float[size_x];
    float *fft_imag = new float[size_x];

    for(unsigned int x = 0; x < size_x; x++)
    {
      // Compute the frequencies for this index
      for(unsigned int n = 0; n < size_y; n++)
      {
	float term = -2 * PI * x * n / size_x;
	fft_real[n] = cos(term);
	fft_imag[n] = sin(term);
      }

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
        // int termIndex = (n * x) % size_x;
        // realOutBuffer[x] += (real_image[n*size_x + y] * termsXreal[termIndex]) - (imag_image[n*size_x + y] * termsXimag[termIndex]);
        // imagOutBuffer[x] += (imag_image[n*size_x + y] * termsXreal[termIndex]) + (real_image[n*size_x + y] * termsXimag[termIndex]);
      	realOutBuffer[x] += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
      	imagOutBuffer[x] += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
      }
    }
    // Write the buffer back to were the original values were
    for(unsigned int x = 0; x < size_x; x++)
    {
      real_image[x*size_x + y] = realOutBuffer[x];
      imag_image[x*size_x + y] = imagOutBuffer[x];
    }
    // Reclaim some memory
    delete [] realOutBuffer;
    delete [] imagOutBuffer;
    delete [] fft_real;
    delete [] fft_imag;
  }
  // // Reclaim some memory
  // delete [] realOutBuffer;
  // delete [] imagOutBuffer;
  // delete [] fft_real;
  // delete [] fft_imag;
}

// This is the same as the thing about it, but it includes a scaling factor
void cpu_iffty(float *real_image, float *imag_image, int size_x, int size_y, float *termsXreal, float *termsXimag)
{
  // // Create some space for storing temporary values
  // float *realOutBuffer = new float[size_y];
  // float *imagOutBuffer = new float[size_y];
  // float *fft_real = new float[size_x];
  // float *fft_imag = new float[size_x];

  #pragma omp parallel for schedule(static, 16)
  for(unsigned int y = 0; y < size_y; y++)
  {
    // Create some space for storing temporary values
    float *realOutBuffer = new float[size_y];
    float *imagOutBuffer = new float[size_y];
    float *fft_real = new float[size_x];
    float *fft_imag = new float[size_x];

    for(unsigned int x = 0; x < size_x; x++)
    {
      // Compute the frequencies for this index
      for(unsigned int n = 0; n < size_y; n++)
      {
	// Note that the negative sign goes away for the term
	float term = 2 * PI * x * n / size_x;
	fft_real[n] = cos(term);
	fft_imag[n] = sin(term);
      }

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
        // int termIndex = (n * x) % size_x;
        // realOutBuffer[x] += (real_image[n*size_x + y] * termsXreal[termIndex]) - (imag_image[n*size_x + y] * termsXimag[termIndex]);
        // imagOutBuffer[x] += (imag_image[n*size_x + y] * termsXreal[termIndex]) + (real_image[n*size_x + y] * termsXimag[termIndex]);
      	realOutBuffer[x] += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
      	imagOutBuffer[x] += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
      }

      // Incorporate the scaling factor here
      realOutBuffer[x] /= size_x;
      imagOutBuffer[x] /= size_x;
    }
    // Write the buffer back to were the original values were
    for(unsigned int x = 0; x < size_x; x++)
    {
      real_image[x*size_x + y] = realOutBuffer[x];
      imag_image[x*size_x + y] = imagOutBuffer[x];
    }
    // Reclaim some memory
    delete [] realOutBuffer;
    delete [] imagOutBuffer;
    delete [] fft_real;
    delete [] fft_imag;
  }
  // // Reclaim some memory
  // delete [] realOutBuffer;
  // delete [] imagOutBuffer;
  // delete [] fft_real;
  // delete [] fft_imag;
}

void cpu_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;

  #pragma omp parallel for schedule(static, eightX)
  for(unsigned int x = 0; x < size_x; x++)
  {
  //   if (y < eightY || y >= eight7Y)
  //   {
  //     memset(real_image + y*size_x + eightX, 0, eight7X - eightX);
  //   }
  //   else
  //     memset(real_image + y*size_x, 0, size_y);
  // }
    for(unsigned int y = 0; y < size_y; y++)
    {
      if(!(x < eightX && y < eightY) &&
	 !(x < eightX && y >= eight7Y) &&
	 !(x >= eight7X && y < eightY) &&
	 !(x >= eight7X && y >= eight7Y))
      {
	// Zero out these values
	real_image[y*size_x + x] = 0;
	imag_image[y*size_x + x] = 0;
      }
    }
  }
}

float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{
  // These are used for timing
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  // Start timing
  gettimeofday(&tv1,&tz1);

  float *termsXreal = new float[size_x];
  float *termsXimag = new float[size_x];
  float *termsYreal = new float[size_y];
  float *termsYimag = new float[size_y];


  for(unsigned int n = 0; n < size_x; n++)
  {
    float term = -2 * PI * n / size_x;
    termsXreal[n] = cos(term);
    termsXimag[n] = sin(term);
  }

  for(unsigned int n = 0; n < size_y; n++)
  {
    float term = -2 * PI * n / size_y;
    termsYreal[n] = cos(term);
    termsYimag[n] = sin(term);
  }


  // Perform fft with respect to the x direction
  cpu_fftx(real_image, imag_image, size_x, size_y, termsYreal, termsYimag);

  // End timing
  gettimeofday(&tv2,&tz2);
  // Compute the time difference in micro-seconds
  float execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel FFTX Execution Time: %f ms\n\n", execution);



  // Perform fft with respect to the y direction
  cpu_ffty(real_image, imag_image, size_x, size_y, termsXreal, termsXimag);

  // End timing
  gettimeofday(&tv2,&tz2);
  // Compute the time difference in micro-seconds
  execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel FFTY Execution Time: %f ms\n\n", execution);


  // Filter the transformed image
  cpu_filter(real_image, imag_image, size_x, size_y);

  // End timing
  gettimeofday(&tv2,&tz2);
  // Compute the time difference in micro-seconds
  execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel Filter Execution Time: %f ms\n\n", execution);


  // Perform an inverse fft with respect to the x direction
  cpu_ifftx(real_image, imag_image, size_x, size_y, termsYreal, termsYimag);

  // End timing
  gettimeofday(&tv2,&tz2);
  // Compute the time difference in micro-seconds
  execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel IFFTX Execution Time: %f ms\n\n", execution);

  // Perform an inverse fft with respect to the y direction
  cpu_iffty(real_image, imag_image, size_x, size_y, termsXreal, termsXimag);

  // End timing
  gettimeofday(&tv2,&tz2);
  // Compute the time difference in micro-seconds
  execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel IFFTY Execution Time: %f ms\n\n", execution);

  // End timing
  gettimeofday(&tv2,&tz2);
  // Compute the time difference in micro-seconds
  execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel Execution Time: %f ms\n\n", execution);


  // for(unsigned int x = 0; x < size_x/2; x++)
  // {
  //   for(unsigned int y = 0; y < size_y/3; y++)
  //   {
  //     // real_image[x*size_y + y] = 0;
  //     // imag_image[x*size_y + y] = 0;

  //     real_image[y*size_x + x] = 0;
  //     imag_image[y*size_x + x] = 0;
  //   }
  // }

  // for(unsigned int n = 0; n < 100 * size_x; n ++)
  // {
  //   real_image[n] = 0;
  //   imag_image[n] = 0;
  // }



  return execution;
}

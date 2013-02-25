#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>

#define PI	3.14159265
#define CACHE_LINE_SIZE 64
#define FLOATS_PER_CACHE_LINE 16

void build_bit_rev_index(short *arr, int size)
{
  arr[0] = 0;
  arr[1] = size >> 1;
  if(size < 8)
    return;
  arr[2] = size >> 2;
  arr[3] = arr[1] + arr[2];
  for (int k = 3; (1 << k) < size; ++k)
  {
    int nk = (1 << k) - 1;
    int nkminus1 = (1 << (k -1)) - 1;
    arr[nk] = arr[nkminus1] + (size >> k);
    for (int l = 1; l < nkminus1 + 1; ++l)
    {
      arr[nk-l] = arr[nk] - arr[l];
    }
  }
}

void bit_reverse(float *values, short *rev, int size)
{
  int l = size / 2;
  float temp;
  for (int i = 0; i < l; ++i)
  {
    short index = rev[i];
    if (i < index)
    {
      temp = values[i];
      values[i] = values[index];
      values[index] = temp;
    }
    if (i + l < index + 1)
    {
      //rev covers only even indices. Odd pairs are the same but translated by size/2.
      temp = values[i + l];
      values[i + l] = values[index+1];
      values[index+1] = temp;
    }
  }
}

void butterfly_forward_dit(float *real, float *imag, int ind1, int ind2, float real_twiddle, float imag_twiddle)
{
  float r1 = real[ind1];
  float r2 = real[ind2];
  real[ind1] = r1 + real_twiddle * r2 - imag_twiddle * imag[ind2];
  real[ind2] = r1 - (real_twiddle * r2 - imag_twiddle * imag[ind2]);

  float i1 = imag[ind1];
  imag[ind1] = i1 + imag_twiddle * r2 + imag[ind2] * real_twiddle;
  imag[ind2] = i1 - (imag_twiddle * r2 + imag[ind2] * real_twiddle);
}
void butterfly_forward_dif(float *real, float *imag, int ind1, int ind2, float real_twiddle, float imag_twiddle)
{
  float r1 = real[ind1];
  float r2 = real[ind2];
  real[ind1] = r1 + r2;
  real[ind2] = real_twiddle * (r1 - r2) - imag_twiddle * (imag[ind1] - imag[ind2]);

  float i1 = imag[ind1];
  imag[ind1] = i1 + imag[ind2];
  imag[ind2] = imag_twiddle * (r1 - r2) + real_twiddle * (i1 - imag[ind2]);
}
void fourier_dit(float *real, float *imag, int size, short *rev, bool invert)
{
  bit_reverse(real, rev, size);
  bit_reverse(imag, rev, size);

  for (int span = 1; span < size; span <<= 1)
  {
    int num_units = size / (2 * span);
    for (int unit = 0; unit < num_units; ++unit)
    {
      int two_unit_span = 2 * unit * span;
      for (int i = 0; i < span; ++i)
      {
        int twiddle_index = i * num_units;
        float angle = 2.0*PI*twiddle_index/ size;
        if (invert)
        {
          angle = -angle;
        }
        float real_twiddle = cos(angle);
        float imag_twiddle = sin(-angle);
        // if (print)
        // {
        //   printf("%d, %d, %d, %f, %f\n", twiddle_index, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
        // }
        butterfly_forward_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
      }
    }
  }
  for (int i = 0; i < size; ++i)
  {
    real[i] /= size;
    imag[i] /= size;
  }
}

void fourier_dif(float *real, float *imag, int size, short *rev, bool invert)
{
  for (int span = size >> 1; span; span >>= 1)
  {
    int num_units = size / (2 * span);
    for (int unit = 0; unit < num_units; ++unit)
    {
      int two_unit_span = 2 * unit * span;
      for (int i = 0; i < span; ++i)
      {
        int twiddle_index = i * num_units;
        float angle = 2.0*PI*twiddle_index/ size;
        if (invert)
        {
          angle = -angle;
        }
        float real_twiddle = cos(angle);
        float imag_twiddle = sin(-angle);
        // if (print)
        // {
        //   printf("%d, %d, %d, %f, %f\n", twiddle_index, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
        // }
        butterfly_forward_dif(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
      }
    }
  }

  for (int i = 0; i < size; ++i)
  {
    real[i] /= size;
    imag[i] /= size;
  }

  bit_reverse(real, rev, size);
  bit_reverse(imag, rev, size);
}

void fft_row(float *real, float *imag, int size, short *rev)
{
  // printf("Real 1st row before fft:\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", real[i]);
  // }
  // printf("\n");
  // printf("Imag 1st row before fft:\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", imag[i]);
  // }
  // printf("\n");
  #pragma parallel for
  for (int row = 0; row < size; ++row)
  {
    fourier_dit(real + row*size, imag + row*size, size, rev, 0);
  }
  // printf("\n");
  // printf("Real 1st row after fft:\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", real[i]);
  // }
  // printf("\n");
  // printf("Imag 1st row after fft:\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", imag[i]);
  // }
  // printf("\n");
}

void fft_col(float *real, float *imag, int size, short *rev)
{
  #pragma parallel for
  for (int col = 0; col < size; ++col)
  {
    float *real_col = new float[size];
    float *imag_col = new float[size];
    for(unsigned int row = 0; row < size; row++)
    {
      real_col[row] = real[row*size + col];
      imag_col[row] = imag[row*size + col];
    }

    fourier_dit(real_col, imag_col, size, rev, 0);

    for(unsigned int row = 0; row < size; row++)
    {
      real[row*size + col] = real_col[row];
      imag[row*size + col] = imag_col[row];
    }
  }
}

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
        realOutBuffer[y] += (real_image[x*size_x + n] * termsYreal[termIndex]) - (imag_image[x*size_x + n] * -termsYimag[termIndex]);
        imagOutBuffer[y] += (imag_image[x*size_x + n] * termsYreal[termIndex]) + (real_image[x*size_x + n] * -termsYimag[termIndex]);
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

    float *realInBuffer = new float[size_y];
    float *imagInBuffer = new float[size_y];
    for(unsigned int x = 0; x < size_x; x++)
    {
      realInBuffer[x] = real_image[x*size_x + y];
      imagInBuffer[x] = imag_image[x*size_x + y];
    }

    // float *fft_real = new float[size_x];
    // float *fft_imag = new float[size_x];

    for(unsigned int x = 0; x < size_x; x++)
    {
      // Compute the frequencies for this index
 //      for(unsigned int n = 0; n < size_y; n++)
 //      {
	// float term = -2 * PI * x * n / size_x;
	// fft_real[n] = cos(term);
	// fft_imag[n] = sin(term);
 //      }

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
        int termIndex = (n * x) % size_x;
        // printf("real term: %f , computed: %f, imag: %f , %f\n", termsXreal[termIndex], fft_real[n], termsXimag[termIndex], fft_imag[n]);
        // realOutBuffer[x] += (real_image[n*size_x + y] * termsXreal[termIndex]) - (imag_image[n*size_x + y] * termsXimag[termIndex]);
        // imagOutBuffer[x] += (imag_image[n*size_x + y] * termsXreal[termIndex]) + (real_image[n*size_x + y] * termsXimag[termIndex]);

        realOutBuffer[x] += (realInBuffer[n] * termsXreal[termIndex]) - (imagInBuffer[n] * termsXimag[termIndex]);
        imagOutBuffer[x] += (imagInBuffer[n] * termsXreal[termIndex]) + (realInBuffer[n] * termsXimag[termIndex]);
      	// realOutBuffer[x] += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
      	// imagOutBuffer[x] += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
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

    delete [] realInBuffer;
    delete [] imagInBuffer;

    // delete [] fft_real;
    // delete [] fft_imag;
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

    float *realInBuffer = new float[size_y];
    float *imagInBuffer = new float[size_y];
    for(unsigned int x = 0; x < size_x; x++)
    {
      realInBuffer[x] = real_image[x*size_x + y];
      imagInBuffer[x] = imag_image[x*size_x + y];
    }
    // float *fft_real = new float[size_x];
    // float *fft_imag = new float[size_x];

    for(unsigned int x = 0; x < size_x; x++)
    {
      // Compute the frequencies for this index
 //      for(unsigned int n = 0; n < size_y; n++)
 //      {
	// // Note that the negative sign goes away for the term
	// float term = 2 * PI * x * n / size_x;
	// fft_real[n] = cos(term);
	// fft_imag[n] = sin(term);
 //      }

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
        int termIndex = ( n * x) % size_x;
        // realOutBuffer[x] += (real_image[n*size_x + y] * termsXreal[termIndex]) - (imag_image[n*size_x + y] * -termsXimag[termIndex]);
        // imagOutBuffer[x] += (imag_image[n*size_x + y] * termsXreal[termIndex]) + (real_image[n*size_x + y] * -termsXimag[termIndex]);

        realOutBuffer[x] += (realInBuffer[n] * termsXreal[termIndex]) - (imagInBuffer[n] * -termsXimag[termIndex]);
        imagOutBuffer[x] += (imagInBuffer[n] * termsXreal[termIndex]) + (realInBuffer[n] * -termsXimag[termIndex]);

        // realOutBuffer[x] += (real_image[y*size_x + n] * termsXreal[termIndex]) - (imag_image[y*size_x + n] * -termsXimag[termIndex]);
        // imagOutBuffer[x] += (imag_image[y*size_x + n] * termsXreal[termIndex]) + (real_image[y*size_x + n] * -termsXimag[termIndex]);
      	// realOutBuffer[x] += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
      	// imagOutBuffer[x] += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
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

      // real_image[y*size_x + x] = realOutBuffer[x];
      // imag_image[y*size_x + x] = imagOutBuffer[x];
    }
    // Reclaim some memory
    delete [] realOutBuffer;
    delete [] imagOutBuffer;

    delete [] realInBuffer;
    delete [] imagInBuffer;

    // delete [] fft_real;
    // delete [] fft_imag;
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
    if (x < eightX || x >= eight7X)
    {
      memset(real_image + x*size_x + eightY, 0, (eight7Y - eightY) * sizeof(float));
      memset(imag_image + x*size_x + eightY, 0, (eight7Y - eightY) * sizeof(float));
    }
    else
    {
      memset(real_image + x*size_x, 0, size_x * sizeof(float));
      memset(imag_image + x*size_x, 0, size_x * sizeof(float));
    }
  }
 //    for(unsigned int y = 0; y < size_y; y++)
 //    {
 //      if(!(x < eightX && y < eightY) &&
	//  !(x < eightX && y >= eight7Y) &&
	//  !(x >= eight7X && y < eightY) &&
	//  !(x >= eight7X && y >= eight7Y))
 //      {
	// // Zero out these values
	// real_image[y*size_x + x] = 0;
	// imag_image[y*size_x + x] = 0;
 //      }
 //    }
 //  }
}

float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{

  // int s = 32;
  // float *real = new float[s];
  // float *imag = new float[s];
  // short *r = new short[s/2];
  // build_bit_rev_index(r, s);

  // for (int i = 0; i < s; ++i)
  // {
  //   real[i] = i;
  //   imag[i] = 0;
  // }
  // forward_fourier_dit(real, imag, s, r, 1);
  // printf("\n");
  // for (int i = 0; i < s; ++i)
  // {
  //   printf("%f + %fi\n", real[i], imag[i]);
  //   imag[i] = 0;
  // }
  // printf("\n");

  // for (int i = 0; i < s; ++i)
  // {
  //   real[i] = i;
  //   imag[i] = 0;
  // }
  // forward_fourier_dif(real, imag, s, r, 1);
  // printf("\n");
  // for (int i = 0; i < s; ++i)
  // {
  //   printf("%f + %fi\n", real[i], imag[i]);
  //   imag[i] = 0;
  // }
  // printf("\n");





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

  int size = size_x;
  // printf("Size is: %d\n", size);
  short *rev = new short[size/2];
  build_bit_rev_index(rev, size);
  printf("Rev array is:\n");
  for (int i = 0; i < size / 2; ++i)
  {
    printf("%d, ", rev[i]);
  }
  printf("\n\n");

  // Perform fft with respect to the x direction
  // cpu_fftx(real_image, imag_image, size_x, size_y, termsYreal, termsYimag);
  fft_row(real_image, imag_image, size, rev);

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
  // cpu_ffty(real_image, imag_image, size_x, size_y, termsXreal, termsXimag);
  fft_col(real_image, imag_image, size, rev);

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

  // cpu_filter(real_image, imag_image, size_x, size_y);



  return execution;
}

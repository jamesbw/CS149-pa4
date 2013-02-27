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

void swap_submatrices(float *submatrix1, float *submatrix2, int sub_size, int mat_size)
{
  int size_bytes = sizeof(float) * sub_size;
  float *temp = new float[sub_size];
  for (int row = 0; row < sub_size; ++row)
  {
    int row_offset = row * mat_size;
    memcpy(temp, submatrix1 + row_offset, size_bytes);
    memcpy(submatrix1 + row_offset, submatrix2 + row_offset, size_bytes);
    memcpy(submatrix2 + row_offset, temp, size_bytes);
  }
}

void transpose_submatrix(float *top_left, int sub_size, int mat_size)
{
  // float *top_left = matrix + mat_size * top + left;
  if (sub_size > FLOATS_PER_CACHE_LINE * 2)
  {
    //transpose sub-matrices
    int sub_size_half = sub_size >> 1;
    transpose_submatrix(top_left, sub_size_half, mat_size);
    transpose_submatrix(top_left + sub_size_half, sub_size_half, mat_size);
    transpose_submatrix(top_left + sub_size_half * mat_size, sub_size_half, mat_size);
    transpose_submatrix(top_left + sub_size_half * (mat_size + 1), sub_size_half, mat_size);
  
    //swap
    float *top_right_submatrix = top_left + sub_size_half;
    float *bottom_left_submatrix = top_left + mat_size * sub_size_half;
    swap_submatrices(top_right_submatrix, bottom_left_submatrix, sub_size_half, mat_size);
  }
  else
  {
    //transpose manually
    float *temp = new float[sub_size * sub_size];
    for (int row = 0; row < sub_size; ++row)
    {
      for (int col = 0; col < sub_size; ++col)
      {
        temp[col*sub_size + row] = top_left[col + row * mat_size];
      }
    }
    for (int row = 0; row < sub_size; ++row)
    {
      memcpy(top_left + row * mat_size, temp + row * sub_size, sub_size * sizeof(float));
    }
    delete[] temp;
  }
}

void transpose(float *matrix, int size)
{
  transpose_submatrix(matrix, size, size);
}

void transpose_parallel(float *real, float *imag, int size)
{
  int half_size = size >> 1;
  #pragma omp sections
      {
      #pragma omp section
        {
          transpose_submatrix(real, half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(real + half_size, half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(real + size * half_size, half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(real + half_size * (size + 1), half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(imag, half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(imag + half_size, half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(imag + size * half_size, half_size, size);
        }

      #pragma omp section
        {
          transpose_submatrix(imag + half_size * (size + 1), half_size, size);
        }

      }  /* end of sections */

  int quarter_size = half_size >> 1;
  int three_quarter_size = half_size + quarter_size;

  #pragma omp sections
      {
      #pragma omp section
        {
          swap_submatrices(real + half_size, real + half_size * size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(real + three_quarter_size, real + half_size * size + quarter_size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(real + half_size + quarter_size * size, real + three_quarter_size * size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(real + three_quarter_size + quarter_size * size, real + quarter_size + three_quarter_size * size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(imag + half_size, imag + half_size * size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(imag + three_quarter_size, imag + half_size * size + quarter_size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(imag + half_size + quarter_size * size, imag + three_quarter_size * size, quarter_size, size);
        }

      #pragma omp section
        {
          swap_submatrices(imag + three_quarter_size + quarter_size * size, imag + quarter_size + three_quarter_size * size, quarter_size, size);
        }

      }  /* end of sections */
}


void butterfly_forward_dit(float *real, float *imag, int ind1, int ind2, float real_twiddle, float imag_twiddle)
{
  float r1 = real[ind1];
  float r2 = real[ind2];
  float temp = real_twiddle * r2 - imag_twiddle * imag[ind2];
  real[ind1] = r1 + temp;
  real[ind2] = r1 - temp;

  float i1 = imag[ind1];
  temp = imag_twiddle * r2 + imag[ind2] * real_twiddle;
  imag[ind1] = i1 + temp;
  imag[ind2] = i1 - temp;
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

void butterfly_trivial_zero(float *real, float *imag, int ind1, int ind2)
{
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  real[ind1] = r1 + r2;
  real[ind2] = r1 - r2;
  imag[ind1] = i1 + i2;
  imag[ind2] = i1 - i2;
}

void butterfly_trivial_minus_j(float *real, float *imag, int ind1, int ind2, bool invert)
{
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  if (invert)
  {
    real[ind1] = r1 - i2;
    real[ind2] = r1 + i2;
    imag[ind1] = i1 + r2;
    imag[ind2] = i1 - r2;
  }
  else
  {
    real[ind1] = r1 + i2;
    real[ind2] = r1 - i2;
    imag[ind1] = i1 - r2;
    imag[ind2] = i1 + r2;
  }
}

void fourier_dit(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_imag)
{
  bit_reverse(real, rev, size);
  bit_reverse(imag, rev, size);

  //span = 1; num_units = size / 2; 
  for (int unit = 0, two_unit_span = 0; unit < (size >> 1); ++unit, two_unit_span += 2)
  {
    butterfly_trivial_zero(real, imag, two_unit_span, two_unit_span+ 1);
  }

  //span = 2: num_units = size / 4;
  for (int unit = 0, two_unit_span = 0; unit < (size >> 2); ++unit, two_unit_span += 4)
  {
    butterfly_trivial_zero(real, imag, two_unit_span,  two_unit_span + 2);
    butterfly_trivial_minus_j(real, imag, 1 + two_unit_span, two_unit_span + 3, invert);
  }


  for (int span = 4, num_units = (size >> 3); span < size; span <<= 1, num_units >>= 1)
  {
    for (int unit = 0, two_unit_span = 0; unit < num_units; ++unit, two_unit_span += (span << 1))
    {
      int half_span = span >> 1;
      //i = 0
      butterfly_trivial_zero(real, imag, two_unit_span, two_unit_span + span);
      for (int i = 1, twiddle_index = num_units; i < half_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float imag_twiddle = invert? -roots_imag[twiddle_index] : roots_imag[twiddle_index];
        butterfly_forward_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
      }
      butterfly_trivial_minus_j(real, imag, half_span + two_unit_span, half_span + two_unit_span + span, invert);
      for (int i = half_span + 1, twiddle_index = (half_span + 1) * num_units; i < span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float imag_twiddle = invert? -roots_imag[twiddle_index] : roots_imag[twiddle_index];
        butterfly_forward_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
      }
    }
  }
  if (invert)
  {
    for (int i = 0; i < size; ++i)
    {
      real[i] /= size;
      imag[i] /= size;
    }
  }
}

void fourier_dif(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_imag)
{
  for (int span = size >> 1, num_units = 1; span; span >>= 1, num_units <<= 1)
  {
    for (int unit = 0, two_unit_span = 0; unit < num_units; ++unit, two_unit_span += (span << 1))
    {
      for (int i = 0, twiddle_index = 0; i < span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float imag_twiddle = roots_imag[twiddle_index];
        if (invert)
        {
          imag_twiddle = -imag_twiddle;
        }
        butterfly_forward_dif(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, imag_twiddle);
      }
    }
  }

  if (invert)
  {
    for (int i = 0; i < size; ++i)
    {
      real[i] /= size;
      imag[i] /= size;
    }
  }

  bit_reverse(real, rev, size);
  bit_reverse(imag, rev, size);
}

void fft_row(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_imag)
{
  #pragma omp for
  for (int row = 0; row < size; ++row)
  {
    fourier_dit(real + row*size, imag + row*size, size, rev, invert, roots_real, roots_imag);
  }
}

void fft_col(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_imag)
{
  #pragma omp for
  for (int col = 0; col < size; ++col)
  {
    float *real_col = new float[size];
    float *imag_col = new float[size];
    for(unsigned int row = 0; row < size; row++)
    {
      real_col[row] = real[row*size + col];
      imag_col[row] = imag[row*size + col];
    }

    fourier_dit(real_col, imag_col, size, rev, invert, roots_real, roots_imag);

    for(unsigned int row = 0; row < size; row++)
    {
      real[row*size + col] = real_col[row];
      imag[row*size + col] = imag_col[row];
    }
  }
}

void cpu_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;

  #pragma omp for schedule(static, eightX)
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
}

float stats(char *msg,struct timeval *tv1, struct timezone *tz1, struct timeval *tv2, struct timezone *tz2)
{
  gettimeofday(tv2,tz2);
  float execution = ((tv2->tv_sec-tv1->tv_sec)*1000000+(tv2->tv_usec-tv1->tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel %s Execution Time: %f ms\n\n", msg,  execution);
  return execution;
}

float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{

  // These are used for timing
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;
  float execution;

  // Start timing
  gettimeofday(&tv1,&tz1);

  

  int size = size_x;
  int half_size = size >> 1;
  float *roots_real = new float[half_size];
  float *roots_imag = new float[half_size];
  float two_pi_over_size = - 2 * PI / size;
  for (int i = 0; i < half_size; ++i)
  {
    float term = i * two_pi_over_size;
    roots_real[i] = cos(term);
    roots_imag[i] = sin(term);
  }
  short *rev = new short[half_size];
  build_bit_rev_index(rev, size);

  execution = stats("Setup", &tv1, &tz1, &tv2, &tz2);

  #pragma omp parallel
  {
    


    // Perform fft with respect to the x direction
    fft_row(real_image, imag_image, size, rev, false, roots_real, roots_imag);

    #pragma omp single
    { 
      execution = stats("FFTX", &tv1, &tz1, &tv2, &tz2);
    }

    transpose_parallel(real_image, imag_image, size);

    #pragma omp single
    { 
      execution = stats("Transpose", &tv1, &tz1, &tv2, &tz2);
    }

    // Perform fft with respect to the y direction
    fft_row(real_image, imag_image, size, rev, false, roots_real, roots_imag);

    #pragma omp single
    { 
      execution = stats("FFTY", &tv1, &tz1, &tv2, &tz2);
    }

    // Filter the transformed image
    cpu_filter(real_image, imag_image, size_x, size_y);

    #pragma omp single
    { 
      execution = stats("Filter", &tv1, &tz1, &tv2, &tz2);
    }

    // Perform an inverse fft with respect to the x direction
    fft_row(real_image, imag_image, size, rev, true, roots_real, roots_imag);

    #pragma omp single
    { 
      execution = stats("IFFTY", &tv1, &tz1, &tv2, &tz2);
    }

    transpose_parallel(real_image, imag_image, size);

    #pragma omp single
    { 
      execution = stats("Transpose", &tv1, &tz1, &tv2, &tz2);
    }


    // Perform an inverse fft with respect to the y direction
    fft_row(real_image, imag_image, size, rev, true, roots_real, roots_imag);

    #pragma omp single
    { 
      execution = stats("IFFTX", &tv1, &tz1, &tv2, &tz2);
    }

  }


  delete[] rev;
  delete[] roots_real;
  delete[] roots_imag;

  return execution;
}

#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>

#define PI	3.14159265
#define CACHE_LINE_SIZE 64
#define FLOATS_PER_CACHE_LINE 16
#define ONE_OVER_SQRT_2 0.70710678
#define SQRT_2 1.41421356

//from http://www.math.fsu.edu/~gallivan/courses/NLA1/bitreversal.pdf
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

void bit_reversed_filter(float *real_values, float *imag_values, short *rev, int size)
{
  int l = size >> 1;
  int eighth = size >> 3;
  int seven_eighths = size - eighth;
  for (int i = 0; i < l; ++i)
  {
    short index = rev[i];
    if (index >= eighth && index < seven_eighths)
    {
      real_values[i] = 0.f;
      imag_values[i] = 0.f;
    }

    short index_plus_one = index + 1;
    if (index_plus_one >= eighth && index_plus_one < seven_eighths)
    {
      real_values[i + l] = 0.f;
      imag_values[i + l] = 0.f;
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


void butterfly_dif(float *real, float *imag, int ind1, int ind2, float real_twiddle, float real_plus_imag_twiddle, float real_minus_imag_twiddle)
{
  // x1 = x1 + x2
  // x2 = twiddle * (x1 - x2)

  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  float real_diff = r1 - r2;
  float imag_diff = i1 - i2;
  float z = real_twiddle * (real_diff - imag_diff);

  real[ind1] = r1 + r2;
  real[ind2] = real_minus_imag_twiddle * imag_diff + z;

  imag[ind1] = i1 + i2;
  imag[ind2] = real_plus_imag_twiddle * real_diff - z;
}

void butterfly_trivial_zero_dif(float *real, float *imag, int ind1, int ind2)
{
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  real[ind1] = r1 + r2;
  real[ind2] = r1 - r2;
  imag[ind1] = i1 + i2;
  imag[ind2] = i1 - i2;
}

void butterfly_trivial_minus_j_dif(float *real, float *imag, int ind1, int ind2, bool invert)
{
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  real[ind1] = r1 + r2;
  imag[ind1] = i1 + i2;
  if (invert)
  {
    real[ind2] = i2 - i1;
    imag[ind2] = r1 - r2;
  }
  else
  {
    real[ind2] = i1 - i2;
    imag[ind2] = r2 - r1;
  }
}

void butterfly_trivial_one_minus_j_dif(float *real, float *imag, int ind1, int ind2, bool invert)
{
  //twiddle = (1-j)/sqrt(2)

  // x1 = x1 + x2
  // x2 = twiddle * (x1 - x2)
  
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  float real_diff = r1 - r2;
  float imag_diff = i1 - i2;
  float z = ONE_OVER_SQRT_2 * (real_diff - imag_diff);

  real[ind1] = r1 + r2;
  imag[ind1] = i1 + i2;

  if (invert)
  {
    real[ind2] = z;
    imag[ind2] = SQRT_2 * real_diff - z;
  }
  else
  {
    real[ind2] = SQRT_2 * imag_diff + z;
    imag[ind2] = - z;
  }

}

void butterfly_trivial_minus_one_minus_j_dif(float *real, float *imag, int ind1, int ind2, bool invert)
{
  //twiddle = -(1+j)/sqrt(2)

  // x1 = x1 + x2
  // x2 = twiddle * (x1 - x2)
  
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  float real_diff = r1 - r2;
  float imag_diff = i1 - i2;
  float z = ONE_OVER_SQRT_2 * (imag_diff - real_diff);

  real[ind1] = r1 + r2;
  imag[ind1] = i1 + i2;

  if (invert)
  {
    real[ind2] = z - SQRT_2 * imag_diff;
    imag[ind2] = - z;
  }
  else
  {
    real[ind2] =  z;
    imag[ind2] = - SQRT_2 * real_diff - z;
  }
}

void butterfly_dit(float *real, float *imag, int ind1, int ind2, float real_twiddle, float real_plus_imag_twiddle, float real_minus_imag_twiddle)
{
  // x1 = x1 + twiddle * x2
  // x2 = x1 - twiddle * x2

  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  float z = real_twiddle * (r2 - i2);
  float temp = real_minus_imag_twiddle * i2 + z;
  real[ind1] = r1 + temp;
  real[ind2] = r1 - temp;

  temp = real_plus_imag_twiddle * r2 - z;
  imag[ind1] = i1 + temp;
  imag[ind2] = i1 - temp;
}

void butterfly_trivial_zero_dit(float *real, float *imag, int ind1, int ind2)
{
  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  real[ind1] = r1 + r2;
  real[ind2] = r1 - r2;
  imag[ind1] = i1 + i2;
  imag[ind2] = i1 - i2;
}

void butterfly_trivial_minus_j_dit(float *real, float *imag, int ind1, int ind2, bool invert)
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

// (C+iS)(X+iY)=CX−SY+i(CY+SX)
// Z=C(X−Y)
// D=C+S
// E=C−S
// CX−SY=EY+Z
// CY+SX=DX−Z

void butterfly_trivial_one_minus_j_dit(float *real, float *imag, int ind1, int ind2, bool invert)
{
  //twiddle = (1-j)/sqrt(2)

  // x1 = x1 + twiddle * x2
  // x2 = x1 - twiddle * x2

  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  float z = ONE_OVER_SQRT_2 * (r2 - i2);
  float temp;

  if (invert)
  {
    real[ind1] = r1 + z;
    real[ind2] = r1 - z;

    temp = SQRT_2 * r2 - z;
    imag[ind1] = i1 + temp;
    imag[ind2] = i1 - temp;
  }
  else
  {
    temp = SQRT_2 * i2 + z;
    real[ind1] = r1 + temp;
    real[ind2] = r1 - temp;
    imag[ind1] = i1 - z;
    imag[ind2] = i1 + z;
  }
}

void butterfly_trivial_minus_one_minus_j_dit(float *real, float *imag, int ind1, int ind2, bool invert)
{
  //twiddle = -(1+j)/sqrt(2)

  // x1 = x1 + twiddle * x2
  // x2 = x1 - twiddle * x2

  float r1 = real[ind1], r2 = real[ind2], i1 = imag[ind1], i2 = imag[ind2];
  float z = ONE_OVER_SQRT_2 * (i2 - r2);
  float temp;

  if (invert)
  {
    temp = z - SQRT_2 * i2;
    real[ind1] = r1 + temp;
    real[ind2] = r1 - temp;

    imag[ind1] = i1 - z;
    imag[ind2] = i1 + z;
  }
  else
  {
    real[ind1] = r1 + z;
    real[ind2] = r1 - z;

    temp = - SQRT_2 * r2 - z;
    imag[ind1] = i1 + temp;
    imag[ind2] = i1 - temp;
  }
}


void fourier_dit_no_reverse(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  //span = 1; num_units = size / 2; 
  for (int unit = 0, two_unit_span = 0; unit < (size >> 1); ++unit, two_unit_span += 2)
  {
    butterfly_trivial_zero_dit(real, imag, two_unit_span, two_unit_span+ 1);
  }

  //span = 2: num_units = size / 4;
  for (int unit = 0, two_unit_span = 0; unit < (size >> 2); ++unit, two_unit_span += 4)
  {
    butterfly_trivial_zero_dit(real, imag, two_unit_span,  two_unit_span + 2);
    butterfly_trivial_minus_j_dit(real, imag, 1 + two_unit_span, two_unit_span + 3, invert);
  }

  //span = 4: num_units = size / 8;
  for (int unit = 0, two_unit_span = 0; unit < (size >> 3); ++unit, two_unit_span += 8)
  {
    butterfly_trivial_zero_dit(real, imag, two_unit_span,  two_unit_span + 4);
    butterfly_trivial_one_minus_j_dit(real, imag, 1 + two_unit_span,  1 + two_unit_span + 4, invert);
    butterfly_trivial_minus_j_dit(real, imag, 2 + two_unit_span, 2 + two_unit_span + 4, invert);
    butterfly_trivial_minus_one_minus_j_dit(real, imag, 3 + two_unit_span, 3 + two_unit_span + 4, invert);
  }


  for (int span = 8, num_units = (size >> 4); span < size; span <<= 1, num_units >>= 1)
  {
    for (int unit = 0, two_unit_span = 0; unit < num_units; ++unit, two_unit_span += (span << 1))
    {
      int half_span = span >> 1;
      int quarter_span = span >> 2;
      int three_quarter_span = half_span + quarter_span;

      //i = 0
      butterfly_trivial_zero_dit(real, imag, two_unit_span, two_unit_span + span);
      for (int i = 1, twiddle_index = num_units; i < quarter_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        // float imag_twiddle = invert? -roots_imag[twiddle_index] : roots_imag[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }

      // i = quarter_span
      butterfly_trivial_one_minus_j_dit(real, imag, quarter_span + two_unit_span,  quarter_span + two_unit_span + span, invert);
      for (int i = quarter_span + 1, twiddle_index = (quarter_span + 1) * num_units; i < half_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }

      // i = halfspan
      butterfly_trivial_minus_j_dit(real, imag, half_span + two_unit_span, half_span + two_unit_span + span, invert);
      for (int i = half_span + 1, twiddle_index = (half_span + 1) * num_units; i < three_quarter_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }

      // i = three_quarter_span
      butterfly_trivial_minus_one_minus_j_dit(real, imag, three_quarter_span + two_unit_span,  three_quarter_span + two_unit_span + span, invert);
      for (int i = three_quarter_span + 1, twiddle_index = (three_quarter_span + 1) * num_units; i < span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dit(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
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


void fourier_dit(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  bit_reverse(real, rev, size);
  bit_reverse(imag, rev, size);

  fourier_dit_no_reverse(real, imag, size, rev, invert, roots_real, roots_real_plus_imag, roots_real_minus_imag);
}

void fourier_dif_no_reverse(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  for (int span = size >> 1, num_units = 1; span > 4; span >>= 1, num_units <<= 1)
  {
    for (int unit = 0, two_unit_span = 0; unit < num_units; ++unit, two_unit_span += (span << 1))
    {
      int half_span = span >> 1;
      int quarter_span = span >> 2;
      int three_quarter_span = half_span + quarter_span;

      //i = 0
      butterfly_trivial_zero_dif(real, imag, two_unit_span, two_unit_span + span);
      for (int i = 1, twiddle_index = num_units; i < quarter_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        // float imag_twiddle = invert? -roots_imag[twiddle_index] : roots_imag[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dif(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }

      // i = quarter_span
      butterfly_trivial_one_minus_j_dif(real, imag, quarter_span + two_unit_span,  quarter_span + two_unit_span + span, invert);
      for (int i = quarter_span + 1, twiddle_index = (quarter_span + 1) * num_units; i < half_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dif(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }

      // i = halfspan
      butterfly_trivial_minus_j_dif(real, imag, half_span + two_unit_span, half_span + two_unit_span + span, invert);
      for (int i = half_span + 1, twiddle_index = (half_span + 1) * num_units; i < three_quarter_span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dif(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }

      // i = three_quarter_span
      butterfly_trivial_minus_one_minus_j_dif(real, imag, three_quarter_span + two_unit_span,  three_quarter_span + two_unit_span + span, invert);
      for (int i = three_quarter_span + 1, twiddle_index = (three_quarter_span + 1) * num_units; i < span; ++i, twiddle_index += num_units)
      {
        float real_twiddle = roots_real[twiddle_index];
        float real_plus_imag_twiddle = invert? roots_real_minus_imag[twiddle_index] : roots_real_plus_imag[twiddle_index];
        float real_minus_imag_twiddle = invert? roots_real_plus_imag[twiddle_index] : roots_real_minus_imag[twiddle_index];
        butterfly_dif(real, imag, i + two_unit_span, i + two_unit_span + span, real_twiddle, real_plus_imag_twiddle, real_minus_imag_twiddle);
      }
    }
  }

  //span = 4: num_units = size / 8;
  for (int unit = 0, two_unit_span = 0; unit < (size >> 3); ++unit, two_unit_span += 8)
  {
    butterfly_trivial_zero_dif(real, imag, two_unit_span,  two_unit_span + 4);
    butterfly_trivial_one_minus_j_dif(real, imag, 1 + two_unit_span,  1 + two_unit_span + 4, invert);
    butterfly_trivial_minus_j_dif(real, imag, 2 + two_unit_span, 2 + two_unit_span + 4, invert);
    butterfly_trivial_minus_one_minus_j_dif(real, imag, 3 + two_unit_span, 3 + two_unit_span + 4, invert);
  }

  //span = 2: num_units = size / 4;
  for (int unit = 0, two_unit_span = 0; unit < (size >> 2); ++unit, two_unit_span += 4)
  {
    butterfly_trivial_zero_dif(real, imag, two_unit_span,  two_unit_span + 2);
    butterfly_trivial_minus_j_dif(real, imag, 1 + two_unit_span, two_unit_span + 3, invert);
  }

  //span = 1; num_units = size / 2; 
  for (int unit = 0, two_unit_span = 0; unit < (size >> 1); ++unit, two_unit_span += 2)
  {
    butterfly_trivial_zero_dif(real, imag, two_unit_span, two_unit_span+ 1);
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

void fourier_dif(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  fourier_dif_no_reverse(real, imag, size, rev, invert, roots_real, roots_real_plus_imag, roots_real_minus_imag);

  bit_reverse(real, rev, size);
  bit_reverse(imag, rev, size);
}

void fft_row(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  #pragma omp for schedule(static, 4)
  for (int row = 0; row < size; ++row)
  {
    fourier_dit(real + row*size, imag + row*size, size, rev, invert, roots_real, roots_real_plus_imag, roots_real_minus_imag);
  }
}

void fft_row_dif_no_reverse(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  #pragma omp for schedule(static, 4)
  for (int row = 0; row < size; ++row)
  {
    fourier_dif_no_reverse(real + row*size, imag + row*size, size, rev, invert, roots_real, roots_real_plus_imag, roots_real_minus_imag);
  }
}

void fft_row_dit_no_reverse(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  #pragma omp for schedule(static, 4)
  for (int row = 0; row < size; ++row)
  {
    fourier_dit_no_reverse(real + row*size, imag + row*size, size, rev, invert, roots_real, roots_real_plus_imag, roots_real_minus_imag);
  }
}


// void fft_col(float *real, float *imag, int size, short *rev, bool invert, float *roots_real, float *roots_imag)
// {
//   #pragma omp for
//   for (int col = 0; col < size; ++col)
//   {
//     float *real_col = new float[size];
//     float *imag_col = new float[size];
//     for(unsigned int row = 0; row < size; row++)
//     {
//       real_col[row] = real[row*size + col];
//       imag_col[row] = imag[row*size + col];
//     }

//     fourier_dit(real_col, imag_col, size, rev, invert, roots_real, roots_imag);

//     for(unsigned int row = 0; row < size; row++)
//     {
//       real[row*size + col] = real_col[row];
//       imag[row*size + col] = imag_col[row];
//     }
//   }
// }

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

void cpu_filter_and_fft(float *real, float *imag, int size, short *rev, float *roots_real, float *roots_real_plus_imag, float *roots_real_minus_imag)
{
  int eighth = size/8;
  int seven_eighths = size - eighth;

  #pragma omp for schedule (dynamic, 4)
  for(unsigned int row = 0; row < size; row++)
  {
    if (row < eighth || row >= seven_eighths)
    {
      fourier_dit(real + row*size, imag + row*size, size, rev, false, roots_real, roots_real_plus_imag, roots_real_minus_imag);
      memset(real + row*size + eighth, 0, (seven_eighths - eighth) * sizeof(float));
      memset(imag + row*size + eighth, 0, (seven_eighths - eighth) * sizeof(float));
      fourier_dit(real + row*size, imag + row*size, size, rev, true, roots_real, roots_real_plus_imag, roots_real_minus_imag);

      // fourier_dif_no_reverse(real + row*size, imag + row*size, size, rev, false, roots_real, roots_real_plus_imag, roots_real_minus_imag);
      // bit_reversed_filter(real + row*size, imag + row*size, rev, size);
      // fourier_dit_no_reverse(real + row*size, imag + row*size, size, rev, true, roots_real, roots_real_plus_imag, roots_real_minus_imag);
    }
    else
    {
      memset(real + row*size, 0, size * sizeof(float));
      memset(imag + row*size, 0, size * sizeof(float));
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
  float *roots_real_plus_imag = new float[half_size];
  float *roots_real_minus_imag = new float[half_size];
  float two_pi_over_size = - 2 * PI / size;
  for (int i = 0; i < half_size; ++i)
  {
    float term = i * two_pi_over_size;
    roots_real[i] = cos(term);
    roots_imag[i] = sin(term);
    roots_real_plus_imag[i] = roots_real[i] + roots_imag[i];
    roots_real_minus_imag[i] = roots_real[i] - roots_imag[i];
  }
  short *rev = new short[half_size];
  build_bit_rev_index(rev, size);

  execution = stats("Setup", &tv1, &tz1, &tv2, &tz2);

  #pragma omp parallel
  {
    


    // Perform fft with respect to the x direction
    fft_row(real_image, imag_image, size, rev, false, roots_real, roots_real_plus_imag, roots_real_minus_imag);

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
    // fft_row(real_image, imag_image, size, rev, false, roots_real, roots_real_plus_imag, roots_real_minus_imag);
    // fft_row_dif_no_reverse(real_image, imag_image, size, rev, false, roots_real, roots_real_plus_imag, roots_real_minus_imag);

    #pragma omp single
    { 
      execution = stats("FFTY", &tv1, &tz1, &tv2, &tz2);
    }

    // Filter the transformed image
    // cpu_filter(real_image, imag_image, size_x, size_y);
    cpu_filter_and_fft(real_image, imag_image, size, rev, roots_real, roots_real_plus_imag, roots_real_minus_imag);

    #pragma omp single
    { 
      execution = stats("Filter", &tv1, &tz1, &tv2, &tz2);
    }

    // Perform an inverse fft with respect to the x direction
    // fft_row(real_image, imag_image, size, rev, true, roots_real, roots_real_plus_imag, roots_real_minus_imag);

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
    fft_row(real_image, imag_image, size, rev, true, roots_real, roots_real_plus_imag, roots_real_minus_imag);

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

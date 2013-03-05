James Whitbeck
CS149 PA4
jamesbw

The speedup achieved was on the order of x50,000. This was achieved through the use of OpenMP parallelization, good data locality and a reasonably fast implementation of the Cooley-Tukey FFT.

Parallelism
-----------

The program contains many loops, such as iterating over all the rows or the columns to do Fourier transforms. The right level of parallelism was to split those outer loops up between the cores. Since the Fourier transform on a row takes about the same time for each row, a static schedule worked well. In the case where the filtering took place, most Fourier transforms did not need to be calculated as their result would be set to zero, or already be zero. In that phase, a dynamic schedule was used to deal with the load imbalance.

As explained below, for data locality, we transpose the matrices. This operation can be parallelized nicely for 4 cores: divide the matrix into 4 square sub-matrices, transpose each, then swap the upper-right with the lower-left.
So each transposition was given to a OpenMP section, then the swap was divided so as to also use 4 cores evenly.
Since we were transposing both the real and imaginary matrices, this made good use of the 8 cores available on AWS. However, this technique is also closely taylored to the number of cores available.

Data Locality
-------------

Operations on rows of the matrix have good data locality as the matrices are stored in row major order. That means that each cache line contains only useful data. Moreover, the cache on the AWS processors was about 20MB, which is big enough to contain 1024 * 2 floats, so the whole real and imaginary row could fit in cache. The Fourier transform was done in place to enhance the data locality.

Because of this locality, operations on rows were much faster than the same operations on columns. Therefore, we opted for a matrix transpose before operations on the columns, so that they would become operations on columns. This sped things up a lot. Moreover, it meant that the same row based code could be used for the columns.

FFT
---

The biggest speed-up was from implementing a fast Fourier transform. We implemented the radix-2 Cooley-Tukey transform. Both decimation in time and decimation in frequency were tried, but were essentially as fast as each other. Several tricks were used to increase the speed of the FFT:
 - hard-coding of trivial butterflies, to reduce the number of operations.
 - pre-computing all the trigonometry values
 - doing complex multiplication in 3 real additions and 3 real multiplications, instead of 2 and 4.
 - separating the inverse transform from the forward transform. This led to a fair amount of code duplication, but removed a lot of conditional code.
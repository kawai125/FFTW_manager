# FFTW_manager

## Introduction
STL like interface for [FFTW](http://fftw.org) library.  

## How to use
Include the `./include/fftw_allocator.hpp` and `./include/fftw_manager.hpp` in your source code, then compile and link with the FFTW liblary.

This library is developed in the environment shown in below.
 - GCC 6.4
 - OpenMPI 2.1.2
 - FFTW 3.3.7

## Calling wrapper function from your C++ program
 - Definition of data type.  
   All definitions are implemented in the namespace of `FFT_DEFS::` .  

   The float type and complex type for FFT are available from `FFT_DEFS::Manager<>` class.  
   ```c++
   //--- for 64bit version FFTW3
   using float_type   = typename FFT_DEFS::Manager<double>::float_type;    //  same to double
   using complex_type = typename FFT_DEFS::Manager<double>::complex_type;  //  same to fftw_complex
   using plan_type    = typename FFT_DEFS::Manager<double>::plan_type;     //  same to fftw_plan

   //  OR

   //--- for 32bit version FFTW3
   using float_type   = typename FFT_DEFS::Manager<float>::float_type;    //  same to float
   using complex_type = typename FFT_DEFS::Manager<float>::complex_type;  //  same to fftwf_complex
   using plan_type    = typename FFT_DEFS::Manager<float>::plan_type;     //  same to fftwf_plan
   ```

   The allocator classes for float type and complex type are defined.
   ```c++
   //--- for 64bit version FFTW3
   FFT_DEFS::Allocator<double>;
   FFT_DEFS::Allocator<std::complex<double>>;
   FFT_DEFS::Allocator<fftw_complex>;

   //  OR

   //--- for 32bit version FFTW3
   FFT_DEFS::Allocator<float>;
   FFT_DEFS::Allocator<std::complex<float>>;
   FFT_DEFS::Allocator<fftwf_complex>;
   ```

   These allocators use `fftw_malloc() / fftwf_malloc()` and `fftw_free() / fftwf_free()` internally.

 - Function interface.
   The wrapper functions for MPI distributed 3D FFT / Inverse FFT transform are defined.  
   ```c++
   template <class Tfloat>
   FFT_DEFS::Manager<Tfloat> fft_mngr;

   //--- initialize
   fft_mngr.mpi_init();
   fft_mngr.set_timelimit(const double time_limit);

   //--- memory manager interface
   void* fft_mngr.malloc(const ptrdiff_t size);
   void free(void* ptr);

   //--- MPI process decomposition interface
   ptrdiff_t fft_mngr.mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                        MPI_Comm comm,
                                        ptrdiff_t *local_n0, ptrdiff_t *local_0_start);

   //--- planner interface
   plan_type fft_mngr.mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                      complex_type *in, complex_type *out,
                                      MPI_Comm comm, int sign, unsigned flags);

   plan_type fft_mngr.mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                          float_type *in, complex_type *out,
                                          MPI_Comm comm, unsigned flags);

   plan_type fft_mngr.mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                          complex_type *in, float_type *out,
                                          MPI_Comm comm, unsigned flags);

   //--- planner interface for std::complex<T> in C++ standard
   plan_type fft_mngr.mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                      std::complex<float_type> *in,
                                      std::complex<float_type> *out,
                                      MPI_Comm comm, int sign, unsigned flags);

   plan_type fft_mngr.mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                          float_type               *in,
                                          std::complex<float_type> *out,
                                          MPI_Comm comm, unsigned flags);

   plan_type fft_mngr.mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                          std::complex<float_type> *in,
                                          float_type               *out,
                                          MPI_Comm comm, unsigned flags);

   //--- plan manager interface
   void fft_mngr.destroy_plan(plan_type plan);
   void fft_mngr.execute(plan_type plan);

   //--- wisdom interface
   void fft_mngr.mpi_broadcast_wisdom(MPI_Comm comm);
   void fft_mngr.mpi_gather_wisdom(MPI_Comm comm);
   int  fft_mngr.export_wisdom_to_filename(const char *filename);
   int  fft_mngr.import_wisdom_from_filename(const char *filename);
   ```

   The manager class dispatches these functions to `fftw_` functions or `fftwf_` functions by template parameter (`double` or `float`).

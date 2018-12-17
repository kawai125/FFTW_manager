/**************************************************************************************************/
/**
* @file  fftw_manager.hpp
* @brief generic C++ interface for FFTW functions.
*/
/**************************************************************************************************/
#pragma once

#include <complex>

#include <fftw3-mpi.h>

namespace FFT_DEFS {

    /*
    *  @brief prototype for SFINAE
    */
    template <class T>
    class Manager;

    /*
    *  @brief wrapper for 64-bit FFTW
    */
    template <>
    class Manager<double>{
    private:
        static bool init_flag;
    public:
        using float_type   = double;
        using complex_type = fftw_complex;
        using plan_type    = fftw_plan;

        Manager()  = default;
        ~Manager() = default;

        void mpi_init(){
            if( ! init_flag ){
                fftw_mpi_init();
                init_flag = true;
            }
        }
        void set_timelimit(const double time_limit){
            fftw_set_timelimit(time_limit);  // time limit for measure new planning
        }

        //--- interface for FFTW
        inline void* malloc(const ptrdiff_t size){
            return fftw_malloc(size);
        }
        inline void free(void* ptr){
            fftw_free(ptr);
        }
        inline ptrdiff_t mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                           MPI_Comm comm,
                                           ptrdiff_t *local_n0, ptrdiff_t *local_0_start){
            return fftw_mpi_local_size_3d(n0, n1, n2, comm, local_n0, local_0_start);
        }
        inline plan_type mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                         complex_type *in, complex_type *out,
                                         MPI_Comm comm, int sign, unsigned flags){
            return fftw_mpi_plan_dft_3d(n0, n1, n2, in, out, comm, sign, flags);
        }
        //! @brief interface for std::complex<T> in C++ standard
        inline plan_type mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                         std::complex<float_type> *in,
                                         std::complex<float_type> *out,
                                         MPI_Comm comm, int sign, unsigned flags){
            return fftw_mpi_plan_dft_3d(n0, n1, n2,
                                        reinterpret_cast<complex_type*>(in),
                                        reinterpret_cast<complex_type*>(out),
                                        comm, sign, flags                    );
        }
        inline plan_type mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             float_type *in, complex_type *out,
                                             MPI_Comm comm, unsigned flags){
            return fftw_mpi_plan_dft_r2c_3d(n0, n1, n2, in, out, comm, flags);
        }
        //! @brief interface for std::complex<T> in C++ standard
        inline plan_type mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             float_type               *in,
                                             std::complex<float_type> *out,
                                             MPI_Comm comm, unsigned flags){
            return fftw_mpi_plan_dft_r2c_3d(n0, n1, n2,
                                            in,
                                            reinterpret_cast<complex_type*>(out),
                                            comm, flags                          );
        }
        inline plan_type mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             complex_type *in, float_type *out,
                                             MPI_Comm comm, unsigned flags){
            return fftw_mpi_plan_dft_c2r_3d(n0, n1, n2, in, out, comm, flags);
        }
        //! @brief interface for std::complex<T> in C++ standard
        inline plan_type mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             std::complex<float_type> *in,
                                             float_type               *out,
                                             MPI_Comm comm, unsigned flags){
            return fftw_mpi_plan_dft_c2r_3d(n0, n1, n2,
                                            reinterpret_cast<complex_type*>(in),
                                            out,
                                            comm, flags                         );
        }
        inline void destroy_plan(plan_type plan){
            fftw_destroy_plan(plan);
        }
        inline void execute(plan_type plan){
            fftw_execute(plan);
        }
        inline void mpi_broadcast_wisdom(MPI_Comm comm){
            fftw_mpi_broadcast_wisdom(comm);
        }
        inline void mpi_gather_wisdom(MPI_Comm comm){
            fftw_mpi_gather_wisdom(comm);
        }
        inline int export_wisdom_to_filename(const char *filename){
            return fftw_export_wisdom_to_filename(filename);
        }
        inline int import_wisdom_from_filename(const char *filename){
            return fftw_import_wisdom_from_filename(filename);
        }
    };
    bool Manager<double>::init_flag = false;

    /*
    *  @brief wrapper for 32-bit FFTW
    */
    template <>
    class Manager<float>{
    private:
        static bool init_flag;
    public:
        using float_type   = float;
        using complex_type = fftwf_complex;
        using plan_type    = fftwf_plan;

        Manager()  = default;
        ~Manager() = default;

        void mpi_init() {
            if( ! init_flag ){
                fftwf_mpi_init();
                init_flag = true;
            }
        }
        void set_timelimit(const double time_limit){
            fftwf_set_timelimit(time_limit);  // time limit for measure new planning
        }

        //--- interface for FFTW
        inline void* malloc(const ptrdiff_t size){
            return fftwf_malloc(size);
        }
        inline void free(void* ptr){
            fftwf_free(ptr);
        }
        inline ptrdiff_t mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, MPI_Comm comm,
                                           ptrdiff_t *local_n0, ptrdiff_t *local_0_start){
            return fftwf_mpi_local_size_3d(n0, n1, n2, comm, local_n0, local_0_start);
        }
        inline plan_type mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                         complex_type *in, complex_type *out,
                                         MPI_Comm comm, int sign, unsigned flags){
            return fftwf_mpi_plan_dft_3d(n0, n1, n2, in, out, comm, sign, flags);
        }
        //! @brief interface for std::complex<T> in C++ standard
        inline plan_type mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                         std::complex<float_type> *in,
                                         std::complex<float_type> *out,
                                         MPI_Comm comm, int sign, unsigned flags){
            return fftwf_mpi_plan_dft_3d(n0, n1, n2,
                                         reinterpret_cast<complex_type*>(in),
                                         reinterpret_cast<complex_type*>(out),
                                         comm, sign, flags                    );
        }
        inline plan_type mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             float_type *in, complex_type *out,
                                             MPI_Comm comm, unsigned flags){
            return fftwf_mpi_plan_dft_r2c_3d(n0, n1, n2, in, out, comm, flags);
        }
        //! @brief interface for std::complex<T> in C++ standard
        inline plan_type mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             float_type               *in,
                                             std::complex<float_type> *out,
                                             MPI_Comm comm, unsigned flags){
            return fftwf_mpi_plan_dft_r2c_3d(n0, n1, n2,
                                             in,
                                             reinterpret_cast<complex_type*>(out),
                                             comm, flags                          );
        }
        inline plan_type mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             complex_type *in, float_type *out,
                                             MPI_Comm comm, unsigned flags){
            return fftwf_mpi_plan_dft_c2r_3d(n0, n1, n2, in, out, comm, flags);
        }
        //! @brief interface for std::complex<T> in C++ standard
        inline plan_type mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                             std::complex<float_type> *in,
                                             float_type               *out,
                                             MPI_Comm comm, unsigned flags){
            return fftwf_mpi_plan_dft_c2r_3d(n0, n1, n2,
                                             reinterpret_cast<complex_type*>(in),
                                             out,
                                             comm, flags                         );
        }
        inline void destroy_plan(plan_type plan){
            fftwf_destroy_plan(plan);
        }
        inline void execute(plan_type plan){
            fftwf_execute(plan);
        }
        inline void mpi_broadcast_wisdom(MPI_Comm comm){
            fftwf_mpi_broadcast_wisdom(comm);
        }
        inline void mpi_gather_wisdom(MPI_Comm comm){
            fftwf_mpi_gather_wisdom(comm);
        }
        inline int export_wisdom_to_filename(const char *filename){
            return fftwf_export_wisdom_to_filename(filename);
        }
        inline int import_wisdom_from_filename(const char *filename){
            return fftwf_import_wisdom_from_filename(filename);
        }
    };
    bool Manager<float>::init_flag = false;

}

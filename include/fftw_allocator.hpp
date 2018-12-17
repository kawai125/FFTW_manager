/**************************************************************************************************/
/**
* @file  fftw_allocator.hpp
* @brief generic C++ interface for float & complex type of FFTW.
*/
/**************************************************************************************************/
#pragma once

#include <complex>

#include <fftw3.h>

namespace FFT_DEFS {

    //--- SFINAE prototype for Allocator<T>
    template <class T>
    class Allocator{};

    template <>
    class Allocator<double>{
    public:
        Allocator() {}

        template <class U>
        Allocator(const Allocator<U>&) {}

        Allocator& operator = (const Allocator& rv) { return *this; }

        using value_type      = double;
        using pointer         = value_type*;
        using const_pointer   = const value_type*;
        using reference       = value_type&;
        using const_reference = const value_type&;

        pointer allocate(const ptrdiff_t size){
            return reinterpret_cast<pointer>(fftw_malloc(sizeof(value_type)*size));
        }
        void deallocate(pointer ptr, size_t n){
            static_cast<void>(n);
            fftw_free(ptr);
        }
        void deallocate(pointer ptr){
            fftw_free(ptr);
        }
    };

    template <>
    class Allocator<std::complex<double>>{
    public:
        Allocator() {}

        template <class U>
        Allocator(const Allocator<U>&) {}

        Allocator& operator = (const Allocator& rv) { return *this; }

        using value_type      = std::complex<double>;
        using pointer         = value_type*;
        using const_pointer   = const value_type*;
        using reference       = value_type&;
        using const_reference = const value_type&;

        pointer allocate(const ptrdiff_t size){
            return reinterpret_cast<pointer>(fftw_malloc(sizeof(value_type)*size));
        }
        void deallocate(pointer ptr, size_t n){
            static_cast<void>(n);
            fftw_free(ptr);
        }
        void deallocate(pointer ptr){
            fftw_free(ptr);
        }
    };

    template <>
    class Allocator<fftw_complex>{
    public:
        Allocator() {}

        template <class U>
        Allocator(const Allocator<U>&) {}

        Allocator& operator = (const Allocator& rv) { return *this; }

        using value_type      = fftw_complex;
        using pointer         = value_type*;
        using const_pointer   = const value_type*;
        using reference       = value_type&;
        using const_reference = const value_type&;

        pointer allocate(const ptrdiff_t size){
            return reinterpret_cast<pointer>(fftw_malloc(sizeof(value_type)*size));
        }
        void deallocate(pointer ptr, size_t n){
            static_cast<void>(n);
            fftw_free(ptr);
        }
        void deallocate(pointer ptr){
            fftw_free(ptr);
        }

        //--- fftw_complex = double[2], this is not class.
        void construct(pointer ptr, const double&& v0, const double&& v1){
            (*ptr)[0] = v0;
            (*ptr)[1] = v1;
        }
        void construct(pointer ptr, const value_type&& value){
            (*ptr)[0] = value[0];
            (*ptr)[1] = value[1];
        }
        void destroy(pointer ptr){}
    };

    template <>
    class Allocator<float>{
    public:
        Allocator() {}

        template <class U>
        Allocator(const Allocator<U>&) {}

        Allocator& operator = (const Allocator& rv) { return *this; }

        using value_type      = float;
        using pointer         = value_type*;
        using const_pointer   = const value_type*;
        using reference       = value_type&;
        using const_reference = const value_type&;

        pointer allocate(const ptrdiff_t size){
            return reinterpret_cast<pointer>(fftwf_malloc(sizeof(value_type)*size));
        }
        void deallocate(pointer ptr, size_t n){
            static_cast<void>(n);
            fftwf_free(ptr);
        }
        void deallocate(pointer ptr){
            fftwf_free(ptr);
        }
    };

    template <>
    class Allocator<std::complex<float>>{
    public:
        Allocator() {}

        template <class U>
        Allocator(const Allocator<U>&) {}

        Allocator& operator = (const Allocator& rv) { return *this; }

        using value_type      = std::complex<float>;
        using pointer         = value_type*;
        using const_pointer   = const value_type*;
        using reference       = value_type&;
        using const_reference = const value_type&;

        pointer allocate(const ptrdiff_t size){
            return reinterpret_cast<pointer>(fftwf_malloc(sizeof(value_type)*size));
        }
        void deallocate(pointer ptr, size_t n){
            static_cast<void>(n);
            fftwf_free(ptr);
        }
        void deallocate(pointer ptr){
            fftwf_free(ptr);
        }
    };

    template <>
    class Allocator<fftwf_complex>{
    public:
        Allocator() {}

        template <class U>
        Allocator(const Allocator<U>&) {}

        Allocator& operator = (const Allocator& rv) { return *this; }

        using value_type      = fftwf_complex;
        using pointer         = value_type*;
        using const_pointer   = const value_type*;
        using reference       = value_type&;
        using const_reference = const value_type&;

        pointer allocate(const ptrdiff_t size){
            return reinterpret_cast<pointer>(fftwf_malloc(sizeof(value_type)*size));
        }
        void deallocate(pointer ptr, size_t n){
            static_cast<void>(n);
            fftwf_free(ptr);
        }
        void deallocate(pointer ptr){
            fftwf_free(ptr);
        }

        //--- fftwf_complex = float[2], this is not class.
        void construct(pointer ptr, const float&& v0, const float&& v1){
            (*ptr)[0] = v0;
            (*ptr)[1] = v1;
        }
        void construct(pointer ptr, const value_type&& value){
            (*ptr)[0] = value[0];
            (*ptr)[1] = value[1];
        }
        void destroy(pointer ptr){}
    };

    template <class T1, class T2>
    bool operator == (const Allocator<T1> &lv, const Allocator<T2> &rv){ return true; }

    template <class T1, class T2>
    bool operator != (const Allocator<T1> &lv, const Allocator<T2> &rv){ return false; }
}

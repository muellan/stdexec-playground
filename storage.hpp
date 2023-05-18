#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include "vecmath.hpp"

#ifdef USE_GPU
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/memory.h>
#endif

#include <cstdio>
#include <vector>
#include <random>
#include <span>
#include <algorithm>


namespace ex {


//-----------------------------------------------------------------------------
// GPU specific declarations & definitions
#ifdef USE_GPU

template <typename T>
using vector = thrust::universal_vector<T>; 


struct random_engine {};

struct random_double_generator 
{
    __device__
    double operator () (int idx)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<double> distr;
        rng.discard(idx);
        return distr(rng);
    }
};

struct random_vec3d_generator 
{
    __device__
    vec3d operator () (int idx)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<double> distr;
        rng.discard(3*idx);
        return vec3d {distr(rng), distr(rng), distr(rng)};
    }
};



struct print_t
{
  __host__ __device__
  void operator()(int x) { printf("%d\n", x); }
};

inline constexpr print_t print {};


// CPU specific declarations & definitions
#else

template <typename T>
using vector = std::vector<T>; 

using random_engine = std::mt19937_64;

#endif



//-----------------------------------------------------------------------------
#ifdef USE_GPU
void fill_random (thrust::universal_vector<double>& v, auto)
{
    thrust::transform(thrust::make_counting_iterator(std::size_t(0)),
                      thrust::make_counting_iterator(v.size()),
                      v.begin(),
                      random_double_generator{});
}
#else
void fill_random (std::span<double> v, auto& urng)
{
    auto distr = std::uniform_real_distribution<double>{-1.0,+1.0};
    std::ranges::generate(v, [&]{ return distr(urng); });
}
#endif




//-----------------------------------------------------------------------------
#ifdef USE_GPU
void fill_random (thrust::universal_vector<vec3d>& v, auto)
{
    thrust::transform(thrust::make_counting_iterator(std::size_t(0)),
                      thrust::make_counting_iterator(v.size()),
                      v.begin(),
                      random_vec3d_generator{});
}
#else
void fill_random (std::span<vec3d> v, auto& urng)
{
    auto distr = std::uniform_real_distribution<double>{-1.0,+1.0};

    std::ranges::generate(v, [&]{ 
        return vec3d{distr(urng), distr(urng), distr(urng)}; });
}
#endif




//-----------------------------------------------------------------------------
#ifdef USE_GPU
template <class Value>
void fill_default (thrust::universal_vector<Value>& v)
{
    thrust::fill(v.begin(), v.end(), Value{});
}
#else
template <std::ranges::input_range Range>
void fill_default (Range& r)
{
    using value_t = std::ranges::range_value_t<Range>;
    std::ranges::fill(r, value_t{});
}
#endif




//-----------------------------------------------------------------------------
#ifdef USE_GPU

template <class T>
struct uninitialized_allocator : thrust::cuda::universal_allocator<T>
{
    __host__
    uninitialized_allocator() {}

    __host__
    uninitialized_allocator (uninitialized_allocator const& other):
        thrust::device_allocator<T>(other)
    {}

    __host__
    ~uninitialized_allocator () {}

    uninitialized_allocator& operator = (const uninitialized_allocator &) = default;

    template <class U>
    struct rebind { using other = uninitialized_allocator<U>; };

    __host__ __device__
    void construct(T *) { }
};

template <class T>
using uninitialized_vector = thrust::universal_vector<T, uninitialized_allocator<T>>;


#else

template <class T, class Alloc = std::allocator<T> >
class uninitialized_allocator : public Alloc
{
    using a_t = std::allocator_traits<Alloc>;

public:
    template <class U>
    struct rebind {
        using other = uninitialized_allocator<U,
              typename a_t::template rebind_alloc<U> >;
    };

    using Alloc::Alloc;  

    template <class U>
    void construct (U* ptr)
    noexcept(std::is_nothrow_default_constructible<U>::value)
    { 
        ::new(static_cast<void*>(ptr)) U;
    }

    template <class U, class... Args>
    void construct (U* ptr, Args&&... args) {
        a_t::construct( static_cast<Alloc&>(*this),
                        ptr, std::forward<Args>(args)...);
    }
};

template <class T>
using uninitialized_vector = std::vector<T, uninitialized_allocator<T>>;

#endif


}  // namespace ex

#endif

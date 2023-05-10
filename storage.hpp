#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include "vecmath.hpp"

#ifdef USE_GPU
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#endif

#include <vector>
#include <random>
#include <span>
#include <algorithm>


//-----------------------------------------------------------------------------
#ifdef USE_GPU

template <typename T>
[[nodiscard]] constexpr auto
view_of (thrust::device_vector<T>& v) noexcept
{ 
    return std::span<T>{thrust::raw_pointer_cast(v.data()), v.size()}; 
}

template <typename T>
[[nodiscard]] constexpr auto
view_of (thrust::device_vector<T> const& v) noexcept { 
    return std::span<T const>{thrust::raw_pointer_cast(v.data()), v.size()}; 
}

#endif


template <std::ranges::contiguous_range Range>
[[nodiscard]] constexpr auto
view_of (Range&& r) noexcept
{ 
    return std::span{std::ranges::data(std::forward<Range>(r)),
                     std::ranges::size(std::forward<Range>(r))};
}


template <typename T>
[[nodiscard]] constexpr auto
view_of (std::span<T> s) noexcept { return s; }




//-----------------------------------------------------------------------------
// GPU specific declarations & definitions
#ifdef USE_GPU
template <typename T>
using bulk_vector = thrust::device_vector<T>; 

struct random_engine {};

struct random_double_generator 
{
    DEVICEQUALIFIER
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
    DEVICEQUALIFIER
    vec3d operator () (int idx)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<double> distr;
        rng.discard(3*idx);
        return vec3d {distr(rng), distr(rng), distr(rng)};
    }
};

// CPU specific declarations & definitions
#else

template <typename T>
using bulk_vector = std::vector<T>; 

using random_engine = std::mt19937_64;

#endif



//-----------------------------------------------------------------------------
#ifdef USE_GPU
void fill_random (thrust::device_vector<double>& v, auto)
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
void fill_random (thrust::device_vector<vec3d>& v, auto)
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


#endif

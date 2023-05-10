#ifndef VECMATH_HPP
#define VECMATH_HPP

#include "cuda_helpers.cuh"


//-----------------------------------------------------------------------------
struct vec3d
{
    double x, y, z;

    vec3d& operator += (vec3d const& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};


[[nodiscard]]
HOSTDEVICEQUALIFIER INLINEQUALIFIER 
constexpr vec3d
operator + (vec3d a, vec3d const& b) noexcept
{
    a += b;
    return a;
}


[[nodiscard]]
HOSTDEVICEQUALIFIER INLINEQUALIFIER 
constexpr double
dot (vec3d const& a, vec3d const& b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


[[nodiscard]]
HOSTDEVICEQUALIFIER INLINEQUALIFIER 
constexpr vec3d
cross (vec3d const& a, vec3d const& b)
{
    return vec3d { a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x };
}


HOSTDEVICEQUALIFIER INLINEQUALIFIER 
constexpr 
void cross (vec3d const& a, vec3d const& b, vec3d& c)
{
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
}



#endif

#ifndef VECMATH_HPP
#define VECMATH_HPP



//-----------------------------------------------------------------------------
struct vec3d
{
    double x, y, z;
};


[[nodiscard]]
inline constexpr vec3d
operator + (vec3d a, vec3d const& b) noexcept
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}


[[nodiscard]]
inline constexpr double
dot (vec3d const& a, vec3d const& b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


[[nodiscard]]
inline constexpr vec3d
cross (vec3d const& a, vec3d const& b)
{
    return vec3d { a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x };
}


inline constexpr 
void cross (vec3d const& a, vec3d const& b, vec3d& c)
{
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
}



#endif

#ifndef VECMATH_HPP
#define VECMATH_HPP

#include "acceleration.hpp"

#ifndef USE_GPU
    #include <fmt/format.h>
#endif


//-----------------------------------------------------------------------------
struct vec3d
{
    inline static constexpr int dims = 3;
    using value_type = double;

    value_type x, y, z;

    HOSTDEVICEQUALIFIER 
    vec3d& operator *= (value_type s) noexcept {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    HOSTDEVICEQUALIFIER 
    vec3d& operator /= (value_type s) noexcept {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    HOSTDEVICEQUALIFIER 
    vec3d& operator += (vec3d const& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    HOSTDEVICEQUALIFIER 
    vec3d& operator -= (vec3d const& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }


    [[nodiscard]]
    HOSTDEVICEQUALIFIER 
    friend constexpr vec3d
    operator + (vec3d a, vec3d const& b) noexcept
    {
        a += b;
        return a;
    }

    [[nodiscard]]
    HOSTDEVICEQUALIFIER 
    friend constexpr vec3d
    operator - (vec3d a, vec3d const& b) noexcept
    {
        a -= b;
        return a;
    }

    [[nodiscard]]
    HOSTDEVICEQUALIFIER 
    friend constexpr vec3d
    operator * (vec3d::value_type s, vec3d v) noexcept
    {
        v *= s;
        return v;
    }

    [[nodiscard]]
    HOSTDEVICEQUALIFIER 
    friend constexpr vec3d
    operator * (vec3d v, vec3d::value_type s) noexcept
    {
        v *= s;
        return v;
    }


    [[nodiscard]]
    HOSTDEVICEQUALIFIER 
    friend constexpr double
    dot (vec3d const& a, vec3d const& b) noexcept
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }


    [[nodiscard]]
    HOSTDEVICEQUALIFIER 
    friend constexpr vec3d
    cross (vec3d const& a, vec3d const& b)
    {
        return vec3d { a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x };
    }


    HOSTDEVICEQUALIFIER 
    friend constexpr 
    void cross (vec3d const& a, vec3d const& b, vec3d& c)
    {
        c.x = a.y * b.z - a.z * b.y;
        c.y = a.z * b.x - a.x * b.z;
        c.z = a.x * b.y - a.y * b.x;
    }
};




//-----------------------------------------------------------------------------
#ifndef USE_GPU
template <>
class fmt::formatter<vec3d>
{
    // format spec storage
    char presentation_ = 'f';
public:
    // parse format specification and store it
    constexpr auto 
    parse (format_parse_context& ctx) { 
        auto i = ctx.begin(), end = ctx.end();
        if (i != end && (*i == 'f' || *i == 'e')) {
            presentation_ = *i++;
        }
        if (i != end && *i != '}') {
            throw format_error("invalid format");
        }
        return i;
    }

    // format value using stored specification
    template <typename FmtContext>
    constexpr auto 
    format (vec3d const& v, FmtContext& ctx) const {
        switch (presentation_) {
            default:
            // 'ctx.out()' is an output iterator
            case 'f': return format_to(ctx.out(), "({:f},{:f},{:f})", v.x, v.y, v.z);
            case 'e': return format_to(ctx.out(), "({:e},{:e},{:e})", v.x, v.y, v.z);
        }  
    }
};
#endif


#endif

#ifndef SPAN2D_HPP_
#define SPAN2D_HPP_

#include "acceleration.hpp"

#ifdef USE_GPU
    #include <thrust/device_ptr.h>
#else
    #include <fmt/format.h>
    #include <fmt/ranges.h>
#endif


//-----------------------------------------------------------------------------
template <std::floating_point ElemT>
class span2d
{
    ElemT* data_ = nullptr;
    std::size_t nrows_ = 0;
    std::size_t ncols_ = 0;

public:
    using value_type = ElemT;

    HOSTDEVICEQUALIFIER
    constexpr 
    span2d () = default;

    HOSTDEVICEQUALIFIER
    constexpr
    span2d (value_type* data, std::size_t nrows, std::size_t ncols) noexcept:
        data_{data}, nrows_{nrows}, ncols_{ncols}
    {}

#ifdef USE_GPU
    HOSTDEVICEQUALIFIER
    constexpr
    span2d (thrust::device_ptr<ElemT> ptr, std::size_t nrows, std::size_t ncols) noexcept:
        data_{ptr.get()}, nrows_{nrows}, ncols_{ncols}
    {}
#endif

    [[nodiscard]] 
    HOSTDEVICEQUALIFIER
    constexpr value_type& 
    operator () (std::size_t row, std::size_t col) noexcept { 
        return data_[row * ncols_ + col];
    }

    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr value_type  
    operator () (std::size_t row, std::size_t col) const noexcept { 
        return data_[row * ncols_ + col];
    }

    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr auto ncols ()  const noexcept { return ncols_; }

    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr auto nrows ()  const noexcept { return nrows_; }
}; 




//-----------------------------------------------------------------------------
#ifndef USE_GPU
template <std::floating_point T>
void print (span2d<T> const& s)
{
    fmt::print("{} x {}\n", s.nrows(), s.ncols());

    fmt::print("   ");
    for (std::size_t c = 0; c < s.ncols(); ++c) {
        fmt::print("  {:2}  ", c);
    }
    fmt::print("\n");

    for (std::size_t r = 0; r < s.nrows(); ++r) {
        fmt::print("{:2} | ", r);
        for (std::size_t c = 0; c < s.ncols(); ++c) {
            fmt::print(" {:4.2f} ", s(r,c));
        }
        fmt::print("\n");
    }
}
#endif

#endif

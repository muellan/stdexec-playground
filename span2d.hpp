#ifndef SPAN2D_HPP_
#define SPAN2D_HPP_


#include <fmt/format.h>
#include <fmt/ranges.h>


//-----------------------------------------------------------------------------
template <std::floating_point ElemT>
class span2d
{
    ElemT* data_;
    std::size_t nrows_ = 0;
    std::size_t ncols_ = 0;
public:
    using value_type = ElemT;

    span2d () = default;

    constexpr
    span2d (value_type* data, std::size_t nrows, std::size_t ncols) noexcept:
        data_{data}, nrows_{nrows}, ncols_{ncols}
    {}

    [[nodiscard]] constexpr value_type& 
    operator () (std::size_t row, std::size_t col) noexcept { 
        return data_[row * ncols_ + col];
    }

    [[nodiscard]] constexpr value_type  
    operator () (std::size_t row, std::size_t col) const noexcept { 
        return data_[row * ncols_ + col];
    }

    [[nodiscard]] constexpr auto ncols ()  const noexcept { return ncols_; }
    [[nodiscard]] constexpr auto nrows ()  const noexcept { return nrows_; }
}; 




//-----------------------------------------------------------------------------
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

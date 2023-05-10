#ifndef STDEXEC_INDICES_HPP
#define STDEXEC_INDICES_HPP


#include <compare>
#include <iterator>
#include <cstdint>


namespace am {


//-----------------------------------------------------------------------------
/** 
 * @brief  non-modifiable range that (conceptually) contains consecutive indices;
 *         the upper bound is excluded from the range
 */
class index_range
{
public:
    using value_type = std::size_t;
    using size_type  = value_type;

private:
    value_type beg_ = 0;
    value_type end_ = 0;

public:
    class iterator {
        value_type i_ = 0;
    public:
        using iterator_category = std::contiguous_iterator_tag;
        using value_type = index_range::value_type;
        using difference_type = std::int64_t;

        constexpr
        iterator () = default;

        constexpr explicit
        iterator (value_type i) noexcept : i_{i} {}

        [[nodiscard]] constexpr
        value_type operator * () const noexcept { return i_; }

        [[nodiscard]] constexpr
        value_type operator [] (difference_type offset) const noexcept { 
            return i_ + offset;
        }

        constexpr auto operator <=> (iterator const&) const noexcept = default;

        constexpr iterator& operator ++ () noexcept { ++i_; return *this; }
        constexpr iterator& operator -- () noexcept { ++i_; return *this; }

        constexpr iterator
        operator ++ (int) noexcept { 
            auto old {*this};
            ++i_;
            return old;
        }

        constexpr iterator
        operator -- (int) noexcept { 
            auto old {*this};
            --i_;
            return old;
        }

        constexpr iterator&
        operator += (difference_type offset) noexcept { 
            i_ += offset;
            return *this;
        }

        constexpr iterator&
        operator -= (difference_type offset) noexcept { 
            i_ -= offset;
            return *this;
        }

        [[nodiscard]] constexpr friend iterator
        operator + (iterator it, difference_type offset) noexcept { 
            return iterator{it.i_ + offset}; 
        }

        [[nodiscard]] constexpr friend iterator
        operator + (difference_type offset, iterator it) noexcept { 
            return iterator{offset + it.i_}; 
        }

        [[nodiscard]] constexpr friend iterator
        operator - (iterator it, difference_type offset) noexcept { 
            return iterator{it.i_ - offset}; 
        }

        [[nodiscard]] constexpr friend iterator
        operator - (difference_type offset, iterator it) noexcept { 
            return iterator{offset - it.i_}; 
        }

        [[nodiscard]] friend constexpr
        difference_type operator - (iterator const& l, iterator const& r) noexcept { 
            return difference_type(r.i_) - difference_type(l.i_);
        }
    };

    using const_iterator = iterator;


    constexpr
    index_range () = default;

    constexpr explicit
    index_range (value_type end) noexcept:
        beg_{0}, end_{end}
    {}

    constexpr explicit
    index_range (value_type beg, value_type end) noexcept:
        beg_{beg}, end_{end}
    {}


    [[nodiscard]] constexpr
    value_type operator [] (size_type offset) const noexcept { return beg_ + offset; }


    [[nodiscard]] constexpr
    size_type size () const noexcept { return end_ - beg_; }

    [[nodiscard]] constexpr
    bool empty () const noexcept { return end_ <= beg_; }


    [[nodiscard]] constexpr
    iterator begin () const noexcept { return iterator{beg_}; }

    [[nodiscard]] constexpr
    iterator end () const noexcept { return iterator{end_}; }


    [[nodiscard]] friend constexpr
    iterator begin (index_range const& r) noexcept { return r.begin(); }

    [[nodiscard]] friend constexpr
    iterator end (index_range const& r) noexcept { return r.end(); }
};



[[nodiscard]] inline constexpr auto
view_of (index_range const& s) noexcept { return s; }






//-----------------------------------------------------------------------------
template <int N, typename ValueT = std::size_t>
class index
{ 
    static_assert(N > 0);

    using value_type      = ValueT;
    using size_type       = std::size_t;
    using difference_type = std::int64_t;

    value_type idx_[N];

public:
    index () = default;

    template <typename... Is>
        // requires requires { sizeof...(Is) == N; }
    explicit constexpr
    index (Is... indices) noexcept : idx_{indices...} {}

    constexpr 
    auto& operator [] (size_type i) noexcept { return idx_[i]; }

    constexpr 
    auto operator [] (size_type i) const noexcept { return idx_[i]; }

    constexpr 
    auto front () const noexcept { return idx_[0]; }

    constexpr 
    auto back () const noexcept { return idx_[N-1]; }

    [[nodiscard]] constexpr
    auto size () const noexcept  { return N; }

    [[nodiscard]] constexpr
    bool empty () const noexcept { return false; }

    constexpr bool operator == (index const&) const noexcept = default;

    constexpr auto operator <=> (index const& rhs) const noexcept
    {
        for (int i = 0; i < N; ++i) {
            auto const c = idx_[i] <=> rhs.idx_[i];
            if (c != std::strong_ordering::equal) return c;
        }
        return std::strong_ordering::equal;
    }
};


using index2 = index<2>;
using index3 = index<3>;
using index4 = index<4>;
using index5 = index<5>;






//-----------------------------------------------------------------------------
/** 
 * @brief  non-modifiable range that (conceptually) contains 
 *         indices of points in an N-dimensional grid
 *       - upper bounds are excluded from the coordinate ranges
 *       - on increment, the first index dimension changes most frequently 
 */
// template <int N>
// class index_grid
// {
//     static_assert(N > 0);
//
// public:
//     class sentinel_type {};
//
//
//     class iterator {
//         friend class index_grid;
//     public:
//         using coord_type = std::size_t;
//         using value_type        = index<N,coord_type>;
//         using difference_type   = std::int64_t;
//         using iterator_category = std::random_access_iterator_tag;
//
//     private:
//         value_type cur_;
//         value_type end_;
//         value_type mul_;
//
//     public:
//         constexpr
//         iterator () = default;
//
//         constexpr explicit
//         iterator (value_type const& end) noexcept : end_{end}
//         {
//             coord_type p = 1; 
//             for (int i = 0; i < N; ++i) {
//                 cur_[i] = 0;
//                 mul_[i] = p;
//                 p *= end_[i];
//             }
//         }
//
//         constexpr explicit
//         iterator (value_type const& start, value_type const& end) noexcept :
//             cur_{start}, end_{end}
//         {
//             coord_type p = 1; 
//             for (int i = 0; i < N; ++i) {
//                 mul_[i] = p;
//                 p *= end_[i];
//             }
//         }
//
//         [[nodiscard]] constexpr
//         value_type operator * () const noexcept { return cur_; }
//
//         [[nodiscard]] constexpr
//         value_type operator [] (difference_type offset) const noexcept { 
//             return cur_ + offset;
//         }
//
//         constexpr auto operator <=> (iterator const& rhs) const noexcept {
//             return cur_ <=> rhs.cur_;
//         }
//
//         constexpr auto operator == (iterator const& rhs) const noexcept {
//             return cur_ == rhs.cur_;
//         }
//
//         constexpr bool operator == (sentinel_type const&) const noexcept {
//             return cur_ != end_;
//         }
//
//         constexpr iterator& operator ++ () noexcept 
//         {
//             for (int i = 0; i < N; ++i) {
//                 ++cur_[i];
//                 if (cur_[i] < end_[i]) return *this;
//                 if (i < N-1 && cur_[i+1] < end_[i+1] - 1) {
//                     cur_[i] = 0;
//                 } else {
//                     --cur_[i];
//                 }
//             }
//             return *this;
//         }
//
//         constexpr iterator& operator -- () noexcept 
//         {
//             for (int i = 0; i < N; ++i) {
//                 if (cur_[i] > 0) {
//                     --cur_[i];
//                     return *this;
//                 }
//                 cur_[i] = 0;
//                 if (i < N-1) cur_[i+1] = 0;
//             }
//             return *this;
//         }
//
//         constexpr iterator&
//         operator += (difference_type offset) noexcept 
//         { 
//             if (offset > mul_[N-1]) {
//                 cur_ = end_;
//                 return *this;
//             }
//             for (int i = N-2; i > 0; --i) {
//                 if (offset < mul_[i]) {
//                     cur_[i] += offset;
//                     return *this;
//                 } else {
//                     cur_[i+1] += offset / mul_[i];
//                     offset = offset % mul_[i];
//                 }
//             }
//             return *this;
//         }
//
//         constexpr iterator&
//         operator -= (difference_type offset) noexcept 
//         { 
//             if (offset > mul_[N-1]) {
//                 for (int i = 0; i < N; ++i) cur_[i] = 0; 
//                 return *this;
//             }
//             for (int i = N-2; i >= 0; --i) {
//                 if (offset < mul_[i]) {
//                     cur_[i] -= offset;
//                     return *this;
//                 } else {
//                     cur_[i+1] -= offset / mul_[i];
//                     offset = offset % mul_[i];
//                 }
//             }
//             return *this;
//         }
//
//         constexpr iterator
//         operator ++ (int) noexcept { 
//             auto old {*this};
//             this->operator++();
//             return old;
//         }
//
//         constexpr iterator
//         operator -- (int) noexcept { 
//             auto old {*this};
//             this->operator--();
//             return old;
//         }
//
//         [[nodiscard]] constexpr friend iterator
//         operator + (iterator it, difference_type offset) noexcept { 
//             it += offset;
//             return it;
//         }
//
//         [[nodiscard]] constexpr friend iterator
//         operator + (difference_type offset, iterator it) noexcept { 
//             it += offset;
//             return it;
//         }
//
//         [[nodiscard]] constexpr friend iterator
//         operator - (iterator it, difference_type offset) noexcept { 
//             it -= offset;
//             return it;
//         }
//
//         [[nodiscard]] constexpr friend iterator
//         operator - (difference_type offset, iterator it) noexcept { 
//             it -= offset;
//             return it;
//         }
//
//         [[nodiscard]] friend constexpr
//         difference_type operator - (iterator const& l, iterator const& r) noexcept { 
//             if (l.end_ != r.end_) return 0;
//             difference_type d = 0;
//             for (std::size_t i = 0; i < N; ++i) {
//                 d += (r[i] - l[i]) * l.mul_[i];
//             }
//             return d;
//
//         }
//     };
//
//     using coord_type     = iterator::coord_type;
//     using value_type     = iterator::value_type;
//     using size_type      = std::size_t;
//     using const_iterator = iterator;
//
//
//     constexpr
//     index_grid () = default;
//
//     constexpr explicit
//     index_grid (value_type end) noexcept:
//         state_{end}
//     {}
//
//     constexpr explicit
//     index_grid (value_type beg, value_type end) noexcept:
//         state_{beg,end}
//     {}
//
//
//     [[nodiscard]] constexpr
//     size_type size () const noexcept { return state_.mul_[N-1]; }
//
//
//     [[nodiscard]] constexpr
//     value_type const& extent () const noexcept { return state_.end_; }
//
//
//     [[nodiscard]] constexpr
//     bool empty () const noexcept { return state_.cur_ == state_.end_; }
//
//
//     [[nodiscard]] constexpr
//     auto begin () const noexcept { return state_; }
//
//     [[nodiscard]] constexpr
//     auto end () const noexcept { return sentinel_type{}; }
//
//
// private:
//     iterator state_;
// };
//
//
//
// // deduction guides
// template <int N, typename T>
// index_grid (index<N,T> const&) -> index_grid<N>;
//
// template <int N, typename T>
// index_grid (index<N,T> const&, index<N,T> const&) -> index_grid<N>;


}  // namespace am

#endif



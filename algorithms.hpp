#ifndef STDEXEC_ALGORITHMS_HPP
#define STDEXEC_ALGORITHMS_HPP


#include "execution.hpp"
#include "span2d.hpp"
#include "indices.hpp"
#include "storage.hpp"
#include <type_traits>

#ifdef USE_GPU
    #include <thrust/device_vector.h>
    #include <thrust/reduce.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <concepts>
#include <span>
#include <array>
#include <algorithm>
#include <execution>
#include <numeric>
#include <ranges>
#include <utility>


//-----------------------------------------------------------------------------
// template <typename... Ts>
// using any_sender_of =
//     typename exec::any_receiver_ref<stdexec::completion_signatures<Ts...>>::template any_sender<>;




//-----------------------------------------------------------------------------
template <typename Fn, typename Range>
concept IndexToValueMapping = 
    std::copy_constructible<Fn> &&
    std::invocable<Fn,std::size_t> &&
    std::convertible_to<std::invoke_result_t<Fn,std::size_t>,
                        std::ranges::range_value_t<Range>>;



template <typename Fn, typename Range>
concept Transformation = 
    std::copy_constructible<Fn> &&
    std::invocable<Fn,std::ranges::range_value_t<Range>> &&
    std::convertible_to<std::invoke_result_t<Fn,std::ranges::range_value_t<Range>>,
                        std::ranges::range_value_t<Range>>;



template <typename Fn, typename Range>
concept IndexTransformation = 
    std::copy_constructible<Fn> &&
    std::invocable<Fn,std::ranges::range_value_t<Range>,std::size_t> &&
    std::convertible_to<std::invoke_result_t<Fn,std::ranges::range_value_t<Range>,std::size_t>,
                        std::ranges::range_value_t<Range>>;



template <typename Fn, typename T>
concept NearestNeighborFn = 
    std::floating_point<T> &&
    std::invocable<Fn,T,T,T,T,T> &&
    std::same_as<T,std::invoke_result_t<Fn,T,T,T,T,T>>;



template <typename Fn, typename T>
concept ReductionOperation =
    std::copy_constructible<Fn> &&
    std::invocable<Fn,T,T> &&
    std::convertible_to<T,std::invoke_result_t<Fn,T,T>>;



template <typename Fn, typename T>
concept PairReductionOperation =
    std::copy_constructible<Fn> &&
    std::invocable<Fn,T,T,T> &&
    std::convertible_to<T,std::invoke_result_t<Fn,T,T,T>>;




//-----------------------------------------------------------------------------
// based on example from PR2300
void inclusive_scan (
    stdexec::scheduler auto sch,
    std::span<const double> input,
    std::span<double> output,
    double init,
    std::size_t tileCount)
{
    std::size_t const tileSize = (input.size() + tileCount - 1) / tileCount;

    std::vector<double> partials(tileCount + 1);
    partials[0] = init;

    auto task = stdexec::transfer_just(sch, std::move(partials))
    |   stdexec::bulk(tileCount,
        [=](std::size_t i, std::vector<double>&& part) {
            auto start = i * tileSize;
            auto end   = std::min(input.size(), (i + 1) * tileSize);
            part[i + 1] = *--std::inclusive_scan(begin(input) + start,
                                                    begin(input) + end,
                                                    begin(output) + start);
        })
    |   stdexec::then(
        [](std::vector<double>&& part) {
            std::inclusive_scan(begin(part), end(part),
                                begin(part));
            return part;
        })
    |   stdexec::bulk(tileCount,
        [=](std::size_t i, std::vector<double>&& part) {
            auto start = i * tileSize;
            auto end   = std::min(input.size(), (i + 1) * tileSize);
            std::for_each(output.begin() + start, output.begin() + end,
                [=] (double& e) { e = part[i] + e; });
        }) ;

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
void transform_matrix_nearest_neigbors (
    stdexec::scheduler auto sch,
    span2d<double> matrix,
    double border,
    std::size_t stripeCount,
    NearestNeighborFn<double> auto fn)
{
    auto stripeHeight = (matrix.nrows() + stripeCount - 1) / stripeCount;
    if (stripeHeight < 2) {
        stripeHeight = 2;
        stripeCount = (matrix.nrows() + 1) / 2;
    }

    auto compute_stripe = [=](std::size_t stripeIdx, span2d<double> m) {
        std::size_t rstart = stripeIdx * stripeHeight;
        std::size_t rend   = std::min(m.nrows(), (stripeIdx+1) * stripeHeight);
        bool const lastRow = rend == m.nrows();
        if (lastRow) --rend;

        if (rstart == 0) {
            m(0,0) = fn(        border,
                        border, m(0,0), m(0,1),
                                m(1,0) );

            for (std::size_t c = 1; c < m.ncols()-1; ++c) {
                m(0,c) = fn(          border,
                            m(0,c-1), m(0,c), m(0,c+1),
                                      m(1,c) );
            }
            auto const c = m.ncols() - 1;
            m(0,c) = fn(          border,
                        m(0,c-1), m(0,c), border,
                                  m(1,c) );
            rstart = 1;
        }
        for (std::size_t r = rstart; r < rend; ++r) {
            m(r,0) = fn(        m(r-1,0), 
                        border, m(r  ,0), m(r,1),
                                m(r+1,0) );

            for (std::size_t c = 1; c < m.ncols()-1; ++c) {
                m(r,c) = fn(          m(r-1,c), 
                            m(r,c-1), m(r  ,c), m(r,c+1),
                                      m(r+1,c) );
            }
            auto const c = m.ncols() - 1;
            m(r,c) = fn(          m(r-1,c), 
                        m(r,c-1), m(r  ,c), border,
                                  m(r+1,c) );
        }
        if (lastRow) {
            auto const r = m.nrows() - 1;
            m(r,0) = fn(        m(r-1,0),
                        border, m(r  ,0), m(r,1),
                                border );

            for (std::size_t c = 1; c < m.ncols()-1; ++c) {
                m(r,c) = fn(          m(r-1,c),
                            m(0,c-1), m(r  ,c), m(r,c+1),
                                      border );
            }
            auto const c = m.ncols() - 1;
            m(r,c) = fn(          m(r-1,c),
                        m(r,c-1), m(r  ,c), border,
                                  border );
        }
    };

    auto task = stdexec::transfer_just(sch, matrix)
        // even-numbered stripes
    |   stdexec::bulk(stripeCount,
        [=](std::size_t stripeIdx, span2d<double> m)
        {
            if (not (stripeIdx % 2)) { compute_stripe(stripeIdx, m); }
        })
        // odd-numbered stripes
    |   stdexec::bulk(stripeCount,
        [=](std::size_t stripeIdx, span2d<double> m)
        {
            if (stripeIdx % 2) { compute_stripe(stripeIdx, m); }
        });
        
    stdexec::sync_wait(std::move(task)).value();
}







//-----------------------------------------------------------------------------
template <typename InRange, typename Body>
requires 
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange> &&
    std::copy_constructible<Body> &&
    std::invocable<Body,std::ranges::range_value_t<InRange>>
void for_each (Execution_Context ctx, InRange&& input, Body body)
{
    using size_t_ = std::ranges::range_size_t<InRange>;

    auto const size        = std::ranges::size(input);
    auto const threadCount = std::min(size, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (size + threadCount - 1) / threadCount;
    auto const tileCount   = (size + tileSize - 1) / tileSize;

    auto task = 
        stdexec::transfer_just(ctx.get_scheduler(),
                               view_of(std::forward<InRange>(input))) 
    |   stdexec::bulk(tileCount,
        [=](std::size_t tileIdx, auto in)
        {
            auto const start = begin(in) + tileIdx * tileSize;
            auto const end   = begin(in) + std::min(size, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                body(*i);
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <typename GridExtents, typename Body>
requires 
    std::copy_constructible<Body>
void for_each_grid_index (Execution_Context ctx, GridExtents ext, Body body)
{
    using size_t_ = std::ranges::range_size_t<GridExtents>;

    // size of collapsed index range
    size_t_ size = 1;
    for (auto x : ext) { size *= static_cast<size_t_>(x); }
    
    auto const N = static_cast<int>(ext.size());
    auto const threadCount = std::min(size, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = static_cast<size_t_>((size + threadCount - 1) / threadCount);
    auto const tileCount   = (size + tileSize - 1) / tileSize;

    auto task = stdexec::schedule(ctx.get_scheduler())
    |   stdexec::bulk(tileCount, [=](size_t_ tileIdx)
        {
            // start/end of collapsed index range
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);
            if (start >= end) return;
            // compute start index
            GridExtents idx;
            for (int i = 0; i < N; ++i) { idx[i] = 0; }

            if (start > 0) {
                size_t_ mul[N];
                mul[0] = 1;
                for (int i = 0; i < N-1; ++i) {
                    mul[i+1] = mul[i] * ext[i]; 
                }
                auto offset = start;
                for (int i = N; i > 0; --i) {
                    if (offset >= mul[i-1]) {
                        idx[i-1] += offset / mul[i-1];
                        offset = offset % mul[i-1];
                        if (offset == 0) break;
                    }
                }
            }
            // execute body on local index subrange
            for (auto ci = start; ci < end; ++ci) {
                body(idx);
                // increment index
                for (int i = 0; i < N; ++i) {
                    ++idx[i];
                    if (idx[i] < ext[i]) break;
                    idx[i] = 0;
                }
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <typename OutRange, typename Generator>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         IndexToValueMapping<Generator,OutRange>
void generate_indexed (Execution_Context ctx, OutRange & output, Generator gen)
{
    using size_t_ = std::ranges::range_size_t<OutRange>;

    auto const outSize     = std::ranges::size(output);
    auto const threadCount = std::min(outSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (outSize + threadCount - 1) / threadCount;
    auto const tileCount   = (outSize + tileSize - 1) / tileSize;

    auto task =
        stdexec::transfer_just(ctx.get_scheduler(), view_of(output))
    |   stdexec::bulk(tileCount,
        [=](size_t_ tileIdx, auto out)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(outSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                out[i] = gen(i);
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <typename InRange, typename OutRange, typename Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
void transform (
    Execution_Context ctx, InRange const& input, OutRange & output, Transf fn)
{
    using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange>,
                                       std::ranges::range_size_t<OutRange>>;

    auto const inSize      = std::ranges::size(input);
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    auto task = 
        stdexec::transfer_just(ctx.get_scheduler(), 
                               view_of(input), view_of(output) )
    |   stdexec::bulk(tileCount,
        [=](std::size_t tileIdx, auto in, auto out)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                out[i] = fn(in[i]);
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <typename InRange, typename OutRange, typename Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
void transform_indexed (
    Execution_Context ctx, InRange const& input, OutRange & output, Transf fn)
{
    using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange>,
                                       std::ranges::range_size_t<OutRange>>;

    auto const inSize      = std::ranges::size(input);
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    auto task = 
        stdexec::transfer_just(ctx.get_scheduler(), 
                               view_of(input), view_of(output) )
    |   stdexec::bulk(tileCount,
        [=](size_t_ tileIdx, auto in, auto out)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                out[i] = fn(i,in[i]);
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <
    typename InRange1,
    typename InRange2,
    typename OutRange,
    typename Transf,
    typename Value1 = std::ranges::range_value_t<InRange1>,
    typename Value2 = std::ranges::range_value_t<InRange2>,
    typename OutValue = std::ranges::range_value_t<OutRange>
>
requires 
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::random_access_range<OutRange> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    std::ranges::sized_range<OutRange> &&
    std::copy_constructible<Transf> &&
    std::invocable<Transf,Value1,Value2> &&
    std::convertible_to<OutValue,std::invoke_result_t<Transf,Value1,Value2>>
void zip_transform (
    Execution_Context ctx,
    InRange1 const& input1,
    InRange2 const& input2,
    OutRange & output,
    Transf fn)
{
    using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange1>,
                                       std::ranges::range_size_t<InRange2>>;

    auto const inSize      = std::min(std::ranges::size(input1), std::ranges::size(input2));
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    auto task = 
        stdexec::transfer_just(ctx.get_scheduler(),
                               view_of(input1), view_of(input2),
                               view_of(output) )
    |   stdexec::bulk(tileCount,
        [=](size_t_ tileIdx, auto in1, auto in2, auto out)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i != end; ++i) {
                out[i] = fn(in1[i], in2[i]);
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <
    typename InRange1,
    typename InRange2,
    typename OutRange,
    typename Transf,
    typename Value1 = std::ranges::range_value_t<InRange1>,
    typename Value2 = std::ranges::range_value_t<InRange2>,
    typename OutValue = std::ranges::range_value_t<OutRange>
>
requires 
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::random_access_range<OutRange> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    std::ranges::sized_range<OutRange> &&
    std::copy_constructible<Transf> &&
    std::invocable<Transf,Value1,Value2,OutValue&>
void zip_transform (
    Execution_Context ctx,
    InRange1 const& input1,
    InRange2 const& input2,
    OutRange & output,
    Transf fn)
{
    using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange1>,
                                       std::ranges::range_size_t<InRange2>>;

    auto const inSize      = std::min(std::ranges::size(input1), std::ranges::size(input2));
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    auto task =
        stdexec::transfer_just(ctx.get_scheduler(),
                               view_of(input1), view_of(input2), view_of(output))
    |   stdexec::bulk(tileCount,
        [=](size_t_ tileIdx, auto in1, auto in2, auto out)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);
            
            for (auto i = start; i < end; ++i) {
                fn(in1[i], in2[i], out[i]);
            }
        });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
#ifndef USE_GPU

template <typename InRange, typename Result, typename ReductionOp>
requires
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange> &&
    ReductionOperation<ReductionOp,Result>
[[nodiscard]] Result 
reduce (
    Execution_Context ctx,
    InRange const& input, Result initValue, ReductionOp redOp)
{
    using size_t_ = std::ranges::range_size_t<InRange>;

    auto const inSize      = std::ranges::size(input);
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    std::vector<Result> partials (tileCount, Result(0));

    auto task = stdexec::transfer_just(ctx.get_scheduler(),
                                       view_of(input), view_of(partials) )
    |   stdexec::bulk(tileCount,
        [=](std::size_t tileIdx, auto in, auto parts)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                parts[tileIdx] = redOp(parts[tileIdx], in[i]);
            }
        })
    |   stdexec::then([=](auto, auto parts)
        {
            return std::reduce(begin(parts), end(parts), initValue);
        });

    return std::get<0>(stdexec::sync_wait(std::move(task)).value());
}




//-----------------------------------------------------------------------------
template <
    typename InRange1,
    typename InRange2, 
    typename Result,
    typename ReductionOp
>
requires
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    PairReductionOperation<ReductionOp,Result>
[[nodiscard]] Result 
zip_reduce (
    Execution_Context ctx,
    InRange1 const& input1,
    InRange2 const& input2,
    Result initValue,
    ReductionOp redOp)
{
    using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange1>,
                                       std::ranges::range_size_t<InRange2>>;

    auto const inSize      = std::min(std::ranges::size(input1), std::ranges::size(input2));
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    std::vector<Result> partials (tileCount, Result(0));

    auto task = stdexec::transfer_just(ctx.get_scheduler(),
                                       view_of(input1), view_of(input2),
                                       view_of(partials))
    |   stdexec::bulk(tileCount,
        [=](std::size_t tileIdx, auto in1, auto in2, auto parts)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                parts[tileIdx] = redOp(parts[tileIdx], in1[i], in2[i]);
            }
        })
    |   stdexec::then([=](auto, auto, auto parts)
        {
            return std::reduce(begin(parts), end(parts), initValue);
        });

    return std::get<0>(stdexec::sync_wait(std::move(task)).value());
}




//-----------------------------------------------------------------------------
template <
    typename InRange1,
    typename InRange2, 
    typename Result,
    typename ReductionOp
>
requires
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    std::copy_constructible<ReductionOp> &&
    std::invocable<ReductionOp,Result,Result> &&
    std::convertible_to<Result,std::invoke_result_t<ReductionOp,Result,Result>>
[[nodiscard]] Result 
zip_reduce_sum (
    Execution_Context ctx,
    InRange1 const& input1,
    InRange2 const& input2,
    Result initValue,
    ReductionOp redOp)
{
    using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange1>,
                                       std::ranges::range_size_t<InRange2>>;

    auto const inSize      = std::min(std::ranges::size(input1), std::ranges::size(input2));
    auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
    auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    auto const tileCount   = (inSize + tileSize - 1) / tileSize;

    std::vector<Result> partials (tileCount, Result(0));

    auto task = stdexec::transfer_just(ctx.get_scheduler(),
                                       view_of(input1), view_of(input2),
                                       view_of(partials))
    |   stdexec::bulk(tileCount,
        [=](std::size_t tileIdx, auto in1, auto in2, auto parts)
        {
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

            for (auto i = start; i < end; ++i) {
                parts[tileIdx] += redOp(in1[i], in2[i]);
            }
        })
    |   stdexec::then([=](auto, auto, auto parts)
        {
            return std::reduce(begin(parts), end(parts), initValue);
        });

    return std::get<0>(stdexec::sync_wait(std::move(task)).value());
}


#else


//-----------------------------------------------------------------------------
template <typename InRange, typename ResultValue, typename ReductionOp>
requires
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange> &&
    ReductionOperation<ReductionOp,ResultValue>
[[nodiscard]] ResultValue 
reduce (
    Execution_Context ctx,
    InRange const& input, ResultValue initValue, ReductionOp redOp)
{
    auto task = 
          stdexec::transfer_just(ctx.get_scheduler(), view_of(input))
        | nvexec::reduce(initValue, cub::Min{});

    return std::get<0>(stdexec::sync_wait(std::move(task)).value());
}




//-----------------------------------------------------------------------------
template <
    typename InRange1,
    typename InRange2, 
    typename Result,
    typename ReductionOp
>
requires
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    PairReductionOperation<ReductionOp,Result>
[[nodiscard]] Result 
zip_reduce (
    Execution_Context,
    InRange1 const&,
    InRange2 const&,
    Result,
    ReductionOp)
{
    throw "GPU-based zip_reduce not implemented yet";
}




//-----------------------------------------------------------------------------
template <
    typename InRange1,
    typename InRange2, 
    typename Result,
    typename ReductionOp
>
requires
    std::ranges::random_access_range<InRange1> &&
    std::ranges::random_access_range<InRange2> &&
    std::ranges::sized_range<InRange1> &&
    std::ranges::sized_range<InRange2> &&
    std::copy_constructible<ReductionOp> &&
    std::invocable<ReductionOp,Result,Result> &&
    std::convertible_to<Result,std::invoke_result_t<ReductionOp,Result,Result>>
[[nodiscard]] Result 
zip_reduce_sum (
    Execution_Context,
    InRange1 const&,
    InRange2 const&,
    Result,
    ReductionOp)
{
    throw "GPU-based zip_reduce_sum not implemented yet";
}


#endif  // USE_GPU

#endif

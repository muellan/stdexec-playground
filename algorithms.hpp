#ifndef STDEXEC_ALGORITHMS_HPP
#define STDEXEC_ALGORITHMS_HPP


#include "acceleration.hpp"
#include "span2d.hpp"
#include "indices.hpp"
#include "storage.hpp"

#include <stdexec/execution.hpp>

#include <exec/static_thread_pool.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/variant_sender.hpp>

#ifdef USE_GPU
#include <nvexec/multi_gpu_context.cuh>
#include <nvexec/stream_context.cuh>

#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <concepts>
#include <execution>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace ex {


//-----------------------------------------------------------------------------
namespace detail {

template <class R>
concept viewable_range =
    std::ranges::random_access_range<R>;
    // requires (R r) {
    //     {*std::ranges::data(r)} -> std::convertible_to<std::ranges::range_value_t<R>>;
    //     {std::ranges::size(r)} -> std::convertible_to<std::ranges::range_size_t<R>>;
    // };
    // std::ranges::contiguous_range<R> &&
    // std::ranges::sized_range<R>;


//-----------------------------------------------------------------------------
struct range_size_t
{
    template <std::ranges::sized_range R>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    auto constexpr operator () (R const& r) const noexcept
    {
        return std::ranges::size(r);
    }

    template <std::ranges::sized_range R1,
              std::ranges::sized_range R2,
              std::ranges::sized_range... Rs>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    auto constexpr operator () (
        R1 const& r1, R2 const& r2, Rs const&... rs) const noexcept
    {
        return std::min(std::ranges::size(r1), operator()(r2, rs...));
    }
};

inline constexpr range_size_t range_size {};




//-----------------------------------------------------------------------------
template <detail::viewable_range Range>
[[nodiscard]] 
HOSTDEVICEQUALIFIER
constexpr auto
make_view (Range&& r) noexcept
{
    return std::span{std::ranges::data(std::forward<Range>(r)),
                     std::ranges::size(std::forward<Range>(r))};
}


template <class T>
[[nodiscard]] 
HOSTDEVICEQUALIFIER
constexpr auto
make_view (std::span<T> s) noexcept
{
    return s;
}


#ifdef USE_GPU

template <class T>
[[nodiscard]] constexpr auto
HOSTDEVICEQUALIFIER
make_view (thrust::universal_vector<T>& v) noexcept
{
    return std::span<T>{thrust::raw_pointer_cast(v.data()), v.size()};
}

template <class T>
[[nodiscard]] constexpr auto
HOSTDEVICEQUALIFIER
make_view (thrust::universal_vector<T> const& v) noexcept {
    return std::span<T const>{thrust::raw_pointer_cast(v.data()), v.size()};
}

#endif



struct view_of_t
{
    template <detail::viewable_range Range>
    [[nodiscard]] constexpr auto
    HOSTDEVICEQUALIFIER
    operator () (Range&& r) const noexcept { return make_view(r); }
};


struct const_view_of_t
{
    template <detail::viewable_range Range>
    [[nodiscard]] constexpr auto
    HOSTDEVICEQUALIFIER
    operator () (Range const& r) const noexcept { return make_view(r); }
};

}  // namespace detail


inline constexpr detail::view_of_t       view_of {};
inline constexpr detail::const_view_of_t const_view_of {};




//-----------------------------------------------------------------------------
template <class C>
concept execution_context =
    std::same_as<std::remove_cvref_t<C>,ex::acceleration_context> ||
#ifdef USE_GPU
    std::same_as<std::remove_cvref_t<C>,nvexec::stream_context> ||
    std::same_as<std::remove_cvref_t<C>,nvexec::multi_gpu_stream_context> ||
    std::same_as<std::remove_cvref_t<C>,
        std::remove_cvref_t<decltype(std::declval<nvexec::stream_context>().get_scheduler())>> ||
    std::same_as<std::remove_cvref_t<C>,
        std::remove_cvref_t<decltype(std::declval<nvexec::multi_gpu_stream_context>().get_scheduler())>> ||
#endif
    std::same_as<std::remove_cvref_t<C>,exec::single_thread_context> ||
    std::same_as<std::remove_cvref_t<C>,exec::static_thread_pool> ||
    std::same_as<std::remove_cvref_t<C>,exec::static_thread_pool::scheduler> ||
    std::same_as<std::remove_cvref_t<C>,
        std::remove_cvref_t<decltype(std::declval<exec::single_thread_context>().get_scheduler())>>;






//-----------------------------------------------------------------------------
/// @brief  used to inidicate dispatch to CPU-only implementations
struct cpu_tag
{
    std::int64_t threads = 1;
};


/// @brief  used to inidicate dispatch to GPU-only implementations
struct gpu_tag
{
    int devices = 1;
};


template <class T>
concept compute_tag =
    std::same_as<cpu_tag,T> ||
    std::same_as<gpu_tag,T>;




//-----------------------------------------------------------------------------
// NOTE:
// there's some code duplication in the 'operator()' implementations
// this is mostly due to problems arising from nesting 'stdexec::let_value'


//-----------------------------------------------------------------------------
/**
 *  @brief   starts an async sender chain on a scheduler
 *           passes along compute resource information (CPU/GPU/#threads/...)
 */
class start_on_t
{
    using static_thread_pool_scheduler = exec::static_thread_pool::scheduler;

    using single_thread_scheduler = std::remove_cvref_t<
        decltype(std::declval<exec::single_thread_context>().get_scheduler())>;

public:
    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (exec::single_thread_context& ctx, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                ctx.get_scheduler(),
                cpu_tag{ .threads = 1 },
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (single_thread_scheduler sched, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                sched,
                cpu_tag{ .threads = 1 },
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (exec::static_thread_pool& pool, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                pool.get_scheduler(),
                cpu_tag{ .threads = pool.available_parallelism() },
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (static_thread_pool_scheduler sched, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                sched,
                cpu_tag{ // TODO get thread count from associated thread pool
                    .threads = std::thread::hardware_concurrency()
                },
                view_of((Ranges&&)(rs))... );
    }


    template <stdexec::scheduler Scheduler, class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (Scheduler&& sched, cpu_tag info, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                        (Scheduler&&)sched, info, view_of((Ranges&&)(rs))... );
    }


#ifdef USE_GPU
private:
    using stream_context_scheduler = std::remove_cvref_t<
        decltype(std::declval<nvexec::stream_context>().get_scheduler())>;

    using multi_gpu_stream_context_scheduler = std::remove_cvref_t<
        decltype(std::declval<nvexec::multi_gpu_stream_context>().get_scheduler())>;


    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    static gpu_tag make_multi_gpu_info ()
    {
        gpu_tag tag { .devices = 1, };
        cudaGetDeviceCount(&tag.devices);
        check_cuda_err();
        return tag;
    }


    HOSTDEVICEQUALIFIER
    static void check_cuda_err ()
    {
        cudaError_t err;
        if ((err = cudaGetLastError()) != cudaSuccess) {
            throw std::runtime_error{
                std::string{"CUDA error: "}
                + std::string{cudaGetErrorString(err)}
                + std::string{" : "}
                + std::string{__FILE__}
                + std::string{", line "}
                + std::to_string(__LINE__)
            };
        }
    }


public:
    template <stdexec::scheduler Scheduler, class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (Scheduler&& sched, gpu_tag info, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                        (Scheduler&&)sched, info, view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (nvexec::stream_context& ctx, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                ctx.get_scheduler(),
                gpu_tag{},
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (stream_context_scheduler sched, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                sched, gpu_tag{},
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (nvexec::multi_gpu_stream_context& ctx, Ranges&& ... rs) const
    {

        return stdexec::transfer_just(
                ctx.get_scheduler(),
                make_multi_gpu_info(),
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (multi_gpu_stream_context_scheduler sched,
                      Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                sched,
                make_multi_gpu_info(),
                view_of((Ranges&&)(rs))... );
    }


    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (acceleration_context ctx, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                ctx.get_scheduler(), gpu_tag{}, view_of((Ranges&&)(rs))... );
    }


#else
    template <class... Ranges>
    requires (detail::viewable_range<Ranges> && ...)
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    stdexec::sender
    auto operator () (acceleration_context ctx, Ranges&& ... rs) const
    {
        return stdexec::transfer_just(
                ctx.get_scheduler(),
                cpu_tag{ .threads = ctx.shape().threads },
                view_of((Ranges&&)(rs))... );
    }

#endif  // USE_GPU
};

inline constexpr start_on_t  start_on {};






//-----------------------------------------------------------------------------
/**
 *  @brief  extracts value from chain
 */

// TODO replace specialization with general TMP solution
template <int index>
struct return_value_t;

template <>
struct return_value_t<0>
{
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    auto operator () () const {
        return stdexec::then(
            [](compute_tag auto, auto&& r, auto&&...){
                return (decltype(r)&&)(r);
            });
    }
};

template <>
struct return_value_t<1>
{
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    auto operator () () const {
        return stdexec::then(
            [](compute_tag auto, auto&&, auto&& r, auto&&...){
                return (decltype(r)&&)(r);
            });
    }
};

template <>
struct return_value_t<2>
{
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    auto operator () () const {
        return stdexec::then(
            [](compute_tag auto, auto&&, auto&&, auto&& r, auto&&...){
                return (decltype(r)&&)(r);
            });
    }
};

inline constexpr return_value_t<0> return_result {};
inline constexpr return_value_t<0> return_first {};
inline constexpr return_value_t<1> return_second {};
inline constexpr return_value_t<2> return_third {};






//-----------------------------------------------------------------------------
/**
 *  @brief
 */
struct apply_t
{
    template <class Work>
    requires std::copy_constructible<Work>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Work&& work) const
    {
        return stdexec::let_value(
            [work=(Work&&)work](auto info, auto&&... values)
            {
                work((decltype(values)&&)(values)...);

                return stdexec::just(info, (decltype(values)&&)(values)...);
            });
    }
};

inline constexpr apply_t apply {};






//-----------------------------------------------------------------------------
/**
 *  @brief
 */

// TODO replace specialization with general TMP solution
template <int index>
struct apply_to_value_t;

template <>
struct apply_to_value_t<0>
{
    template <class Work>
    requires std::copy_constructible<Work>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Work&& work) const
    {
        return stdexec::let_value(
            [work=(Work&&)work]
            (compute_tag auto tag, auto&& v0, auto&&... vs)
            {
                work((decltype(v0)&&)(v0));
                return stdexec::just(tag,
                                     (decltype(v0)&&)(v0),
                                     (decltype(vs)&&)(vs)...);
            });
    }
};

template <>
struct apply_to_value_t<1>
{
    template <class Work>
    requires std::copy_constructible<Work>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Work&& work) const
    {
        return stdexec::let_value(
            [work=(Work&&)work]
            (compute_tag auto tag, auto&& v0, auto&& v1, auto&&... vs)
            {
                work((decltype(v1)&&)(v1));
                return stdexec::just(tag,
                                     (decltype(v0)&&)(v0),
                                     (decltype(v1)&&)(v1),
                                     (decltype(vs)&&)(vs)...);
            });
    }
};

template <>
struct apply_to_value_t<2>
{
    template <class Work>
    requires std::copy_constructible<Work>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Work&& work) const
    {
        return stdexec::let_value(
            [work=(Work&&)work]
            (compute_tag auto tag, auto&& v0, auto&& v1, auto&& v2, auto&&... vs)
            {
                work((decltype(v2)&&)(v2));
                return stdexec::just(tag,
                                     (decltype(v0)&&)(v0),
                                     (decltype(v1)&&)(v1),
                                     (decltype(v2)&&)(v2),
                                     (decltype(vs)&&)(vs)...);
            });
    }
};

inline constexpr apply_to_value_t<0> apply_to_first {};
inline constexpr apply_to_value_t<1> apply_to_second {};
inline constexpr apply_to_value_t<2> apply_to_third {};






//-----------------------------------------------------------------------------
/**
 * @brief sets values for each element of an input range
 */
struct fill_t
{
    // synchronous variant
    template <execution_context Context, class OutRange, class Value>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Value>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (Context&& ctx, OutRange& output, Value&& value) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(output, (Value&&)value);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant
    template <class OutRange, class Value>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Value>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (OutRange& output, Value&& value) const
    {
        return stdexec::let_value(
            [outview=view_of(output), val=(Value&&)value] (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outview),
                    std::ranges::size(outview),
                    [=](std::size_t i, compute_tag auto, auto out){
                        out[i] = val;
                    });
            });
    }

    // pipable, asynchronous variant
    template <class Value>
    requires std::copy_constructible<Value>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Value&& value) const
    {
        return stdexec::let_value(
            [val=(Value&&)value](compute_tag auto tag, auto output)
            {
                return stdexec::bulk(
                    stdexec::just(tag, output),
                    std::ranges::size(output),
                    [=](std::size_t i, compute_tag auto, auto out){
                        out[i] = val;
                    });
            });
    }
};

inline constexpr fill_t fill {};






//-----------------------------------------------------------------------------
/**
 * @brief sets values for each element of an input range
 *        using a generator function(object)
 */
struct generate_t
{
    // synchronous variant
    template <execution_context Context, class OutRange, class Generator>
    requires detail::viewable_range<OutRange>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (Context&& ctx, OutRange& output, Generator&& gen) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(output, (Generator&&)gen);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant, uses range from sender chain
    template <class Generator>
    requires std::copy_constructible<Generator>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Generator&& gen) const
    {
        return stdexec::let_value(
            [gen=(Generator&&)gen](compute_tag auto tag, auto output)
            {
                return stdexec::bulk(
                    stdexec::just(tag, output),
                    std::ranges::size(output),
                    [=](std::size_t i, auto, auto out) 
                    {
                        out[i] = gen();
                    });
            });
    }


    // pipable, asynchronous variant, uses supplied range
    template <class OutRange, class Generator>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Generator>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (OutRange& output, Generator&& gen) const
    {
        return stdexec::let_value(
            [out=view_of(output), gen=(Generator&&)gen]
            (compute_tag auto tag) {
                return stdexec::bulk(
                    stdexec::just(tag, out),
                    std::ranges::size(out),
                    [=](std::size_t i, auto, auto out) 
                    {
                        out[i] = gen();
                    });
            });
    }
};

inline constexpr generate_t generate {};






//-----------------------------------------------------------------------------
/**
 * @brief sets values for each element of an input range
 *        using an index-based generator
 */
struct generate_indexed_t
{
    // synchronous variant
    template <execution_context Context, class OutRange, class Generator>
    requires
        detail::viewable_range<OutRange> &&
        std::invocable<Generator,std::size_t>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (Context&& ctx, OutRange& output, Generator&& gen) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(output, (Generator&&)gen);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant, uses range from sender chain
    template <class Generator>
    requires
        std::copy_constructible<Generator> &&
        std::invocable<Generator,std::size_t>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Generator&& gen) const
    {
        return stdexec::let_value(
            [gen=(Generator&&)gen]
            (compute_tag auto tag, auto output)
            {
                return stdexec::bulk(
                    stdexec::just(tag, output),
                    std::ranges::size(output),
                    [=](std::size_t i, auto, auto out) {
                        out[i] = gen(i);
                    });
            });
    }


    // pipable, asynchronous variant, uses supplied range
    template <class OutRange, class Generator>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Generator> &&
        std::invocable<Generator,std::size_t>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (OutRange& output, Generator&& gen) const
    {
        return stdexec::let_value(
            [out=view_of(output), gen=(Generator&&)gen]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, out),
                    std::ranges::size(out),
                    [=](std::size_t i, auto, auto out) {
                        out[i] = gen(i);
                    });
            });
    }
};

inline constexpr generate_indexed_t generate_indexed {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct for_each_t
{
    // synchronous variant
    template <execution_context Context, class InRange, class Body>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Body>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx, InRange&& input, Body&& body) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()((InRange&&)(input), (Body&&)body);

        stdexec::sync_wait(std::move(task)).value();
    }


    template <execution_context Context, class InRange1, class InRange2, class Body>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<Body>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx, InRange1&& input1, InRange2&& input2, Body&& body) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()((InRange1&&)(input1), (InRange2&&)(input2),
                               (Body&&)body);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant
    template <class Body>
    requires std::copy_constructible<Body>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Body&& body) const
    {
        return stdexec::let_value(
            [body=(Body&&)body](compute_tag auto tag, auto... inputs)
            {
                return stdexec::bulk(
                    stdexec::just(tag, inputs...),
                    detail::range_size(inputs...),
                    [=](std::size_t i, compute_tag auto, auto... inputs)
                    {
                        body(inputs[i]...);
                    });
            });
    }


    // pipable, asynchronous variant
    template <class InRange, class Body>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Body>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange&& input, Body&& body) const
    {
        return stdexec::let_value(
            [inView=view_of(input), body=(Body&&)body]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, inView),
                    detail::range_size(inView),
                    [=](std::size_t i, auto, auto in)
                    {
                        body(in[i]);
                    });
            });
    }


    template <class InRange1, class InRange2, class Body>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<Body>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange1&& input1, InRange2&& input2, Body&& body) const
    {
        return stdexec::let_value(
            [inView1=view_of(input1), inView2=view_of(input2), body=(Body&&)body]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, inView1, inView2),
                    detail::range_size(inView1, inView2),
                    [=](std::size_t i, auto, auto in1, auto in2)
                    {
                        body(in1[i], in2[i]);
                    });
            });
    }
};

inline constexpr for_each_t for_each {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct for_each_indexed_t
{
    // synchronous variant
    template <execution_context Context, class InRange, class Body>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Body>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx, InRange&& input, Body&& body) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()((InRange&&)(input), (Body&&)body);

        stdexec::sync_wait(std::move(task)).value();
    }


    template <execution_context Context, class InRange1, class InRange2, class Body>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<Body>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx, InRange1&& input1, InRange2&& input2, Body&& body) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()((InRange1&&)(input1), (InRange2&&)(input2),
                               (Body&&)body);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant
    template <class Body>
    requires std::copy_constructible<Body>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Body&& body) const
    {
        return stdexec::let_value(
            [body=(Body&&)body](compute_tag auto tag, auto... inputs)
            {
                return stdexec::bulk(
                    stdexec::just(tag, inputs...),
                    detail::range_size(inputs...),
                    [=](std::size_t i, compute_tag auto, auto... inputs)
                    {
                        body(i, inputs[i]...);
                    });
            });
    }


    // pipable, asynchronous variant
    template <class InRange, class Body>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Body>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange&& input, Body&& body) const
    {
        return stdexec::let_value(
            [inView=view_of(input), body=(Body&&)body]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, inView),
                    detail::range_size(inView),
                    [=](std::size_t i, auto, auto in)
                    {
                        body(i, in[i]);
                    });
            });
    }


    template <class InRange1, class InRange2, class Body>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<Body>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange1&& input1, InRange2&& input2, Body&& body) const
    {
        return stdexec::let_value(
            [inView1=view_of(input1), inView2=view_of(input2), body=(Body&&)body]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, inView1, inView2),
                    detail::range_size(inView1, inView2),
                    [=](std::size_t i, auto, auto in1, auto in2)
                    {
                        body(i, in1[i], in2[i]);
                    });
            });
    }
};

inline constexpr for_each_indexed_t for_each_indexed {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct transform_t
{
    // synchronous variant
    template <execution_context Context, class InRange, class OutRange, class Transf>
    requires
        detail::viewable_range<InRange> &&
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx, InRange const& input, OutRange & output, Transf&& fn) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(input, output, (Transf&&)fn);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant; uses external custom output range
    template <class OutRange, class Transf>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (OutRange & output, Transf&& transf) const
    {
        return stdexec::let_value(
            [outView = view_of(output), fn = (Transf&&)transf]
            (compute_tag auto tag, auto in)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outView),
                    detail::range_size(in, outView),
                    [=](std::size_t i, compute_tag auto, auto out)
                    {
                        out[i] = fn(in[i]);
                    });
            });
    }


    // pipable, asynchronous variant; uses external custom output range
    template <class InRange, class OutRange, class Transf>
    requires
        detail::viewable_range<InRange> &&
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange const& input, OutRange & output, Transf&& transf) const
    {
        return stdexec::let_value(
            [in = view_of(input), outView = view_of(output), fn = (Transf&&)transf]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outView),
                    detail::range_size(in, outView),
                    [=](std::size_t i, compute_tag auto, auto out)
                    {
                        out[i] = fn(in[i]);
                    });
            });
    }
};

inline constexpr transform_t transform {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct transform_indexed_t
{

    // synchronous variant
    template <execution_context Context, class InRange, class OutRange, class Transf>
    requires
        detail::viewable_range<InRange> &&
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    constexpr
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx, InRange const& input, OutRange & output, Transf&& fn) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(input, output, (Transf&&)fn);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable, asynchronous variant; uses external custom output range
    template <class OutRange, class Transf>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (OutRange & output, Transf&& transf) const
    {
        return stdexec::let_value(
            [outView = view_of(output), fn = (Transf&&)transf]
            (compute_tag auto tag, auto in)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outView),
                    detail::range_size(in, outView),
                    [=](std::size_t i, compute_tag auto, auto out)
                    {
                        out[i] = fn(i, in[i]);
                    });
            });
    }


    // pipable, asynchronous variant; uses external custom output range
    template <class InRange, class OutRange, class Transf>
    requires
        detail::viewable_range<InRange> &&
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange const& input, OutRange & output, Transf&& transf) const
    {
        return stdexec::let_value(
            [in = view_of(input), outView = view_of(output), fn = (Transf&&)transf]
            (compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outView),
                    detail::range_size(in, outView),
                    [=](std::size_t i, compute_tag auto, auto out)
                    {
                        out[i] = fn(i, in[i]);
                    });
            });
    }
};

inline constexpr transform_indexed_t transform_indexed {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct zip_transform_t
{
    // synchronous variant
    template <
        execution_context Context,
        class InRange1, class InRange2, class OutRange,
        class Transf
    >
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    HOSTDEVICEQUALIFIER
    void operator () (
        Context&& ctx,
        InRange1 const& input1,
        InRange2 const& input2,
        OutRange & output,
        Transf&& fn) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(input1, input2, output, (Transf&&)fn);

        stdexec::sync_wait(std::move(task)).value();
    }


    // pipable variant, for cases whithout ranges in sender chain
    // uses external range for output
    template <class InRange1, class InRange2, class OutRange, class Transf>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (
        InRange1 const& input1, InRange2 const& input2,
        OutRange & output, Transf&& transf) const
    {
        return stdexec::let_value(
            [in1=view_of(input1), in2=view_of(input2), outView=view_of(output),
             fn=(Transf&&)transf](compute_tag auto tag)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outView),
                    detail::range_size(in1,in2,outView),
                    [=](std::size_t i, compute_tag auto, auto out)
                    {
                        out[i] = fn(in1[i], in2[i]);
                    });
            });
    }


    // pipable variant, for cases with 2 range in sender chain and
    // external output storage
    template <class OutRange, class Transf>
    requires
        detail::viewable_range<OutRange> &&
        std::copy_constructible<Transf>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (OutRange & output, Transf&& transf) const
    {
        return stdexec::let_value(
            [outView=view_of(output), fn=(Transf&&)transf]
            (compute_tag auto tag, auto in1, auto in2)
            {
                return stdexec::bulk(
                    stdexec::just(tag, outView),
                    detail::range_size(in1,in2,outView),
                    [=](std::size_t i, compute_tag auto, auto out)
                    {
                        out[i] = fn(in1[i], in2[i]);
                    });
            });

    }
};

inline constexpr zip_transform_t zip_transform {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct reduce_t
{
#ifdef USE_GPU

    template <class InRange, class Result, class ReductionOp>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Result> &&
        std::copy_constructible<ReductionOp>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange const& input, Result init, ReductionOp redOp) const
    {
        return nvexec::reduce(view_of(input), init, redOp);
    }


    template <class InRange, class Result>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Result>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange const& input, Result init) const
    {
        return nvexec::reduce(view_of(input), init);
    }


    template <class InRange>
    requires
        detail::viewable_range<InRange>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (InRange const& input) const
    {
        using result_t = std::ranges::range_value_t<InRange>;
        return nvexec::reduce(view_of(input), result_t{});
    }
  
  
    template <class Result, class ReductionOp>
    requires
        std::copy_constructible<Result> &&
        std::copy_constructible<ReductionOp>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Result init, ReductionOp redOp) const
    {
        return stdexec::let_value([=](gpu_tag, auto inView){
            return nvexec::reduce(stdexec::just(inView), init, redOp);
        });
    }
    
   
    template <class Result>
    requires
        std::copy_constructible<Result>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Result init) const
    {
        return stdexec::let_value([=](gpu_tag, auto inView){
            return nvexec::reduce(stdexec::just(inView), init);
        });
    }


    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    auto operator () () const
    {
        return stdexec::let_value([=](gpu_tag, auto inView){
            using result_t = std::ranges::range_value_t<decltype(inView)>;
            return nvexec::reduce(stdexec::just(inView), result_t{});
        });
    }


#else

    // template <class InRange, class Result, class ReductionOp>
    // requires
    //     detail::viewable_range<InRange> &&
    //     std::copy_constructible<Result> &&
    //     std::copy_constructible<ReductionOp>
    // [[nodiscard]]
    // constexpr
    // auto operator () (
    //     InRange const& input,
    //     Result init = std::ranges::range_value_t<InRange>{},
    //     ReductionOp redOp = std::plus<>{}) const
    // {
    //     return stdexec::let_value([=,inView=view_of(input)](cpu_tag cpu) {
    //             auto const inSize      = std::ranges::size(inView);
    //             auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
    //             auto const tileSize    = (inSize + threadCount - 1) / threadCount;
    //             auto const tileCount   = (inSize + tileSize - 1) / tileSize;
    //
    //             std::vector<Result> partials (tileCount);
    //
    //             return stdexec::bulk(
    //                 stdexec::just(cpu, inView, std::move(partials)),
    //                 inSize,
    //                 [=](std::size_t tileIdx, cpu_tag, auto in, auto&& parts)
    //                 {
    //                     using value_t = std::ranges::range_value_t<decltype(in)>;
    //
    //                     auto const start = tileIdx * tileSize;
    //                     auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);
    //
    //                     parts[tileIdx] = std::reduce(
    //                                         std::ranges::begin(in) + start,
    //                                         std::ranges::begin(in) + end,
    //                                         value_t{}, redOp);
    //                 })
    //             |   stdexec::let_value([=](cpu_tag, auto, auto&& parts)
    //                 {
    //                     auto result = std::reduce(begin(parts), end(parts), init, redOp);
    //                     return stdexec::just(std::move(result));
    //                 });
    //         });
    // }


    template <class Result, class ReductionOp>
    requires
        std::copy_constructible<Result> &&
        std::copy_constructible<ReductionOp>
    [[nodiscard]]
    constexpr
    auto operator () (Result init, ReductionOp redOp) const
    {
        return stdexec::let_value([=](cpu_tag cpu, auto inView) {
                auto const inSize      = std::ranges::size(inView);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                std::vector<Result> partials (tileCount);

                return stdexec::bulk(
                    stdexec::just(cpu, inView, std::move(partials)),
                    inSize,
                    [=](std::size_t tileIdx, cpu_tag, auto in, auto&& parts)
                    {
                        using value_t = std::ranges::range_value_t<decltype(in)>;

                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        if (start < inSize) {
                            parts[tileIdx] = std::reduce(
                                                std::ranges::begin(in) + start,
                                                std::ranges::begin(in) + end,
                                                value_t{}, redOp);
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts), init, redOp);
                        return stdexec::just(std::move(result));
                    });
            });
    }


    template <class Result>
    requires
        std::copy_constructible<Result>
    [[nodiscard]]
    constexpr
    auto operator () (Result init) const
    {
        return stdexec::let_value([=](cpu_tag cpu, auto inView) {
                auto const inSize      = std::ranges::size(inView);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                std::vector<Result> partials (tileCount);

                return stdexec::bulk(
                    stdexec::just(cpu, inView, std::move(partials)),
                    inSize,
                    [=](std::size_t tileIdx, cpu_tag, auto in, auto&& parts)
                    {
                        using value_t = std::ranges::range_value_t<decltype(in)>;

                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        if (start < inSize) {
                            parts[tileIdx] = std::reduce(
                                                std::ranges::begin(in) + start,
                                                std::ranges::begin(in) + end,
                                                value_t{});
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts), init);
                        return stdexec::just(std::move(result));
                    });
            });
    }


    [[nodiscard]]
    auto operator () () const
    {
        return stdexec::let_value([=](cpu_tag cpu, auto inView) {
                auto const inSize      = std::ranges::size(inView);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using value_t = std::ranges::range_value_t<decltype(inView)>;
                std::vector<value_t> partials (tileCount);

                return stdexec::bulk(
                    stdexec::just(cpu, inView, std::move(partials)),
                    inSize,
                    [=](std::size_t tileIdx, cpu_tag, auto in, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        if (start < inSize) {
                            parts[tileIdx] = std::reduce(
                                                std::ranges::begin(in) + start,
                                                std::ranges::begin(in) + end,
                                                value_t{});
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts));
                        return stdexec::just(std::move(result));
                    });
            });
    }

#endif

    // synchronous variant; uses initial value & custom reduction operation
    template <execution_context Context, class InRange, class Result, class ReductionOp>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<ReductionOp>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr Result
    operator () (
        Context&& ctx, InRange const& input, Result init, ReductionOp redOp) const
    {
        auto task = start_on((Context&&)ctx, input)
                  | operator()(init, redOp);

        return std::get<0>(stdexec::sync_wait(std::move(task)).value());
    }


    // synchronous variant; uses inital value
    template <execution_context Context, class InRange, class Result>
    requires detail::viewable_range<InRange>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    Result operator () (
        Context&& ctx, InRange const& input, Result init) const
    {
        auto task = start_on((Context&&)ctx, input)
                  | operator()(init);

        return std::get<0>(stdexec::sync_wait(std::move(task)).value());
    }


    // synchronous variant; default initial value & 'sum/plus' as reduction op
    template <execution_context Context, class InRange>
    requires detail::viewable_range<InRange>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (
        Context&& ctx, InRange const& input) const
    {
        auto task = start_on((Context&&)ctx, input) | operator()();

        return std::get<0>(stdexec::sync_wait(std::move(task)).value());
    }
};

inline constexpr reduce_t reduce {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct transform_reduce_t
{
    // synchronous variant
    template <execution_context Context, class InRange, class Transf, class RedOp>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Transf> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (
        Context&& ctx, InRange const& input, Transf&& transf, RedOp&& redOp) const
    {
        auto task = start_on((Context&&)ctx, input)
                  | operator()((Transf&&)transf, (RedOp&&)redOp);

        return std::get<0>(stdexec::sync_wait(std::move(task)).value());
    }


#ifdef USE_GPU


    // template <class InRange, class Transf, class RedOp>
    // requires
    //     detail::viewable_range<InRange> &&
    //     std::copy_constructible<Transf> &&
    //     std::copy_constructible<RedOp>
    // [[nodiscard]]
    // HOSTDEVICEQUALIFIER
    // constexpr
    // auto operator () (InRange const& input, Transf&& transf, RedOp&& reductionOp) const
    // {
    //     // TODO
    // }


    // // pipable, asynchronous variant
    // template <class Transf, class RedOp>
    // requires
    //     std::copy_constructible<Transf> &&
    //     std::copy_constructible<RedOp>
    // [[nodiscard]]
    // HOSTDEVICEQUALIFIER
    // constexpr
    // auto operator () (Transf&& transf, RedOp&& reductionOp) const
    // {
    //     // TODO
    // }


#else


    // pipable, asynchronous variant
    template <class InRange, class Transf, class RedOp>
    requires
        detail::viewable_range<InRange> &&
        std::copy_constructible<Transf> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    constexpr
    auto operator () (InRange const& input, Transf&& transf, RedOp&& reductionOp) const
    {
        return stdexec::let_value(
            [inView=view_of(input), tfn=(Transf&&)transf, redOp=(RedOp&&)reductionOp]
            (cpu_tag cpu)
            {
                auto const inSize      = std::ranges::size(inView);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using in_value_t = std::ranges::range_value_t<decltype(input)>;
                using tf_out_t   = std::invoke_result_t<Transf,in_value_t>;
                using result_t   = std::invoke_result_t<RedOp,tf_out_t,tf_out_t>;

                std::vector<result_t> partials (tileCount);

                return
                    stdexec::just(cpu, inView, std::move(partials))
                |   stdexec::bulk(tileCount,
                    [=](std::size_t tileIdx, cpu_tag, auto in, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        if (start < inSize) {
                            parts[tileIdx] = std::transform_reduce(
                                                std::ranges::begin(in) + start,
                                                std::ranges::begin(in) + end,
                                                result_t{},
                                                redOp, tfn);
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts), result_t{}, redOp);
                        return stdexec::just(std::move(result));
                    });
            });
    }


    // pipable, asynchronous variant
    template <class Transf, class RedOp>
    requires
        std::copy_constructible<Transf> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    constexpr
    auto operator () (Transf&& transf, RedOp&& reductionOp) const
    {
        return stdexec::let_value(
            [tfn=(Transf&&)transf, redOp=(RedOp&&)reductionOp]
            (cpu_tag cpu, auto input)
            {
                auto const inSize      = std::ranges::size(input);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using in_value_t = std::ranges::range_value_t<decltype(input)>;
                using tf_out_t   = std::invoke_result_t<Transf,in_value_t>;
                using result_t   = std::invoke_result_t<RedOp,tf_out_t,tf_out_t>;

                std::vector<result_t> partials (tileCount);

                return
                    stdexec::just(cpu, input, std::move(partials))
                |   stdexec::bulk(tileCount,
                    [=](std::size_t tileIdx, cpu_tag, auto in, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        if (start < inSize) {
                            parts[tileIdx] = std::transform_reduce(
                                                std::ranges::begin(in) + start,
                                                std::ranges::begin(in) + end,
                                                result_t{},
                                                redOp, tfn);
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts), result_t{}, redOp);
                                return stdexec::just(std::move(result));
                    });
            });
    }
#endif
};

inline constexpr transform_reduce_t transform_reduce {};






//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct zip_reduce_t
{
    // synchronous variant
    template <execution_context Context, class InRange, class RedOp>
    requires
        detail::viewable_range<InRange> &&
        detail::viewable_range<InRange> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (
        Context&& ctx, InRange const& input1, InRange const& input2, RedOp&& redOp) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(input1, input2, (RedOp&&)redOp);

        return std::get<0>(stdexec::sync_wait(std::move(task)).value());
    }


#ifdef USE_GPU


    // pipable variant, for cases whithout ranges in sender chain
    // template <class InRange1, class InRange2, class RedOp>
    // requires
    //     detail::viewable_range<InRange1> &&
    //     detail::viewable_range<InRange2> &&
    //     std::copy_constructible<RedOp>
    // [[nodiscard]]
    // constexpr
    // auto operator () (
    //     InRange1 const& input1, InRange2 const& input2, RedOp&& reductionOp) const
    // {
    //     // TODO
    // }


    // pipable variant, for cases with already two ranges in sender chain
    // template <class RedOp>
    // requires
    //     std::copy_constructible<RedOp>
    // [[nodiscard]]
    // constexpr
    // auto operator () (RedOp&& reductionOp) const
    // {
    //     // TODO
    // }


#else


    // pipable variant, for cases whithout ranges in sender chain
    template <class InRange1, class InRange2, class RedOp>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    constexpr
    auto operator () (
        InRange1 const& input1, InRange2 const& input2, RedOp&& reductionOp) const
    {
        return stdexec::let_value(
            [inView1=view_of(input1), inView2=view_of(input2),
             redOp=(RedOp&&)reductionOp](cpu_tag cpu)
            {
                auto const inSize = detail::range_size(inView1, inView2);

                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using value_t = std::ranges::range_value_t<decltype(inView1)>;

                std::vector<value_t> partials (tileCount);

                return
                    stdexec::just(cpu, inView1, inView2, std::move(partials))
                |   stdexec::bulk(tileCount,
                    [=](std::size_t tileIdx,
                        cpu_tag, auto in1, auto in2, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        for (auto i = start; i < end; ++i) {
                            parts[tileIdx] = redOp(std::as_const(parts[tileIdx]), in1[i], in2[i]);
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto, auto&& parts)
                    {
                        if (parts.empty()) {
                            return stdexec::just(value_t{});
                        }
                        if (parts.size() == 1) {
                            return stdexec::just(parts[0]);
                        }
                        // split off first value in case of odd number of partials
                        auto const startIdx = tileCount % 2;
                        auto result = (startIdx > 0) ? parts[0] : value_t{};
                        // reduce in pairs of 2
                        for (std::size_t i = startIdx; i < parts.size(); i += 2) {
                            result = redOp(std::as_const(result), parts[i], parts[i+1]);
                        }

                        return stdexec::just(std::move(result));
                    });
            });
    }


    // pipable variant, for cases with already two ranges in sender chain
    template <class RedOp>
    requires
        std::copy_constructible<RedOp>
    [[nodiscard]]
    constexpr
    auto operator () (RedOp&& reductionOp) const
    {
        return stdexec::let_value(
            [redOp=(RedOp&&)reductionOp] (cpu_tag cpu, auto in1, auto in2)
            {
                auto const inSize      = detail::range_size(in1,in2);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using value_t = std::ranges::range_value_t<decltype(in1)>;

                std::vector<value_t> partials (tileCount);

                return
                    stdexec::just(cpu, in1, in2, std::move(partials))
                |   stdexec::bulk(tileCount,
                    [=](std::size_t tileIdx,
                        cpu_tag, auto in1, auto in2, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        for (auto i = start; i < end; ++i) {
                            parts[tileIdx] = redOp(std::as_const(parts[tileIdx]), in1[i], in2[i]);
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto, auto&& parts)
                    {
                        if (parts.empty()) {
                            return stdexec::just(value_t{});
                        }
                        if (parts.size() == 1) {
                            return stdexec::just(parts[0]);
                        }
                        // split off first value in case of odd number of partials
                        auto const startIdx = tileCount % 2;
                        auto result = (startIdx > 0) ? parts[0] : value_t{};
                        // reduce in pairs of 2
                        for (std::size_t i = startIdx; i < parts.size(); i += 2) {
                            result = redOp(std::as_const(result), parts[i], parts[i+1]);
                        }

                        return stdexec::just(std::move(result));
                    });
            });
    }

#endif
};

inline constexpr zip_reduce_t zip_reduce {};




//-----------------------------------------------------------------------------
/**
 * @brief
 */
struct zip_transform_reduce_t
{
    // synchronous variant
    template <execution_context Context, class InRange1, class InRange2, class Transf, class RedOp>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<Transf> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (Context&& ctx,
        InRange1 const& input1, InRange2 const& input2,
        Transf&& transf, RedOp&& redOp) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator()(input1, input2, (Transf&&)transf, (RedOp&&)redOp);

        return std::get<0>(stdexec::sync_wait(std::move(task)).value());
    }


#ifdef USE_GPU


    // // pipable, asynchronous variant
    // template <class InRange1, class InRange2, class Transf, class RedOp>
    // requires
    //     detail::viewable_range<InRange1> &&
    //     detail::viewable_range<InRange2> &&
    //     std::copy_constructible<Transf> &&
    //     std::copy_constructible<RedOp>
    // [[nodiscard]]
    // HOSTDEVICEQUALIFIER
    // constexpr
    // auto operator () (
    //     InRange1 const& input1, InRange2 const& input2,
    //     Transf&& transf, RedOp&& reductionOp) const
    // {
    //     // TODO
    // }


    // // pipable, asynchronous variant
    // template <class Transf, class RedOp>
    // requires
    //     std::copy_constructible<Transf> &&
    //     std::copy_constructible<RedOp>
    // [[nodiscard]]
    // HOSTDEVICEQUALIFIER
    // constexpr
    // auto operator () (Transf&& transf, RedOp&& reductionOp) const
    // {
    //     // TODO
    // }


#else


    // pipable, asynchronous variant
    template <class InRange1, class InRange2, class Transf, class RedOp>
    requires
        detail::viewable_range<InRange1> &&
        detail::viewable_range<InRange2> &&
        std::copy_constructible<Transf> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    constexpr
    auto operator () (
        InRange1 const& input1, InRange2 const& input2,
        Transf&& transf, RedOp&& reductionOp) const
    {
        return stdexec::let_value(
            [inView1=view_of(input1), inView2=view_of(input2),
             tfn=(Transf&&)transf, redOp=(RedOp&&)reductionOp]
            (cpu_tag cpu)
            {
                auto const inSize      = detail::range_size(inView1, inView2);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using in1_value_t = std::ranges::range_value_t<decltype(inView1)>;
                using in2_value_t = std::ranges::range_value_t<decltype(inView2)>;
                using tf_out_t    = std::invoke_result_t<Transf,in1_value_t,in2_value_t>;
                using result_t    = std::invoke_result_t<RedOp,tf_out_t,tf_out_t>;

                std::vector<result_t> partials (tileCount);

                return
                    stdexec::just(cpu, inView1, inView2, std::move(partials))
                |   stdexec::bulk(tileCount,
                    [=](std::size_t tileIdx, cpu_tag, auto in1, auto in2, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        for (auto i = start; i < end; ++i) {
                            parts[tileIdx] = redOp(parts[tileIdx], tfn(in1[i], in2[i]));
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts), result_t{}, redOp);
                        return stdexec::just(std::move(result));
                    });
                
            });
    }


    // pipable, asynchronous variant
    template <class Transf, class RedOp>
    requires
        std::copy_constructible<Transf> &&
        std::copy_constructible<RedOp>
    [[nodiscard]]
    constexpr
    auto operator () (Transf&& transf, RedOp&& reductionOp) const
    {
        return stdexec::let_value(
            [tfn=(Transf&&)transf, redOp=(RedOp&&)reductionOp]
            (cpu_tag cpu, auto in1, auto in2)
            {
                auto const inSize      = detail::range_size(in1, in2);
                auto const threadCount = std::min(inSize, static_cast<std::size_t>(cpu.threads));
                auto const tileSize    = (inSize + threadCount - 1) / threadCount;
                auto const tileCount   = (inSize + tileSize - 1) / tileSize;

                using in1_value_t = std::ranges::range_value_t<decltype(in1)>;
                using in2_value_t = std::ranges::range_value_t<decltype(in2)>;
                using tf_out_t    = std::invoke_result_t<Transf,in1_value_t,in2_value_t>;
                using result_t    = std::invoke_result_t<RedOp,tf_out_t,tf_out_t>;

                std::vector<result_t> partials (tileCount);

                return
                    stdexec::just(cpu, in1, in2, std::move(partials))
                |   stdexec::bulk(tileCount,
                    [=](std::size_t tileIdx, cpu_tag, auto in1, auto in2, auto&& parts)
                    {
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);

                        for (auto i = start; i < end; ++i) {
                            parts[tileIdx] = redOp(parts[tileIdx], tfn(in1[i], in2[i]));
                        }
                    })
                |   stdexec::let_value([=](cpu_tag, auto, auto, auto&& parts)
                    {
                        auto result = std::reduce(begin(parts), end(parts), result_t{}, redOp);
                        return stdexec::just(std::move(result));
                    });
            });
    }
#endif
};

inline constexpr zip_transform_reduce_t zip_transform_reduce {};







//-----------------------------------------------------------------------------
/**
 * @brief synchronous, parallel loop over an n-dimensional index space
 */
struct for_each_grid_index_t
{
    template <execution_context Context, class GridExtents, class Body>
    requires
        std::copy_constructible<Body>
    HOSTDEVICEQUALIFIER
    constexpr
    void operator () (Context&& ctx, GridExtents ext, Body&& body) const
    {
        auto task = start_on((Context&&)ctx)
                  | operator ()(ext, (Body&&)body);

        stdexec::sync_wait(std::move(task)).value();
    }


#ifdef USE_GPU


    template <class GridExtents, class Body>
    requires
        std::copy_constructible<Body>
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (GridExtents ext, Body&& body) const
    {
        return stdexec::let_value(
            [ext,body=(Body&&)body](gpu_tag gpu)
            {
                // size of collapsed index range
                std::size_t size = 1;
                for (auto x : ext) { size *= static_cast<std::size_t>(x); }

                auto const ndims = static_cast<int>(ext.size());

                return
                    stdexec::just(gpu)
                |   stdexec::bulk(size, [=](std::size_t tidx, gpu_tag)
                    {
                        // compute n-dimensional index
                        GridExtents idx;
                        for (int i = 0; i < ndims; ++i) { idx[i] = 0; }

                        if (tidx > 0) {
                        std::size_t mul[ndims];
                            mul[0] = 1;
                            for (int i = 0; i < ndims-1; ++i) {
                                mul[i+1] = mul[i] * ext[i];
                            }
                            auto offset = tidx;
                            for (int i = ndims; i > 0; --i) {
                                if (offset >= mul[i-1]) {
                                    idx[i-1] += offset / mul[i-1];
                                    offset = offset % mul[i-1];
                                    if (offset == 0) break;
                                }
                            }
                        }

                        body(idx);
                    });
            });
    }


#else


    template <class GridExtents, class Body>
    requires
        std::copy_constructible<Body>
    HOSTDEVICEQUALIFIER
    constexpr
    auto operator () (GridExtents ext, Body&& body) const
    {
        return stdexec::let_value(
            [ext,body=(Body&&)body](cpu_tag cpu)
            {
                // size of collapsed index range
                std::size_t size = 1;
                for (auto x : ext) { size *= static_cast<std::size_t>(x); }

                auto const ndims = static_cast<int>(ext.size());

                auto const threadCount = std::min(size, static_cast<std::size_t>(cpu.threads));
                auto const tileSize  = static_cast<std::size_t>((size + threadCount - 1) / threadCount);
                auto const tileCount = (size + tileSize - 1) / tileSize;

                return
                    stdexec::just(cpu)
                |   stdexec::bulk(tileCount, [=](std::size_t tileIdx, cpu_tag)
                    {
                        // start/end of collapsed index range
                        auto const start = tileIdx * tileSize;
                        auto const end   = std::min(size, (tileIdx + 1) * tileSize);
                        if (start >= end) return;
                        // compute start index
                        GridExtents idx;
                        for (int i = 0; i < ndims; ++i) { idx[i] = 0; }

                        if (start > 0) {
                        std::size_t mul[ndims];
                            mul[0] = 1;
                            for (int i = 0; i < ndims-1; ++i) {
                                mul[i+1] = mul[i] * ext[i];
                            }
                            auto offset = start;
                            for (int i = ndims; i > 0; --i) {
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
                            for (int i = 0; i < ndims; ++i) {
                                ++idx[i];
                                if (idx[i] < ext[i]) break;
                                idx[i] = 0;
                            }
                        }
                    });
                
            });
    }
#endif
};

inline constexpr for_each_grid_index_t for_each_grid_index {};






//-----------------------------------------------------------------------------
template <class Fn, class T>
concept NearestNeighborFn =
    std::floating_point<T> &&
    std::invocable<Fn,T,T,T,T,T> &&
    std::same_as<T,std::invoke_result_t<Fn,T,T,T,T,T>>;


[[nodiscard]]
HOSTDEVICEQUALIFIER
stdexec::sender auto
transform_matrix_nearest_neigbors_async (
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

    return
        stdexec::transfer_just(sch, matrix)
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
        })
    |
        stdexec::then([](span2d<double> m){ return m; });
}




//-----------------------------------------------------------------------------
template <NearestNeighborFn<double> Fn>
HOSTDEVICEQUALIFIER
void transform_matrix_nearest_neigbors (
    stdexec::scheduler auto sched,
    span2d<double> matrix,
    double border,
    std::size_t stripeCount,
    Fn&& fn)
{
    auto task = transform_matrix_nearest_neigbors_async(
            sched, matrix, border, stripeCount, (Fn&&)(fn));

    stdexec::sync_wait(std::move(task)).value();
}






//-----------------------------------------------------------------------------
template <class Fun, class... Args>
concept Predicate =
    std::invocable<Fun, Args...> &&
    std::same_as<std::invoke_result_t<Fun, Args...>, bool>;

template <class... Ts>
using just_sender_t = decltype(stdexec::just(std::declval<Ts>()...));

template <
    class Pred,
    stdexec::__sender_adaptor_closure Then,
    stdexec::__sender_adaptor_closure Else
>
auto if_then_else (Pred pred, Then then_, Else else_)
{
    return stdexec::let_value(
        [=]<class... Args>(Args&&... args) mutable
        {
            return std::move(then_)(stdexec::just((Args&&)args...));
        }
    );

    // return stdexec::let_value(
    //     [=]<class... Args>(Args&&... args) mutable
    //         -> exec::variant_sender<
    //             std::invoke_result_t<Then, just_sender_t<Args...>>,
    //             std::invoke_result_t<Else, just_sender_t<Args...>>>
    //         requires Predicate<Pred, Args&...>
    //     {
    //         if (pred(args...)) {
    //             return std::move(then_)(stdexec::just((Args&&)args...));
    //         }
    //         else {
    //             return std::move(else_)(stdexec::just((Args&&)args...));
    //         }
    //     }
    // );
}



}  // namespace ex


#endif

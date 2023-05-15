
#include "execution.hpp"
#include "timer.hpp"
#include "algorithms.hpp"
#include "storage.hpp"

#ifdef USE_GPU
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#endif

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdio>
#include <iostream>




//-----------------------------------------------------------------------------
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




//-----------------------------------------------------------------------------
void hello_world (int nthreads)
{
    Compute_Resource rsc;
    Execution_Context ctx {rsc};

    auto const rshp = ctx.resource_shape();
    fmt::print("devices: {}\n", rshp.devices);

    auto hello = stdexec::schedule(ctx.get_scheduler()) 
        |   stdexec::bulk(nthreads, [](auto id){ 
                printf("hello from thread %d\n",id);
            });

    stdexec::sync_wait(hello).value();
}




//-----------------------------------------------------------------------------
[[nodiscard]] stdexec::sender auto
schedule_value (stdexec::scheduler auto sched)
{
    int i = 123;
    return stdexec::transfer_just(sched, i)
        | stdexec::then([](int i){ return i + 1; } );
}

void test_value ()
{
    fmt::print("TEST: value\n");
    exec::static_thread_pool tpool (4);
    stdexec::scheduler auto sched = tpool.get_scheduler();
    auto task = schedule_value(sched);
    auto [x] = stdexec::sync_wait(task).value();
    fmt::print("result: {}\n", x);
}




//-----------------------------------------------------------------------------
void test_cpu_bulk ()
{
    fmt::print("CPU test bulk\n");

    auto const maxThreads = std::thread::hardware_concurrency();

    exec::static_thread_pool tpool (maxThreads);
    stdexec::scheduler auto sched = tpool.get_scheduler();

    auto task = stdexec::schedule(sched)
        |   stdexec::bulk(maxThreads, [](auto id){
                fmt::print("Hello from {}\n", id);
            });

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
// void test_scan (std::size_t nelems)
// {
//     fmt::print("CPU test scan\n");
//
//     bulk_vector<double> input (nelems);
//     bulk_vector<double> output (nelems);
//     random_engine urng;
//     fill_random(input, urng);
//     std::ranges::fill(output, 0.0);
//     fmt::print("input ready\n");
//
//
//     auto const nthreads = std::thread::hardware_concurrency();
//
//     exec::static_thread_pool tpool (nthreads);
//     stdexec::scheduler auto sched = tpool.get_scheduler();
//
//     am::timer time;
//     time.start();
//
//     auto task = inclusive_scan_async(sched, input, output, 0.0, nthreads);
//
//     stdexec::sync_wait(std::move(task)).value();
//
//     time.stop();
//
//     fmt::print("result: {}\n", output);
//     fmt::print("time: {} ms\n", time.milliseconds());
// }




//-----------------------------------------------------------------------------
#ifdef USE_GPU
void test_gpu_bulk ()
{
    fmt::print("GPU test bulk\n");

    nvexec::stream_context gpu_ctx;
    stdexec::scheduler auto sched = gpu_ctx.get_scheduler();

    auto task = stdexec::schedule(sched)
        |   stdexec::bulk(1024, [](auto id){
                printf("Hello from %d\n", id);
            });

    stdexec::sync_wait(std::move(task)).value();
}
#endif



//-----------------------------------------------------------------------------
void test_matrix_nn_sweep (std::size_t nrows, std::size_t ncols, int maxThreads)
{
    fmt::print("TEST: matrix NN sweep\n");

    exec::static_thread_pool tpool (maxThreads);
    stdexec::scheduler auto sched = tpool.get_scheduler();

    bulk_vector<double> data (nrows*ncols,0.0);
    double const borderValue = 1.0;

    // bool flip = false;
    // for (auto& x : data) {
    //     x = flip ? 1 : 0;
    //     flip = not flip;
    // }

    span2d matrix {data.data(),nrows,ncols};

    am::timer time;
    time.start();

    auto task = transform_matrix_nearest_neigbors_async(
        sched, matrix, borderValue, maxThreads,
        [](            double up,
            double le, double mi, double re,
                    double dn )
        {
            return 0.25 * (0.25*up + 0.25*le + 3.0*mi + 0.25*re + 0.25*dn);
        });

    stdexec::sync_wait(std::move(task)).value();

    time.stop();

    // print(matrix);
    auto const sum = std::accumulate(data.begin(),data.end(),0.0);
    fmt::print("sum:  {}\n", sum);
    fmt::print("time: {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
// compute horizontal stripes in parallel - omit first and last row
void matrix_sweep (stdexec::scheduler auto sched, span2d<double> matrix, std::size_t stripeCount)
{
    std::size_t const stripeHeight = (matrix.nrows() + stripeCount - 1) / stripeCount;

    auto task = stdexec::transfer_just(sched, matrix)
        | stdexec::bulk(stripeCount,
            [=](std::size_t stripeIdx, span2d<double> m)
            {
                std::size_t const rstart = std::max(std::size_t(1), stripeIdx * stripeHeight);
                std::size_t const rend   = std::min(m.nrows()-1, (stripeIdx+1) * stripeHeight);

                for (std::size_t r = rstart; r < rend; ++r) {
                    for (std::size_t c = 0; c < m.ncols(); ++c) {
                        m(r,c) = r * 10 + c;
                    }
                }
            })
        | stdexec::then([](span2d<double> m){ return m; });

    stdexec::sync_wait(task).value();
}


void test_matrix_sweep ()
{
    fmt::print("TEST: matrix sweep\n");

    auto const maxThreads = 4;

    exec::static_thread_pool tpool (maxThreads);
    stdexec::scheduler auto sched = tpool.get_scheduler();

    std::size_t const nrows = 10;
    std::size_t const ncols = 10;

    bulk_vector<double> data (nrows*ncols,1.0);
    span2d matrix {data.data(),nrows,ncols};

    matrix_sweep(sched, matrix, maxThreads);

    print(matrix);
}




//-----------------------------------------------------------------------------
void test_openmp (std::size_t nelems)
{
#ifdef _OPENMP
    fmt::print("OpenMP {}\n", omp_get_num_threads());
#endif

    // using value_t = double;
    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    bulk_vector<value_t> output (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    // #pragma omp parallel for simd
    // #pragma omp parallel for
    for (std::size_t i = 0; i < nelems; ++i) {
        output[i] = cross(input1[i], input2[i]);
    }

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:  {}\n", sum);
    fmt::print("time: {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
#ifndef USE_GPU
void test_stdpar (std::size_t nelems)
{
    fmt::print("TEST: stdpar\n");

    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    bulk_vector<value_t> output (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    auto const idx = am::index_range{0,nelems};
    // std::for_each(
    // std::for_each(std::execution::unseq,
    // std::for_each(std::execution::par_unseq,
    std::for_each(std::execution::par,
        idx.begin(), idx.end(),
        [&](std::size_t i){
            output[i] = cross(input1[i], input2[i]);
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:  {}\n", sum);
    fmt::print("time: {} ms\n", time.milliseconds());
}
#endif



//-----------------------------------------------------------------------------
void test_for_each (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: for_each\n");

    // using value_t = double;
    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    bulk_vector<value_t> output (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    for_each(ctx, am::index_range{0,nelems},
        [in1 = view_of(input1),
         in2 = view_of(input2),
         out = view_of(output) ] ( std::size_t i) 
        {
            out[i] = cross(in1[i],in2[i]);  
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_for_each_grid_index (
    Execution_Context ctx, 
    std::size_t na, std::size_t nb, std::size_t nc, std::size_t nd, std::size_t ne)
{
    fmt::print("TEST: for_each_grid_index\n");

    // using value_t = double;
    using value_t = vec3d;
    auto const nelems = na * nb * nc * nd * ne;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    bulk_vector<value_t> output (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready: {}\n", nelems);

    am::timer time;
    time.start();

    for_each_grid_index(ctx, std::array{na,nb,nc,nd,ne},
        [=,in1 = view_of(input1),
           in2 = view_of(input2),
           out = view_of(output) ] (auto const idx)
        {
            auto const i = idx[0] + idx[1]*na + idx[2]*(na*nb)
                         + idx[3]*(na*nb*nc) + idx[4]*(na*nb*nc*nd); 

            out[i] = cross(in1[i],in2[i]);
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_generate_indexed (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: generate_indexed\n");

    // using value_t = double;
    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    bulk_vector<value_t> output (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    // generate_indexed(ctx, output,
    //     [in1 = view_of(input1),
    //      in2 = view_of(input2)] (std::size_t i) 
    //     {
    //         return cross(in1[i],in2[i]); 
    //     });

    auto task = generate_indexed_async(ctx, output,
        [in1 = view_of(input1),
         in2 = view_of(input2)] (std::size_t i) 
        {
            return cross(in1[i],in2[i]); 
        });

    stdexec::sync_wait(std::move(task)).value();

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_transform (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: transform\n");

    bulk_vector<double> input (nelems);
    bulk_vector<double> output (nelems);
    random_engine urng;
    fill_random(input, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    transform(ctx, input, output, [] (double x) { return 2.0 * x; });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), 0.0);
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_zip_transform (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: zip_transform\n");

    // using value_t = double;
    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    bulk_vector<value_t> output (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    zip_transform(ctx, input1, input2, output,
        [](vec3d const& in1, vec3d const& in2) {
            return cross(in1, in2) ;
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_reduce (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: reduce\n");

    bulk_vector<double> input (nelems);
    random_engine urng;
    fill_random(input, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    auto task = reduce_async(ctx, input, 0.0,
        [](double total, double in) { return total + in; });

    auto [sum] = stdexec::sync_wait(std::move(task)).value();

    time.stop();

    if (input.size() <= 20) { fmt::print("input:   {}\n", input); }
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_zip_reduce (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: zip_reduce\n");

    // using value_t = double;
    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    auto const sum = zip_reduce(ctx, input1, input2, vec3d{0,0,0},
        [](vec3d total, vec3d const& in1, vec3d const& in2) {
            total += in1;
            total += in2;
            return total;
        });

    time.stop();

    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_zip_reduce_sum (Execution_Context ctx, std::size_t nelems)
{
    fmt::print("TEST: zip_reduce_sum\n");

    // using value_t = double;
    using value_t = vec3d;
    bulk_vector<value_t> input1 (nelems);
    bulk_vector<value_t> input2 (nelems);
    random_engine urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    am::timer time;
    time.start();

    auto const sum = zip_reduce_sum(ctx, input1, input2, vec3d{0,0,0},
        [](vec3d const& in1, vec3d const& in2) {
            return vec3d{ in1.x + in2.x, in1.y + in2.y, in1.z + in2.z };
        });

    time.stop();

    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.resource_shape().devices);
    fmt::print("threads: {}\n", ctx.resource_shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}






//-----------------------------------------------------------------------------
int main (int argc, char* argv[])
{

// #ifdef USE_GPU
//     auto const defaultNthreads = 1024;
// #else // CPU
//     auto const defaultNthreads = std::thread::hardware_concurrency();
// #endif

    // int const nthreads = (argc > 1) ? std::atoi(argv[1]) : defaultNthreads;
    // fmt::print("max. threads: {}\n", nthreads);
    // hello_world(nthreads);

    // test_value();
    // test_bulk();

    // int const nrows = (argc > 1) ? std::atoi(argv[1]) : 10;
    // int const ncols = (argc > 2) ? std::atoi(argv[2]) : 10;
    // int const nthreads = (argc > 3) ? std::atoi(argv[3]) : defaultNthreads;
    // fmt::print("threads: {}\n", nthreads);
    // test_matrix_nn_sweep(nrows, ncols, nthreads);
    
    // int const nthreads = (argc > 2) ? std::atoi(argv[2]) : defaultNthreads;
    // fmt::print("max. threads: {}\n", nthreads);

    // int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;
    // test_openmp(nelems);
    // test_stdpar(nelems);

    int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;

    // test_scan(nelems);

    // for (auto threads : {16,8,4,2,1}) {
    //     Compute_Resource cpr {threads};
    //     test_reduce(cpr, nelems);
    // }

    Compute_Resource cpr;

    // test_for_each(cpr, nelems);
    // test_generate_indexed(cpr, nelems);
    // test_transform(cpr, nelems);
    // test_zip_transform(cpr, nelems);
    test_reduce(cpr, nelems);
    // test_zip_reduce(cpr, nelems);
    // test_zip_reduce_sum(cpr, nelems);

    // int const na = (argc > 1) ? std::atoi(argv[1]) : 2;
    // int const nb = (argc > 2) ? std::atoi(argv[2]) : na;
    // int const nc = (argc > 3) ? std::atoi(argv[3]) : nb;
    // int const nd = (argc > 4) ? std::atoi(argv[4]) : nc;
    // int const ne = (argc > 5) ? std::atoi(argv[5]) : nd;
    // test_for_each_grid_index(cpr, na,nb,nc,nd,ne);

    // run_test();

    // any_sender_of<stdexec::set_value_t(std::span<value_t>),
    //               stdexec::set_stopped_t(),
    //               stdexec::set_error_t(std::exception_ptr)> 
    //     task0 = with_elements(sched, output);
}


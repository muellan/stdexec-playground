
#include "acceleration.hpp"
#include "timer.hpp"
#include "algorithms.hpp"
#include "vecmath.hpp"
#include "storage.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>



//-----------------------------------------------------------------------------
void demo_sync_for_each (ex::acceleration_context ctx, std::size_t nelems)
{
    using ex::view_of;

    fmt::print("DEMO: for_each\n");

    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::vector<vec3d> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    // ex::for_each(ctx, ex::index_range{0,nelems},
    ex::for_each(ctx, std::array{1,3,5,7},
        [in1 = view_of(input1),
         in2 = view_of(input2),
         out = view_of(output) ] ( std::size_t i) 
        {
            out[i] = cross(in1[i],in2[i]);  
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void demo_sync_for_each_grid_index (
    ex::acceleration_context ctx, 
    std::size_t na, std::size_t nb, std::size_t nc, std::size_t nd, std::size_t ne)
{
    using ex::view_of;

    fmt::print("DEMO: for_each_grid_index\n");

    auto const nelems = na * nb * nc * nd * ne;
    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::vector<vec3d> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready: {}\n", nelems);

    ex::timer time;
    time.start();

    ex::for_each_grid_index(ctx, std::array{na,nb,nc,nd,ne},
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
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void demo_sync_generate_indexed (ex::acceleration_context ctx, std::size_t nelems)
{
    using ex::view_of;

    fmt::print("DEMO: generate_indexed\n");

    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::vector<vec3d> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    ex::generate_indexed(ctx, output,
        [in1 = view_of(input1),
         in2 = view_of(input2)] (std::size_t i) 
        {
            return cross(in1[i],in2[i]); 
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void demo_sync_transform (ex::acceleration_context ctx, std::size_t nelems)
{
    fmt::print("DEMO: transform\n");

    ex::vector<double> input (nelems);
    ex::vector<double> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    ex::transform(ctx, input, output, [] (double x) { return 2.0 * x; });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), 0.0);
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void demo_sync_zip_transform (ex::acceleration_context ctx, std::size_t nelems)
{
    fmt::print("DEMO: zip_transform\n");

    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::vector<vec3d> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    ex::zip_transform(ctx, input1, input2, output,
        [](vec3d const& in1, vec3d const& in2) {
            return cross(in1, in2) ;
        });

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void demo_sync_reduce (ex::acceleration_context ctx, std::size_t nelems)
{
    fmt::print("DEMO: reduce\n");

    ex::vector<double> input (nelems);
    ex::random_engine urng;
    ex::fill_random(input, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    auto sum = ex::reduce(ctx, input, 0.0,
        [](double total, double in) { return total + in; });

    time.stop();

    if (input.size() <= 20) { fmt::print("input:   {}\n", input); }
    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void demo_sync_zip_reduce (ex::acceleration_context ctx, std::size_t nelems)
{
    fmt::print("DEMO: zip_reduce\n");

    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    auto const sum = ex::zip_reduce(ctx, input1, input2,  // vec3d{0,0,0},
        [](vec3d acc, vec3d const& in1, vec3d const& in2) {
            acc += in1;
            acc += in2;
            return acc;
        });

    time.stop();

    fmt::print("sum:     {}\n", sum);
    fmt::print("devices: {}\n", ctx.shape().devices);
    fmt::print("threads: {}\n", ctx.shape().threads);
    fmt::print("time:    {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
int main (int argc, char* argv[])
{
    int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;

#ifdef USE_GPU
    ex::acceleration_resource cpr {};
#else
    auto const defaultNthreads = std::thread::hardware_concurrency();
    int const nthreads = (argc > 2) ? std::atoi(argv[2]) : defaultNthreads;
    ex::acceleration_resource cpr {nthreads};
#endif
       
    demo_sync_generate_indexed(cpr, nelems);
    demo_sync_for_each(cpr, nelems);
    demo_sync_transform(cpr, nelems);
    demo_sync_zip_transform(cpr, nelems);
    demo_sync_reduce(cpr, nelems);
    demo_sync_zip_reduce(cpr, nelems);

    // int const na = (argc > 1) ? std::atoi(argv[1]) : 2;
    // int const nb = (argc > 2) ? std::atoi(argv[2]) : na;
    // int const nc = (argc > 3) ? std::atoi(argv[3]) : nb;
    // int const nd = (argc > 4) ? std::atoi(argv[4]) : nc;
    // int const ne = (argc > 5) ? std::atoi(argv[5]) : nd;
    // demo_sync_for_each_grid_index(cpr, na,nb,nc,nd,ne);
}


#include "timer.hpp"
#include "algorithms.hpp"
#include "vecmath.hpp"
#include "storage.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>


int main (int argc, char* argv[])
{
    int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;

    // host or device vectors depending on "USE_GPU"
    ex::vector<vec3d> vecRange1 (nelems);
    ex::vector<vec3d> vecRange2 (nelems);

    ex::random_engine urng;
    ex::fill_random(vecRange1, urng);
    // ex::fill_random(vecRange2, urng);
    // vecRange2 = vecRange1;

    // ex::fill_default(vecRange1);
    // ex::fill_default(vecRange2);

    fmt::print("input ready\n");

    // execution context
#ifdef USE_GPU
    {
        nvexec::stream_context ctx;
#else  // CPU
    for (std::uint32_t nthreads : {16,8,4,2,1})
    {
        exec::static_thread_pool ctx {nthreads};
#endif
        ex::timer time;
        time.start();

        auto task =
            // start chain on execution context or scheduler
            // attaches compute information (#threas,#devices,etc.)
            // ex::start_on(ctx, vecRange1)
            ex::start_on(ctx)
            // async range algorithm
        // |   ex::fill(vec3d{1.0,2.0,3.0})
        // |   ex::generate_indexed([](std::size_t i){ return vec3d{1.0*i,2.0*i,3.0*i}; })
        // |   ex::transform([](vec3d const& a){ return vec3d{-a.x, -a.y, -a.z}; })
        // |   ex::transform(vecRange2, [](vec3d const& a){ return vec3d{-a.x, -a.y, -a.z}; })
        |   ex::transform(vecRange1,vecRange2, [](vec3d const& a){ return vec3d{-a.x, -a.y, -a.z}; })
        // |   ex::transform_reduce(
        //         [](vec3d const& a){  // transformation
        //             return vec3d{-a.x, -a.y, -a.z};
        //         },
        //         [](vec3d const& a, vec3d const& b){  // reduction
        //             return vec3d{a.x + b.x, a.y + b.y, a.z + b.z};
        //         }
        //     )
            //  returns first value in chain (the reduction result in this case)
        // |   ex::return_result()
        ;

        // auto [result] = stdexec::sync_wait(std::move(task)).value();
        stdexec::sync_wait(std::move(task)).value();

        time.stop();

#ifndef USE_GPU
        fmt::print("threads: {}\n", nthreads);
#endif
        if (nelems <= 100) {
            fmt::print("vecRange1: {}\n{:-<20}\n", vecRange1,"");
            fmt::print("vecRange2: {}\n{:-<20}\n", vecRange2,"");
        }
        // fmt::print("result:  {}\n{:-<20}\n", result,"");
        fmt::print("time:    {} ms\n", time.milliseconds());
    }
}


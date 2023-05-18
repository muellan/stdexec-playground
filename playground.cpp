
#include "acceleration.hpp"
#include "algorithms.hpp"
// #include "storage.hpp"
// #include "timer.hpp"
// #include "vecmath.hpp"

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#ifdef USE_GPU
    #include <nvexec/stream_context.cuh>
    #include <nvexec/multi_gpu_context.cuh>
    #include <nvexec/stream/reduce.cuh>
    #include <thrust/universal_vector.h>
#else
    #include <fmt/format.h>
    #include <fmt/ranges.h>
#endif

#include <vector>
#include <span>
#include <algorithm>
#include <numeric>
// #include <iostream>
// #include <cstdlib>
// #include <cstdio>
// #include <random>


// template <stdexec::__sender_adaptor_closure Then>
// auto relay (Then then_)
// {
//     return stdexec::let_value(
//         [=]<class... Args>(Args&&... args) mutable
//         {
//             return std::move(then_)(stdexec::just((Args&&)args...));
//         }
//     );
// }

template <stdexec::scheduler Sched>
auto start (Sched sched) {
    return stdexec::schedule(sched);
}

template <stdexec::scheduler Sched, std::ranges::range Range>
auto start (Sched sched, Range&& r) {
    return stdexec::transfer_just(sched, ex::view_of(r));
}

// template <class Fn>
// auto gen (Fn&& fn) {
//     return stdexec::let_value([fn = (Fn&&)fn](auto out){
//         return  stdexec::just(out)
//             |   stdexec::bulk(out.size(),
//                 [=](std::size_t i, auto out) {
//                     return out[i] = fn(i);
//                 });
//     });
// }

// template <std::ranges::range Range, class Fn>
// auto gen (Range&& r, Fn&& fn)
// {
//     return stdexec::let_value([fn = (Fn&&)fn, out = ex::view_of(r)](){
//         auto const size = std::ranges::size(out);
//         return  stdexec::just(out)
//             |   stdexec::bulk(size,
//                 [fn](std::size_t i, auto out) { 
//                     return out[i] = fn(i);
//                 });
//     });
// }

// template <stdexec::scheduler Sched, std::ranges::range Range, class Fn>
// auto gen (Sched sched, Range&& r, Fn&& fn)
// {
//     return  stdexec::transfer_just(sched, ex::view_of(r))
//         |   stdexec::bulk(std::ranges::size(r),
//             [fn = (Fn&&)fn](std::size_t i, auto out) {
//                 return out[i] = fn(i);
//             });
// }




//-----------------------------------------------------------------------------
int main (int argc, char* argv[])
{
#ifdef USE_GPU
    using vector_t  = thrust::universal_vector<double>;
    using context_t = nvexec::stream_context;
#else
    using vector_t  = std::vector<double>;
    using context_t = exec::static_thread_pool;
#endif

    int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;

    vector_t inRange (nelems);
    std::iota(inRange.begin(), inRange.end(), 0.0);
    // vector_t outRange (nelems);

    context_t ctx;

    // auto task = 
    //         stdexec::transfer_just(ctx.get_scheduler(),ex::view_of(inRange))
    //     |   stdexec::let_value([](auto output){
    //             // return 
    //             //     stdexec::bulk(stdexec::just(output), output.size(),
    //             //     [](std::size_t i, auto out) { out[i] = double(i); });
    //             
    //             return  stdexec::just(output)
    //                 |   stdexec::bulk(output.size(),
    //                     [](std::size_t i, auto out) { 
    //                         auto const n = out.size() * out.size();
    //                         for (int j = 0; j < n; ++j) {
    //                             out[i] += double(j); 
    //                         }
    //                     });
    //         });
    
    // auto task =
    //     ex::start_on(ctx, inRange)
    // |   ex::generate_indexed([](std::size_t i){ return 1.0 * i; });

    // auto task =
    //     stdexec::transfer_just(ctx.get_scheduler(), ex::view_of(inRange))
    // |   nvexec::reduce(0.0);

    // auto task =
    //     ex::start_on(ctx, inRange)
    // |   ex::reduce();

    // auto task =
    //     stdexec::transfer_just(ctx.get_scheduler(),ex::view_of(inRange))
    // |   stdexec::let_value([](auto input){
    //         return nvexec::reduce(stdexec::just(input), 0.0);
    //     });

    // auto task =
    //     ex::start_on(ctx)
    // |   ex::reduce(inRange, 0.0);
    // |   ex::reduce(inRange, 0.0);
    // |   ex::reduce(inRange, 0.0, std::plus<>{});


    // auto task = start(ctx.get_scheduler(), inRange)
    //           | gen([](std::size_t i){ return 2.0 * i; });

    // auto task = start(ctx.get_scheduler())
    //           | gen(inRange, [](std::size_t i){ return 2.0 * i; });

    // auto task = gen(ctx.get_scheduler(), inRange, [](std::size_t i){ return 2.0 * i; });


    // auto task = stdexec::transfer_just(ctx.get_scheduler(),ex::view_of(inRange))
    //           | nvexec::reduce(0.0);

    // auto task = ex::start_on(ctx.get_scheduler(),ex::view_of(inRange))
    //           | stdexec::let_value([](ex::execution_context auto, auto in){
    //                 return nvexec::reduce(stdexec::just(in), 0.0);
    //             });

    // auto task = nvexec::reduce(stdexec::just(stdexec::schedule(ctx.get_scheduler()), ex::view_of(inRange)), 0.0);

    // stdexec::sync_wait(std::move(task)).value();

    // auto result = 0.0;


    // auto task =
    //     ex::start_on(ctx, inRange)
    // |   ex::if_then_else(
    //         [](ex::compute_tag auto, auto) {
    //             return true; 
    //         },
    //         stdexec::then([](ex::compute_tag auto, auto r) {
    //             return r.back();
    //         }),
    //         stdexec::then([](ex::compute_tag auto, auto r) {
    //             return r.front();
    //         })
    //     );
    //
   
    auto task = 
            stdexec::schedule(ctx.get_scheduler())
        |   stdexec::read()
        |   stdexec::let_value([](auto sched) {
                return stdexec::on(sched, stdexec::then([]{}));
            });

    /*
    auto task =
        // ex::start_on(ctx, inRange)
        stdexec::schedule(ctx.get_scheduler())
    |   ex::if_then_else(
            []() { return true; }
            ,
            // ex::generate_indexed([](std::size_t i){ return 1.0 * i; })
            stdexec::then([](){})
            // stdexec::let_value([](auto, auto in){ 
            //     return  stdexec::just(in)
            //         |   stdexec::bulk(in.size(), [](std::size_t i, auto in){ 
            //                 in[i] = 2.0 * i; 
            //             });
            // })
            ,
            // ex::generate_indexed([](std::size_t i){ return 2.0 * i; })
            stdexec::then([](){})
            // stdexec::let_value([](auto, auto in){ 
            //     return  stdexec::just(in)
            //         |   stdexec::bulk(in.size(), [](std::size_t i, auto in){ 
            //                 in[i] = 3.0 * i; 
            //             });
            // })
        );
    */

    stdexec::sync_wait(std::move(task)).value();
    // auto [result] = stdexec::sync_wait(std::move(task)).value();


#ifdef USE_GPU
    // printf("res: %f\n", result);
    printf("res: ");
    for (auto x : inRange) { printf("%f ", x); } printf("\n");
    // printf("res: %f\n", result);
#else
    fmt::print("in:  {}\n", inRange);
    // fmt::print("res: {}\n", result);
#endif
}



// int main (int argc, char* argv[])
// {
//     int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;
//
//     nvexec::stream_context ctx; 
//        
//     double* inout = nullptr;
//     cudaMallocManaged(&inout, nelems*sizeof(double));
//
//     // runs fine as expected without the 'let_value' in between the 2 bulks
//     // otherwise causes an 'illegal memory access' cuda error on the 2nd bulk op
//     auto task =
//         stdexec::transfer_just(ctx.get_scheduler(),std::span<double>{inout,nelems})
//     |   stdexec::bulk(nelems,
//             [](std::size_t i, std::span<double> out){ out[i] = i; })
//     |   stdexec::let_value(
//             [](std::span<double> out){ return stdexec::just(out); })
//     |   stdexec::bulk(nelems,
//             [](std::size_t i, std::span<double> out){ out[i] = 2.0 * out[i]; });
//     
//     stdexec::sync_wait(std::move(task)).value();
//
//     for (int i = 0; i < nelems; ++i) { std::cout << inout[i] << ' '; }
//     std::cout << '\n';
//     
//     cudaFree(inout);
// }



// void demo_any_sender_of (stdexec::scheduler auto sched)
// {
//     any_sender_of<stdexec::set_value_t(std::span<value_t>),
//                   stdexec::set_stopped_t(),
//                   stdexec::set_error_t(std::exception_ptr)> 
//         task0 = with_elementst



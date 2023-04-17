
#include "execution.hpp"
#include "timer.hpp"
#include "algorithms.hpp"
#include "vecmath.hpp"

#include <fmt/ranges.h>

#include <vector>
#include <iostream>
#include <random>
// #include <mutex>




//-----------------------------------------------------------------------------
void fill_random (std::span<double> v, auto& urng)
{
    auto distr = std::uniform_real_distribution<double>{-1.0,+1.0};
    std::ranges::generate(v, [&]{ return distr(urng); });
}




//-----------------------------------------------------------------------------
void fill_random (std::span<vec3d> v, auto& urng)
{
    auto distr = std::uniform_real_distribution<double>{-1.0,+1.0};

    std::ranges::generate(v, [&]{ 
        return vec3d{distr(urng), distr(urng), distr(urng)}; });
}




//-----------------------------------------------------------------------------
// void test_scan ()
// {
//     fmt::print("TEST: scan\n");
//     exec::static_thread_pool tpool (4);
//     stdexec::scheduler auto sched = tpool.get_scheduler();
//
//     std::vector<double> in (100);
//     std::vector<double> out (in.size());
//     std::fill(begin(in), end(in), 1.0);
//
//     stdexec::sender auto scan = inclusive_scan(sched, in, out, 0.0, 100);
//
//     fmt::print("output: {}\n", out);
// }




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
void test_bulk ()
{
    fmt::print("TEST: bulk\n");

    auto const maxThreads = std::thread::hardware_concurrency();

    exec::static_thread_pool tpool (maxThreads);
    stdexec::scheduler auto sched = tpool.get_scheduler();

    auto task = stdexec::schedule(sched)
        |   stdexec::bulk(maxThreads, [](auto id){
                fmt::print("Hello from {}\n", id);
            });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
void test_matrix_nn_sweep (std::size_t nrows, std::size_t ncols, int maxThreads)
{
    fmt::print("TEST: matrix NN sweep\n");

    exec::static_thread_pool tpool (maxThreads);
    stdexec::scheduler auto sched = tpool.get_scheduler();

    std::vector<double> data (nrows*ncols,0.0);
    double const borderValue = 1.0;

    bool flip = false;
    for (auto& x : data) {
        x = flip ? 1 : 0;
        flip = not flip;
    }

    span2d matrix {data.data(),nrows,ncols};

    am::timer time;
    time.start();

    transform_matrix_nearest_neigbors(sched, matrix, borderValue, maxThreads,
    [](            double up,
        double le, double mi, double re,
                    double dn )
    {
        return 0.25 * (0.25*up + 0.25*le + 3.0*mi + 0.25*re + 0.25*dn);
    });

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

    std::vector<double> data (nrows*ncols,1.0);
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
    std::vector<value_t> input1 (nelems);
    std::vector<value_t> input2 (nelems);
    std::vector<value_t> output (nelems);
    std::mt19937_64 urng;
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
    fmt::print("sum:  {} {} {}\n", sum.x, sum.y, sum.z);
    fmt::print("time: {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
void test_stdpar (std::size_t nelems)
{
    fmt::print("TEST: stdpar\n");

    using value_t = vec3d;
    std::vector<value_t> input1 (nelems);
    std::vector<value_t> input2 (nelems);
    std::vector<value_t> output (nelems);
    std::mt19937_64 urng;
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
    fmt::print("sum:  {} {} {}\n", sum.x, sum.y, sum.z);
    fmt::print("time: {} ms\n", time.milliseconds());
}




//-----------------------------------------------------------------------------
// void test_zip_reduce (std::size_t nelems, int maxThreads)
// {
//     fmt::print("TEST: zip_reduce\n");
//
//     std::vector<double> input1 (nelems, 0.0);
//     std::iota(input1.begin(), input1.end(), 1.0);
//
//     std::vector<double> input2 (nelems, 1.0);
//     // std::iota(input2.begin(), input2.end(), 1.0);
//
//     double const initValue = 0.0;
//
//     am::timer time;
//     time.start();
//     auto task = 
//         zip_reduce(ctx, input1, input2, initValue, maxThreads,
//         [](double sofar, double value1, double value2)
//         {
//             return sofar + value1 + value2; 
//         });
//
//     auto const result = stdexec::sync_wait(task).value();
//     time.stop();
//
//     fmt::print("result:  {}\n", result);
//     fmt::print("time: {} ms\n", time.milliseconds());
// }




//-----------------------------------------------------------------------------
void test_for_each (std::size_t nelems, int maxThreads)
{
    fmt::print("TEST: for_each\n");

    // using value_t = double;
    using value_t = vec3d;
    std::vector<value_t> input1 (nelems);
    std::vector<value_t> input2 (nelems);
    std::vector<value_t> output (nelems);
    std::mt19937_64 urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    for (auto threads : {16,8,4,2,1}) {
        if (threads <= maxThreads) {
            Compute_Resource cr {threads};

            am::timer time;
            time.start();

            for_each(cr, am::index_range{0,nelems},
                [&]( std::size_t i) {
                     output[i] = cross(input1[i],input2[i]);  
                });

            time.stop();

            auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
            fmt::print("sum:     {} {} {}\n", sum.x, sum.y, sum.z);
            fmt::print("threads: {}\n", threads);
            fmt::print("time:    {} ms\n", time.milliseconds());
        }
    }
}




//-----------------------------------------------------------------------------
void test_for_each_grid_index (
    std::size_t na, std::size_t nb, std::size_t nc, std::size_t nd, std::size_t ne)
{
    fmt::print("TEST: for_each_grid_index\n");

    auto const maxThreads = std::thread::hardware_concurrency();

    // using value_t = double;
    using value_t = vec3d;
    auto const nelems = na * nb * nc * nd * ne;
    std::vector<value_t> input1 (nelems);
    std::vector<value_t> input2 (nelems);
    std::vector<value_t> output (nelems);
    std::mt19937_64 urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready: {}\n", nelems);

    // auto const grid = am::index_grid{am::index5{na,nb,nc,nd,ne}};

    for (auto threads : {16,8,4,2,1}) {
        if (threads <= maxThreads) {
            Compute_Resource cr {threads};

            am::timer time;
            time.start();

            for_each_grid_index(cr, std::array{na,nb,nc,nd,ne},
                [&](auto const idx) {
                    auto const i = idx[0] + idx[1]*na + idx[2]*(na*nb)
                                 + idx[3]*(na*nb*nc) + idx[4]*(na*nb*nc*nd); 

                    output[i] = cross(input1[i],input2[i]);
                });

            time.stop();

            auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
            fmt::print("sum:     {} {} {}\n", sum.x, sum.y, sum.z);
            fmt::print("threads: {}\n", threads);
            fmt::print("time:    {} ms\n", time.milliseconds());
        }
    }
}




//-----------------------------------------------------------------------------
void test_zip_transform (std::size_t nelems, int maxThreads)
{
    fmt::print("TEST: zip_transform\n");

    // using value_t = double;
    using value_t = vec3d;
    std::vector<value_t> input1 (nelems);
    std::vector<value_t> input2 (nelems);
    std::vector<value_t> output (nelems);
    std::mt19937_64 urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");

    for (auto threads : {16,8,4,2,1}) {
        if (threads <= maxThreads) {
            Compute_Resource cr {threads};

            am::timer time;
            time.start();

            zip_transform(cr, input1, input2, output,
                [](vec3d const& in1, vec3d const& in2) {
                    return cross(in1, in2) ;
                });

            time.stop();

            auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
            fmt::print("sum:     {} {} {}\n", sum.x, sum.y, sum.z);
            fmt::print("threads: {}\n", threads);
            fmt::print("time:    {} ms\n", time.milliseconds());
        }
    }

}




//-----------------------------------------------------------------------------
void test_generate_indexed (std::size_t nelems, int maxThreads)
{
    fmt::print("TEST: generate_indexed\n");

    // using value_t = double;
    using value_t = vec3d;
    std::vector<value_t> input1 (nelems);
    std::vector<value_t> input2 (nelems);
    std::vector<value_t> output (nelems);
    std::mt19937_64 urng;
    fill_random(input1, urng);
    fill_random(input2, urng);
    fmt::print("input ready\n");
 
    // any_sender_of<stdexec::set_value_t(std::span<value_t>),
    //               stdexec::set_stopped_t(),
    //               stdexec::set_error_t(std::exception_ptr)> 
    //     task0 = with_elements(sched, output);

    for (auto threads : {16,8,4,2,1}) {
        if (threads <= maxThreads) {
            Compute_Resource cr {threads};

            am::timer time;
            time.start();

            generate_indexed(cr, output,
                [&](std::size_t i) { return cross(input1[i],input2[i]); });

            time.stop();

            auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
            fmt::print("sum:     {} {} {}\n", sum.x, sum.y, sum.z);
            fmt::print("threads: {}\n", threads);
            fmt::print("time:    {} ms\n", time.milliseconds());
        }
    }
}






//-----------------------------------------------------------------------------
int main (int argc, char* argv[])
{
    // test_value();
    // test_bulk();
    // test_scan();
    
    // int const nrows = (argc > 1) ? std::atoi(argv[1]) : 10;
    // int const ncols = (argc > 2) ? std::atoi(argv[2]) : 10;
    // int const nthreads = (argc > 3) ? std::atoi(argv[3]) : std::thread::hardware_concurrency();
    // fmt::print("threads: {}\n", nthreads);
    // test_matrix_nn_sweep(nrows, ncols, nthreads);
    
    int const nelems   = (argc > 1) ? std::atoi(argv[1]) : 10;
    int const nthreads = (argc > 2) ? std::atoi(argv[2]) : std::thread::hardware_concurrency();
    fmt::print("max. threads: {}\n", nthreads);

    // test_openmp(nelems);
    // test_stdpar(nelems);
    // test_zip_transform(nelems, nthreads);
    // test_for_each(nelems, nthreads);
    test_generate_indexed(nelems, nthreads);

    // int const na = (argc > 1) ? std::atoi(argv[1]) : 2;
    // int const nb = (argc > 2) ? std::atoi(argv[2]) : na;
    // int const nc = (argc > 3) ? std::atoi(argv[3]) : nb;
    // int const nd = (argc > 4) ? std::atoi(argv[4]) : nc;
    // int const ne = (argc > 5) ? std::atoi(argv[5]) : nd;
    // test_for_each_grid_index(na,nb,nc,nd,ne);

    // run_test();
}


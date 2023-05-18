
#include "timer.hpp"
#include "algorithms.hpp"
#include "storage.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>



//-----------------------------------------------------------------------------
void demo_matrix_nn_sweep (std::size_t nrows, std::size_t ncols, int maxThreads)
{
    fmt::print("DEMO: matrix NN sweep\n");

    exec::static_thread_pool tpool (maxThreads);
    stdexec::scheduler auto sched = tpool.get_scheduler();

    ex::vector<double> data (nrows*ncols,0.0);
    double const borderValue = 1.0;

    span2d matrix {data.data(),nrows,ncols};

    ex::timer time;
    time.start();

    ex::transform_matrix_nearest_neigbors(
        sched, matrix, borderValue, maxThreads,
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
int main (int argc, char* argv[])
{
#ifndef USE_GPU
    auto const defaultNthreads = std::thread::hardware_concurrency();
#endif

    int const nrows = (argc > 1) ? std::atoi(argv[1]) : 10;
    int const ncols = (argc > 2) ? std::atoi(argv[2]) : 10;
    int const nthreads = (argc > 3) ? std::atoi(argv[3]) : defaultNthreads;
    fmt::print("threads: {}\n", nthreads);
    demo_matrix_nn_sweep(nrows, ncols, nthreads);
}


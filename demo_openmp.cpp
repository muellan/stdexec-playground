
#include "acceleration.hpp"
#include "timer.hpp"
#include "algorithms.hpp"
#include "vecmath.hpp"
#include "storage.hpp"

#include <omp.h>

#include <fmt/format.h>
#include <fmt/ranges.h>



int main (int argc, char* argv[])
{
    int const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;

    fmt::print("OpenMP {}\n", omp_get_num_threads());

    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::vector<vec3d> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    // #pragma omp parallel for simd
    #pragma omp parallel for
    for (std::size_t i = 0; i < nelems; ++i) {
        output[i] = cross(input1[i], input2[i]);
    }

    time.stop();

    auto const sum = std::accumulate(output.begin(), output.end(), vec3d{0.,0.,0.});
    fmt::print("sum:  {}\n", sum);
    fmt::print("time: {} ms\n", time.milliseconds());
}

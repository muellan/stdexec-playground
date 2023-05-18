
#include "timer.hpp"
#include "algorithms.hpp"
#include "vecmath.hpp"
#include "storage.hpp"
#include "indices.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>



int main (int argc, char* argv[])
{
    std::size_t const nelems = (argc > 1) ? std::atoi(argv[1]) : 10;

    ex::vector<vec3d> input1 (nelems);
    ex::vector<vec3d> input2 (nelems);
    ex::vector<vec3d> output (nelems);
    ex::random_engine urng;
    ex::fill_random(input1, urng);
    ex::fill_random(input2, urng);
    fmt::print("input ready\n");

    ex::timer time;
    time.start();

    auto const idx = ex::index_range{0,nelems};
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

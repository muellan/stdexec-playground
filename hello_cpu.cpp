
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#include <fmt/format.h>

#include <cstdlib>
#include <thread>



int main (int argc, char* argv[])
{
    int const nthreads = (argc > 1)
                       ? std::atoi(argv[1])
                       : std::thread::hardware_concurrency();

    fmt::print("threads: {}\n", nthreads);

    exec::static_thread_pool pool (nthreads);

    auto task = stdexec::schedule(pool.get_scheduler()) 
              | stdexec::bulk(nthreads, [](auto id){ 
                    printf("hello from thread %d\n",id);
                });

    stdexec::sync_wait(std::move(task)).value();
}


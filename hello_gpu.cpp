
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#include <nvexec/stream_context.cuh>
// #include <nvexec/multi_gpu_context.cuh>

#include <cstdio>
#include <thread>



int main (int argc, char* argv[])
{
    nvexec::stream_context gpu;
    stdexec::scheduler auto sched = gpu.get_scheduler();

    auto task = stdexec::schedule(sched)
        |   stdexec::bulk(1024, [](auto id){
                printf("Hello from %d\n", id);
            });

    stdexec::sync_wait(std::move(task)).value();
}


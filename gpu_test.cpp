
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
void hello_gpu ()
{
    nvexec::stream_context gpu;
    stdexec::scheduler auto sch = gpu.get_scheduler();

    auto hello = stdexec::schedule(sch) 
        |   stdexec::bulk(1024, [](auto id){ 
                printf("Hello %d\n",id);
            });

    stdexec::sync_wait(hello).value();
}




//-----------------------------------------------------------------------------
void hello_gpu_context ()
{
    Compute_Resource rsc;
    Execution_Context ctx {rsc};

    auto hello = stdexec::schedule(ctx.get_scheduler()) 
        |   stdexec::bulk(ctx.resource_shape().threads, [](auto id){ 
                printf("Hello %d\n",id);
            });

    stdexec::sync_wait(hello).value();
}






//-----------------------------------------------------------------------------
int main ()
{
    // hello_gpu();
    hello_gpu_context();

}


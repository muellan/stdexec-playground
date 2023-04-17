#ifndef STDEXEC_EXECUTION_HPP
#define STDEXEC_EXECUTION_HPP


#include <stdexec/execution.hpp>

#ifdef USE_GPU
    #include <nvexec/stream_context.cuh>
    #include <nvexec/multi_gpu_context.cuh>
#else
    #include <exec/static_thread_pool.hpp>
#endif
// #include <exec/any_sender_of.hpp>

#include <thread>                   
#include <exception>                   



struct Void_Schedule {};

struct Void_Context {
    static Void_Schedule get_scheduler() noexcept { return Void_Schedule{}; }
};



struct Execution_Shape
{
    int devices = 1;
    int blocks  = 1;
    int threads = 1;
    int grid_dim[3]  {1,1,1};
    int block_dim[3] {1,1,1};
};



#ifdef NO_STDEXEC

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource (int = 0) {}

private:
    Execution_Shape shape_;
    Void_Context thread_pool_;
    Void_Context stream_context_;
};

#else   // NO_STDEXEC

#ifdef USE_GPU
class Compute_Resource {
public:
    friend class Execution_Context;

    explicit
    Compute_Resource ():
        // TODO query actual device capabilities
        shape_{
            .devices = 1,
            .blocks  = 1,
            .threads = 1024,
            .grid_dim  {1,1,1},
            .block_dim {1,1,1}
        }
    {}

private:
    Execution_Shape shape_;
    nvexec::stream_context stream_context_; 
};

#else  // USE_GPU
       
class Compute_Resource {
public:
    friend class Execution_Context;

    explicit
    Compute_Resource (int num_threads = std::thread::hardware_concurrency()):
        shape_{ .threads = num_threads },
        thread_pool_(num_threads)
    {}

private:
    Execution_Shape shape_;
    exec::static_thread_pool thread_pool_;
};
#endif  // USE_GPU
#endif  // NO_STDEXEC




/* 
 * Reference to compute resources that can be stored / passed by value
 */
class Execution_Context
{
public:
    Execution_Context () = default;

    Execution_Context (Compute_Resource& res) noexcept: res_{&res} {}


    [[nodiscard]]
    bool is_engaged () const noexcept {
        return static_cast<bool>(res_);
    }

    [[nodiscard]]
    auto get_scheduler () { 
        if (not is_engaged()) {
            throw std::runtime_error{"not engaged"};
        }
#ifdef USE_GPU
        return res_->stream_context_.get_scheduler();
#else
        return res_->thread_pool_.get_scheduler();
#endif
    }

    [[nodiscard]]
    Execution_Shape resource_shape () const noexcept { 
        return static_cast<bool>(res_) ? res_->shape_ : Execution_Shape{};
    }

            
private:
    Compute_Resource* res_ = nullptr;

};


#endif


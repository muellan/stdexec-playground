#ifndef STDEXEC_EXECUTION_HPP
#define STDEXEC_EXECUTION_HPP


#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#ifdef USE_GPU
    #include <nvexec/stream_context.cuh>
    #include <nvexec/multi_gpu_context.cuh>
#endif
// #include <exec/any_sender_of.hpp>

#include <cstdint>                   
#include <thread>                   
#include <exception>                   



struct Void_Schedule {};

struct Void_Context {
    static Void_Schedule get_scheduler() noexcept { return Void_Schedule{}; }
};



#ifdef NO_STDEXEC

struct Resource_Shape
{
    inline static constexpr int devices = 1;
    inline static constexpr int grid_dims[3]  {1,1,1};
    inline static constexpr int block_dims[3] {1,1,1};
    inline static constexpr int block_threads = 1;
    inline static constexpr int warp_threads  = 1;
    inline static constexpr int threads = 1;
};

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource (int = 0) {}

private:
    Resource_Shape shape_;
    Void_Context thread_pool_;
    Void_Context stream_context_;
};

#else   // NO_STDEXEC


#ifdef USE_GPU

struct Resource_Shape
{
    int devices = 1;
    inline static constexpr int grid_dims[3]  {2147483647,65536,65536};
    inline static constexpr int block_dims[3] {1024,1024,64};
    inline static constexpr int block_threads = 1024;
    inline static constexpr int warp_threads  = 32;
    inline static constexpr std::int64_t threads = 2147483647L*65536L*65536L;
};  

class Compute_Resource {
public:
    friend class Execution_Context;

    explicit
    Compute_Resource ():
        shape_{ .devices = 1 }
    {
#ifdef USE_MULTI_GPU
        cudaGetDeviceCount(&shape_.devices);
#endif
    }

private:
    Resource_Shape shape_;
#ifdef USE_MULTI_GPU
    nvexec::multi_gpu_stream_context stream_context_;
#else
    nvexec::stream_context stream_context_; 
#endif
};
       
#else  // USE_GPU

struct Resource_Shape
{
    inline static constexpr int devices       = 1;
    inline static constexpr int grid_dims[3]  {1,1,1};
    inline static constexpr int block_dims[3] {1,1,1};
    inline static constexpr int block_threads = 1;
    inline static constexpr int warp_threads  = 1;
    int threads = 1;
};

class Compute_Resource {
public:
    friend class Execution_Context;

    explicit
    Compute_Resource (int num_threads = std::thread::hardware_concurrency()):
        shape_{ .threads = num_threads },
        thread_pool_(num_threads)
    {}

private:
    Resource_Shape shape_;
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
    Resource_Shape resource_shape () const noexcept { 
        return static_cast<bool>(res_) ? res_->shape_ : Resource_Shape{};
    }

            
private:
    Compute_Resource* res_ = nullptr;
};


#endif


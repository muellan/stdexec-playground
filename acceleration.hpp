#ifndef STDEXEC_ACCELERATION_HPP
#define STDEXEC_ACCELERATION_HPP


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


namespace ex {


#ifdef USE_GPU
    #define HOSTDEVICEQUALIFIER  __host__ __device__
#else
    #define HOSTDEVICEQUALIFIER
#endif


#ifdef USE_GPU

struct resource_shape
{
    int devices = 1;
    inline static constexpr std::int64_t threads = 2147483647L*65536L*65536L;
};

class acceleration_resource {
public:
    friend class acceleration_context;

    explicit
    acceleration_resource ():
        shape_{ .devices = 1 }
    {
#ifdef USE_MULTI_GPU
        cudaGetDeviceCount(&shape_.devices);
#endif
    }

private:
    resource_shape shape_;
#ifdef USE_MULTI_GPU
    nvexec::multi_gpu_stream_context resource_;
#else
    nvexec::stream_context resource_;
#endif
};

#else  // don't use GPU

struct resource_shape
{
    inline static constexpr int devices = 1;
    int threads = 1;
};

class acceleration_resource {
public:
    friend class acceleration_context;

    explicit
    acceleration_resource (int num_threads = std::thread::hardware_concurrency()):
        shape_{ .threads = num_threads },
        resource_(num_threads)
    {}

private:
    resource_shape shape_;
    exec::static_thread_pool resource_;
};

#endif  // USE_GPU




/*
 * Reference to compute resources that can be stored / passed by value
 */
class acceleration_context
{
public:
    acceleration_context () = default;

    acceleration_context (acceleration_resource& res) noexcept: res_{&res} {}

    [[nodiscard]]
    auto get_scheduler () {
        return res_->resource_.get_scheduler();
    }

    [[nodiscard]]
    auto get_cpu_scheduler () {
        return res_->resource_.get_scheduler();
    }

    [[nodiscard]]
    resource_shape shape () const noexcept {
        return static_cast<bool>(res_) ? res_->shape_ : resource_shape{};
    }


private:
    acceleration_resource* res_ = nullptr;
};


}  // namespace ex

#endif


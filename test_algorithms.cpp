#include "acceleration.hpp"
#include "algorithms.hpp"
#include "vecmath.hpp"
// #include "timer.hpp"

#ifdef USE_GPU
    #include <thrust/universal_vector.h>
#endif

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cstdio>
#include <array>
#include <vector>
#include <iostream>
#include <numeric>
#include <random>


using Catch::WithinRel;


#ifdef USE_GPU
    using context_t = nvexec::stream_context;

    template <class T>
    using vector_t  = thrust::universal_vector<T>;
#else
    using context_t = exec::static_thread_pool;

    template <class T>
    using vector_t  = std::vector<T>;
#endif


inline constexpr int    nelems  = 1000;
inline constexpr double epsilon = 1e-7;



//-----------------------------------------------------------------------------
TEST_CASE( "start_on" )
{
    vector_t<double> dblRange (nelems);

    // CPU or GPU resource depending on build type
    ex::acceleration_resource resource;

    SECTION( "start on ex::acceleration_resource" ) {
        auto task = ex::start_on(resource);
        stdexec::sync_wait(std::move(task)).value();
    }

    SECTION( "start on ex::acceleration_context" ) {
        auto task = ex::start_on(ex::acceleration_context{resource});
        stdexec::sync_wait(std::move(task)).value();
    }

#ifdef USE_GPU
    // explicit GPU contexts
    nvexec::stream_context gpuStream;

    SECTION( "start on thread pool" ) {
        auto task = ex::start_on(gpuStream);
        stdexec::sync_wait(std::move(task)).value();
    }

    SECTION( "start on thread pool" ) {
        auto task = ex::start_on(gpuStream.get_scheduler());
        stdexec::sync_wait(std::move(task)).value();
    }

#else
    // explicit CPU contexts
    exec::static_thread_pool threadPool;

    SECTION( "start on thread pool" ) {
        auto task = ex::start_on(threadPool);
        stdexec::sync_wait(std::move(task)).value();
    }

    SECTION( "start on thread pool scheduler" ) {
        auto task = ex::start_on(threadPool.get_scheduler());
        stdexec::sync_wait(std::move(task)).value();
    }

    SECTION( "start on thread pool scheduler" ) {
        auto task = ex::start_on(threadPool.get_scheduler(),
                                 ex::cpu_tag{ .threads = 8 });

        stdexec::sync_wait(std::move(task)).value();
    }

    exec::single_thread_context singleThread;

    SECTION( "start on single thread" ) {
        auto task = ex::start_on(singleThread);
        stdexec::sync_wait(std::move(task)).value();
    }

    SECTION( "start on single thread scheduler" ) {
        auto task = ex::start_on(singleThread.get_scheduler());
        stdexec::sync_wait(std::move(task)).value();
    }

    SECTION( "start on single thread scheduler" ) {
        auto task = ex::start_on(singleThread.get_scheduler());

        stdexec::sync_wait(std::move(task)).value();
    }
#endif
}




//-----------------------------------------------------------------------------
TEST_CASE( "for_each" )
{
    context_t ctx;

    vector_t<double> dblRange1 (nelems);

    SECTION( "put 1 range in chain" ) {
        auto task =
            ex::start_on(ctx)
        |   ex::for_each(dblRange1, [](double& x){ x = 123.4; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems-1], WithinRel(123.4, epsilon));
    }

    SECTION( "work on 1 range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange1)
        |   ex::for_each([](double& x){ x = 123.4; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems-1], WithinRel(123.4, epsilon));
    }

    vector_t<double> dblRange2 (nelems);

    SECTION( "put 2 range in chain" ) {
        auto task =
            ex::start_on(ctx)
        |   ex::for_each(dblRange1, dblRange2,
            [](double& x, double& y){
                x = 123.4;
                y = 987.6;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems-1], WithinRel(123.4, epsilon));

        REQUIRE_THAT(dblRange2[0],        WithinRel(987.6, epsilon));
        REQUIRE_THAT(dblRange2[nelems/2], WithinRel(987.6, epsilon));
        REQUIRE_THAT(dblRange2[nelems-1], WithinRel(987.6, epsilon));
    }

    SECTION( "work on 2 range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange1, dblRange2)
        |   ex::for_each(
            [](double& x, double& y){
                x = 123.4;
                y = 987.6;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems-1], WithinRel(123.4, epsilon));

        REQUIRE_THAT(dblRange2[0],        WithinRel(987.6, epsilon));
        REQUIRE_THAT(dblRange2[nelems/2], WithinRel(987.6, epsilon));
        REQUIRE_THAT(dblRange2[nelems-1], WithinRel(987.6, epsilon));
    }

    vector_t<double> dblRange3 (nelems);

    SECTION( "work on 3 range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange1, dblRange2, dblRange3)
        |   ex::for_each(
            [](double& x, double& y, double& z){
                x = 123.4;
                y = 456.7;
                z = 987.6;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange1[nelems-1], WithinRel(123.4, epsilon));

        REQUIRE_THAT(dblRange2[0],        WithinRel(456.7, epsilon));
        REQUIRE_THAT(dblRange2[nelems/2], WithinRel(456.7, epsilon));
        REQUIRE_THAT(dblRange2[nelems-1], WithinRel(456.7, epsilon));

        REQUIRE_THAT(dblRange3[0],        WithinRel(987.6, epsilon));
        REQUIRE_THAT(dblRange3[nelems/2], WithinRel(987.6, epsilon));
        REQUIRE_THAT(dblRange3[nelems-1], WithinRel(987.6, epsilon));
    }

    SECTION( "minimum size" ) {
        vector_t<int> intRange1 (15, 0);
        vector_t<int> intRange2 (10, 0);
        vector_t<int> intRange3 (20, 0);

        auto task =
            ex::start_on(ctx, intRange1, intRange2, intRange3)
        |   ex::for_each(
            [](int& x, int& y, int& z){
                x = 1;
                y = 2;
                z = 3;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(intRange1[  0], WithinRel( 1, epsilon));
        REQUIRE_THAT(intRange1[  1], WithinRel( 1, epsilon));
        REQUIRE_THAT(intRange1[  9], WithinRel( 1, epsilon));
        REQUIRE_THAT(intRange1[ 14], WithinRel( 0, epsilon));

        REQUIRE_THAT(intRange2[  0], WithinRel( 2, epsilon));
        REQUIRE_THAT(intRange2[  1], WithinRel( 2, epsilon));
        REQUIRE_THAT(intRange2[  9], WithinRel( 2, epsilon));

        REQUIRE_THAT(intRange3[  0], WithinRel( 3, epsilon));
        REQUIRE_THAT(intRange3[  1], WithinRel( 3, epsilon));
        REQUIRE_THAT(intRange3[  9], WithinRel( 3, epsilon));
        REQUIRE_THAT(intRange3[ 19], WithinRel( 0, epsilon));
    }

}




//-----------------------------------------------------------------------------
TEST_CASE( "for_each_indexed" )
{
    context_t ctx;

    vector_t<double> dblRange1 (nelems);

    SECTION( "put 1 range in chain" ) {
        auto task =
            ex::start_on(ctx)
        |   ex::for_each_indexed(dblRange1,
                [](std::size_t i, double& x){ x = 1.0 * i; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange1[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange1[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange1[100], WithinRel(100.0, epsilon));
    }

    SECTION( "work on 1 range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange1)
        |   ex::for_each_indexed(
                [](std::size_t i, double& x){ x = 1.0 * i; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange1[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange1[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange1[100], WithinRel(100.0, epsilon));
    }

    vector_t<double> dblRange2 (nelems);

    SECTION( "put 2 range in chain" ) {
        auto task =
            ex::start_on(ctx)
        |   ex::for_each_indexed(dblRange1, dblRange2,
            [](std::size_t i, double& x, double& y){
                x = 1.0 * i;
                y = 2.0 * i;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange1[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange1[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange1[100], WithinRel(100.0, epsilon));

        REQUIRE_THAT(dblRange2[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange2[  1], WithinRel(  2.0, epsilon));
        REQUIRE_THAT(dblRange2[ 10], WithinRel( 20.0, epsilon));
        REQUIRE_THAT(dblRange2[100], WithinRel(200.0, epsilon));
    }

    SECTION( "work on 2 range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange1, dblRange2)
        |   ex::for_each_indexed(
            [](std::size_t i, double& x, double& y){
                x = 1.0 * i;
                y = 2.0 * i;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange1[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange1[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange1[100], WithinRel(100.0, epsilon));

        REQUIRE_THAT(dblRange2[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange2[  1], WithinRel(  2.0, epsilon));
        REQUIRE_THAT(dblRange2[ 10], WithinRel( 20.0, epsilon));
        REQUIRE_THAT(dblRange2[100], WithinRel(200.0, epsilon));
    }

    vector_t<double> dblRange3 (nelems);

    SECTION( "work on 3 range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange1, dblRange2, dblRange3)
        |   ex::for_each_indexed(
            [](std::size_t i, double& x, double& y, double& z){
                x = 1.0 * i;
                y = 2.0 * i;
                z = 3.0 * i;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange1[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange1[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange1[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange1[100], WithinRel(100.0, epsilon));

        REQUIRE_THAT(dblRange2[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange2[  1], WithinRel(  2.0, epsilon));
        REQUIRE_THAT(dblRange2[ 10], WithinRel( 20.0, epsilon));
        REQUIRE_THAT(dblRange2[100], WithinRel(200.0, epsilon));

        REQUIRE_THAT(dblRange3[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange3[  1], WithinRel(  3.0, epsilon));
        REQUIRE_THAT(dblRange3[ 10], WithinRel( 30.0, epsilon));
        REQUIRE_THAT(dblRange3[100], WithinRel(300.0, epsilon));
    }

    SECTION( "minimum size" ) {
        vector_t<int> intRange1 (15, 0);
        vector_t<int> intRange2 (10, 0);
        vector_t<int> intRange3 (20, 0);

        auto task =
            ex::start_on(ctx, intRange1, intRange2, intRange3)
        |   ex::for_each_indexed(
            [](std::size_t i, int& x, int& y, int& z){
                x = 1 * i;
                y = 2 * i;
                z = 3 * i;
            });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(intRange1[  0], WithinRel( 0, epsilon));
        REQUIRE_THAT(intRange1[  1], WithinRel( 1, epsilon));
        REQUIRE_THAT(intRange1[  9], WithinRel( 9, epsilon));
        REQUIRE_THAT(intRange1[ 14], WithinRel( 0, epsilon));

        REQUIRE_THAT(intRange2[  0], WithinRel( 0, epsilon));
        REQUIRE_THAT(intRange2[  1], WithinRel( 2, epsilon));
        REQUIRE_THAT(intRange2[  9], WithinRel(18, epsilon));

        REQUIRE_THAT(intRange3[  0], WithinRel( 0, epsilon));
        REQUIRE_THAT(intRange3[  1], WithinRel( 3, epsilon));
        REQUIRE_THAT(intRange3[  9], WithinRel(27, epsilon));
        REQUIRE_THAT(intRange3[ 19], WithinRel( 0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "fill" )
{
    context_t ctx;

    SECTION( "double" ) {
        vector_t<double> dblRange (nelems);

        SECTION( "put range in chain" ) {
            auto task =
                ex::start_on(ctx)
            |   ex::fill(dblRange, 123.4);

            stdexec::sync_wait(std::move(task)).value();

            REQUIRE_THAT(dblRange[0],        WithinRel(123.4, epsilon));
            REQUIRE_THAT(dblRange[nelems/2], WithinRel(123.4, epsilon));
            REQUIRE_THAT(dblRange[nelems-1], WithinRel(123.4, epsilon));
        }

        SECTION( "work on range (view) already in chain" ) {
            auto task =
                ex::start_on(ctx, dblRange)
            |   ex::fill(123.4);

            stdexec::sync_wait(std::move(task)).value();

            REQUIRE_THAT(dblRange[0],        WithinRel(123.4, epsilon));
            REQUIRE_THAT(dblRange[nelems/2], WithinRel(123.4, epsilon));
            REQUIRE_THAT(dblRange[nelems-1], WithinRel(123.4, epsilon));
        }
    }

    SECTION( "vec3d" ) {
        vector_t<vec3d> vecRange (nelems);

        auto task =
            ex::start_on(ctx)
        |   ex::fill(vecRange, vec3d{1.0,2.0,3.0});

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(vecRange[0].x,        WithinRel(1.0, epsilon));
        REQUIRE_THAT(vecRange[0].y,        WithinRel(2.0, epsilon));
        REQUIRE_THAT(vecRange[0].z,        WithinRel(3.0, epsilon));

        REQUIRE_THAT(vecRange[nelems/2].x, WithinRel(1.0, epsilon));
        REQUIRE_THAT(vecRange[nelems/2].y, WithinRel(2.0, epsilon));
        REQUIRE_THAT(vecRange[nelems/2].z, WithinRel(3.0, epsilon));

        REQUIRE_THAT(vecRange[nelems-1].x, WithinRel(1.0, epsilon));
        REQUIRE_THAT(vecRange[nelems-1].y, WithinRel(2.0, epsilon));
        REQUIRE_THAT(vecRange[nelems-1].z, WithinRel(3.0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "generate" )
{
    context_t ctx;

    vector_t<double> dblRange (nelems);

    SECTION( "put range in chain" ) {
        auto task =
            ex::start_on(ctx)
        |   ex::generate(dblRange, [](){ return 123.4; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange[nelems-1], WithinRel(123.4, epsilon));
    }

    SECTION( "work on range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange)
        |   ex::generate([](){ return 123.4; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange[0],        WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange[nelems/2], WithinRel(123.4, epsilon));
        REQUIRE_THAT(dblRange[nelems-1], WithinRel(123.4, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "generate_indexed" )
{
    context_t ctx;

    vector_t<double> dblRange (nelems);

    SECTION( "put range in chain" ) {
        auto task =
            ex::start_on(ctx)
        |   ex::generate_indexed(dblRange, [](std::size_t i){ return 1.0 * i; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange[100], WithinRel(100.0, epsilon));
        REQUIRE_THAT(dblRange[999], WithinRel(999.0, epsilon));
    }

    SECTION( "work on range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange)
        |   ex::generate_indexed([](std::size_t i){ return 1.0 * i; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange[  0], WithinRel(  0.0, epsilon));
        REQUIRE_THAT(dblRange[  1], WithinRel(  1.0, epsilon));
        REQUIRE_THAT(dblRange[ 10], WithinRel( 10.0, epsilon));
        REQUIRE_THAT(dblRange[100], WithinRel(100.0, epsilon));
        REQUIRE_THAT(dblRange[999], WithinRel(999.0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "apply" )
{
    context_t ctx;

    vector_t<double> dblRange (10, 0);

    SECTION( "work on range (view) already in chain" ) {
        auto task =
            ex::start_on(ctx, dblRange)
        |   ex::apply([](auto r){
                r[0] = 1.0;
                r[1] = 2.0;
                r[2] = 3.0;
            });
        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(dblRange[0], WithinRel(1.0, epsilon));
        REQUIRE_THAT(dblRange[1], WithinRel(2.0, epsilon));
        REQUIRE_THAT(dblRange[2], WithinRel(3.0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "apply_to_xxx" )
{
    context_t ctx;

    vector_t<int> intRange1 (10, 0);
    vector_t<int> intRange2 (10, 0);
    vector_t<int> intRange3 (10, 0);

    SECTION( "apply_to_first" ) {
        auto task = ex::start_on(ctx, intRange1, intRange2, intRange3)
                  | ex::apply_to_first([](auto r){
                        r[0] = 1;
                        r[1] = 2;
                    });
        stdexec::sync_wait(std::move(task)).value();

        REQUIRE(intRange1[0] == 1);
        REQUIRE(intRange1[1] == 2);

        REQUIRE(intRange2[0] == 0);
        REQUIRE(intRange2[1] == 0);

        REQUIRE(intRange3[0] == 0);
        REQUIRE(intRange3[1] == 0);
    }

    SECTION( "apply_to_second" ) {
        auto task = ex::start_on(ctx, intRange1, intRange2, intRange3)
                  | ex::apply_to_second([](auto r){
                        r[0] = 1;
                        r[1] = 2;
                    });
        stdexec::sync_wait(std::move(task)).value();

        REQUIRE(intRange1[0] == 0);
        REQUIRE(intRange1[1] == 0);

        REQUIRE(intRange2[0] == 1);
        REQUIRE(intRange2[1] == 2);

        REQUIRE(intRange3[0] == 0);
        REQUIRE(intRange3[1] == 0);
    }

    SECTION( "apply_to_third" ) {
        auto task = ex::start_on(ctx, intRange1, intRange2, intRange3)
                  | ex::apply_to_third([](auto r){
                        r[0] = 1;
                        r[1] = 2;
                    });
        stdexec::sync_wait(std::move(task)).value();

        REQUIRE(intRange1[0] == 0);
        REQUIRE(intRange1[1] == 0);

        REQUIRE(intRange2[0] == 0);
        REQUIRE(intRange2[1] == 0);

        REQUIRE(intRange3[0] == 1);
        REQUIRE(intRange3[1] == 2);
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "transform" )
{
    context_t ctx;

    // SECTION( "vector of double; output allocated" ) {
    //     vector_t<double> inRange (nelems);
    //     std::iota(inRange.begin(), inRange.end(), 0.0);
    //
    //     auto task =
    //         ex::start_on(ctx, inRange)
    //     |   ex::transform([](double a){ return 2.0 * a; })
    //     |   ex::return_result();
    //     auto [result] = stdexec::sync_wait(std::move(task)).value();
    //
    //     REQUIRE_THAT(result[ 0], WithinRel( 0.0, epsilon));
    //     REQUIRE_THAT(result[ 1], WithinRel( 2.0, epsilon));
    //     REQUIRE_THAT(result[ 2], WithinRel( 4.0, epsilon));
    //     REQUIRE_THAT(result[10], WithinRel(20.0, epsilon));
    // }

    SECTION( "vector of double" ) {
        vector_t<double> inRange (nelems);
        std::iota(inRange.begin(), inRange.end(), 0.0);

        vector_t<double> outRange (nelems);

        auto task =
            ex::start_on(ctx, inRange)
        |   ex::transform(outRange,[](double a){ return 2.0 * a; });

        stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(outRange[ 0], WithinRel( 0.0, epsilon));
        REQUIRE_THAT(outRange[ 1], WithinRel( 2.0, epsilon));
        REQUIRE_THAT(outRange[ 2], WithinRel( 4.0, epsilon));
        REQUIRE_THAT(outRange[10], WithinRel(20.0, epsilon));
    }


    SECTION( "vector of vec3d" ) {
        vector_t<vec3d> inRange (nelems);
        vector_t<vec3d> outRange (nelems);

        auto task =
            ex::start_on(ctx, inRange)
        |   ex::generate_indexed([](std::size_t i){
                return vec3d{1.0 * i, 2.0 * i, 3.0 * i};
            })
        |   ex::transform(outRange,[](vec3d const& a){
                return vec3d{2.0*a.x, 2.0*a.y, 2.0*a.z};
            })
        |   ex::return_result();
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(outRange[0].x, WithinRel( 0.0, epsilon));
        REQUIRE_THAT(outRange[0].y, WithinRel( 0.0, epsilon));
        REQUIRE_THAT(outRange[0].z, WithinRel( 0.0, epsilon));

        REQUIRE_THAT(outRange[1].x, WithinRel( 2.0, epsilon));
        REQUIRE_THAT(outRange[1].y, WithinRel( 4.0, epsilon));
        REQUIRE_THAT(outRange[1].z, WithinRel( 6.0, epsilon));

        REQUIRE_THAT(outRange[10].x, WithinRel(20.0, epsilon));
        REQUIRE_THAT(outRange[10].y, WithinRel(40.0, epsilon));
        REQUIRE_THAT(outRange[10].z, WithinRel(60.0, epsilon));

        REQUIRE_THAT(result[0].x, WithinRel( 0.0, epsilon));
        REQUIRE_THAT(result[0].y, WithinRel( 0.0, epsilon));
        REQUIRE_THAT(result[0].z, WithinRel( 0.0, epsilon));

        REQUIRE_THAT(result[1].x, WithinRel( 2.0, epsilon));
        REQUIRE_THAT(result[1].y, WithinRel( 4.0, epsilon));
        REQUIRE_THAT(result[1].z, WithinRel( 6.0, epsilon));

        REQUIRE_THAT(result[10].x, WithinRel(20.0, epsilon));
        REQUIRE_THAT(result[10].y, WithinRel(40.0, epsilon));
        REQUIRE_THAT(result[10].z, WithinRel(60.0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "zip_transform" )
{
    context_t ctx;

    SECTION( "input & output same type" ) {

        vector_t<double> inDblRange1 (nelems);
        vector_t<double> inDblRange2 (nelems);
        vector_t<double> outDblRange (nelems);

        // SECTION( "2 ranges already in chain; output allocated" ) {
        //     auto task =
        //         ex::start_on(ctx, inDblRange1, inDblRange2)
        //     |   ex::for_each_indexed(
        //             [](std::size_t i, double& a, double& b){
        //                 a = 1.0 * i;
        //                 b = 2.0 * i;
        //             })
        //     |   ex::zip_transform([](double v1, double v2){
        //                 return v1 + v2;
        //             })
        //
        //     |   ex::return_result();
        //     auto [result] = stdexec::sync_wait(std::move(task)).value();
        //
        //     REQUIRE_THAT(result[  0], WithinRel(  0.0, epsilon));
        //     REQUIRE_THAT(result[  1], WithinRel(  3.0, epsilon));
        //     REQUIRE_THAT(result[  2], WithinRel(  6.0, epsilon));
        //     REQUIRE_THAT(result[ 10], WithinRel( 30.0, epsilon));
        //     REQUIRE_THAT(result[100], WithinRel(300.0, epsilon));
        // }

        SECTION( "2 ranges already in chain; external output" ) {
            auto task =
                ex::start_on(ctx, inDblRange1, inDblRange2)
            |   ex::for_each_indexed(
                    [](std::size_t i, double& a, double& b){
                        a = 1.0 * i;
                        b = 2.0 * i;
                    })
            |   ex::zip_transform(outDblRange, [](double v1, double v2){
                        return v1 + v2;
                    });

            stdexec::sync_wait(std::move(task)).value();

            REQUIRE_THAT(outDblRange[  0], WithinRel(  0.0, epsilon));
            REQUIRE_THAT(outDblRange[  1], WithinRel(  3.0, epsilon));
            REQUIRE_THAT(outDblRange[  2], WithinRel(  6.0, epsilon));
            REQUIRE_THAT(outDblRange[ 10], WithinRel( 30.0, epsilon));
            REQUIRE_THAT(outDblRange[100], WithinRel(300.0, epsilon));
        }


        // SECTION( "output allocated" ) {
        //     auto initTask =
        //         ex::start_on(ctx, inDblRange1, inDblRange2, outDblRange)
        //     |   ex::for_each_indexed(
        //         [](std::size_t i, double& a, double& b, double& c){
        //             a = 1.0 * i;
        //             b = 2.0 * i;
        //             c = 0.0;
        //         });
        //     stdexec::sync_wait(std::move(initTask)).value();
        //
        //
        //     auto task =
        //         ex::start_on(ctx)
        //     |   ex::zip_transform(inDblRange1, inDblRange2,
        //             [](double v1, double v2){ return v1 + v2; })
        //
        //     |   ex::return_result();
        //     auto [result] = stdexec::sync_wait(std::move(task)).value();
        //
        //     REQUIRE_THAT(result[  0], WithinRel(  0.0, epsilon));
        //     REQUIRE_THAT(result[  1], WithinRel(  3.0, epsilon));
        //     REQUIRE_THAT(result[  2], WithinRel(  6.0, epsilon));
        //     REQUIRE_THAT(result[ 10], WithinRel( 30.0, epsilon));
        //     REQUIRE_THAT(result[100], WithinRel(300.0, epsilon));
        // }


        SECTION( "external output" ) {
            auto initTask =
                ex::start_on(ctx, inDblRange1, inDblRange2, outDblRange)
            |   ex::for_each_indexed(
                [](std::size_t i, double& a, double& b, double& c){
                    a = 1.0 * i;
                    b = 2.0 * i;
                    c = 0.0;
                });
            stdexec::sync_wait(std::move(initTask)).value();


            auto task =
                ex::start_on(ctx)
            |   ex::zip_transform(inDblRange1, inDblRange2, outDblRange,
                    [](double v1, double v2){ return v1 + v2; })

            |   ex::return_result();
            stdexec::sync_wait(std::move(task)).value();

            REQUIRE_THAT(outDblRange[  0], WithinRel(  0.0, epsilon));
            REQUIRE_THAT(outDblRange[  1], WithinRel(  3.0, epsilon));
            REQUIRE_THAT(outDblRange[  2], WithinRel(  6.0, epsilon));
            REQUIRE_THAT(outDblRange[ 10], WithinRel( 30.0, epsilon));
            REQUIRE_THAT(outDblRange[100], WithinRel(300.0, epsilon));
        }
    }


    SECTION( "input & output different type" ) {

        vector_t<vec3d> inVecRange1  (nelems);
        vector_t<vec3d> inVecRange2  (nelems);
        vector_t<double> outDblRange (nelems);


        // SECTION( "2 ranges already in chain; output allocated" ) {
        //     auto task =
        //         ex::start_on(ctx, inVecRange1, inVecRange2)
        //     |   ex::for_each_indexed(
        //             [](std::size_t i, vec3d& a, vec3d& b){
        //                 a.x = 1.0 * i; a.y = 2.0 * i; a.z = 3.0 * i;
        //                 b.x = 2.0 * i; b.y = 4.0 * i; b.z = 6.0 * i;
        //             })
        //     |   ex::zip_transform(
        //             [](vec3d const& v1, vec3d const& v2) {
        //                 return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        //             })
        //
        //     |   ex::return_result();
        //     auto [result] = stdexec::sync_wait(std::move(task)).value();
        //
        //     REQUIRE_THAT(result[  0], WithinRel(     0.0, epsilon));
        //     REQUIRE_THAT(result[  1], WithinRel(    28.0, epsilon));
        //     REQUIRE_THAT(result[  2], WithinRel(   112.0, epsilon));
        //     REQUIRE_THAT(result[ 10], WithinRel(  2800.0, epsilon));
        //     REQUIRE_THAT(result[100], WithinRel(280000.0, epsilon));
        // }

        SECTION( "2 ranges already in chain; external output" ) {
            auto task =
                ex::start_on(ctx, inVecRange1, inVecRange2)
            |   ex::for_each_indexed(
                    [](std::size_t i, vec3d& a, vec3d& b){
                        a.x = 1.0 * i; a.y = 2.0 * i; a.z = 3.0 * i;
                        b.x = 2.0 * i; b.y = 4.0 * i; b.z = 6.0 * i;
                    })
            |   ex::zip_transform(outDblRange,
                    [](vec3d const& v1, vec3d const& v2) {
                        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
                    });

            stdexec::sync_wait(std::move(task)).value();

            REQUIRE_THAT(outDblRange[  0], WithinRel(     0.0, epsilon));
            REQUIRE_THAT(outDblRange[  1], WithinRel(    28.0, epsilon));
            REQUIRE_THAT(outDblRange[  2], WithinRel(   112.0, epsilon));
            REQUIRE_THAT(outDblRange[ 10], WithinRel(  2800.0, epsilon));
            REQUIRE_THAT(outDblRange[100], WithinRel(280000.0, epsilon));
        }
    }
}




#ifndef USE_GPU


//-----------------------------------------------------------------------------
TEST_CASE( "reduce" )
{
    context_t ctx;

    SECTION( "vector of double; sum" ) {
        vector_t<double> dblRange (nelems);
        std::iota(dblRange.begin(), dblRange.end(), 0.0);

        auto task =
            ex::start_on(ctx, dblRange)
        |   ex::reduce();  // sum
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(499500.0, epsilon));
    }


    SECTION( "vector of double; biased sum" ) {
        vector_t<double> dblRange (nelems);
        std::iota(dblRange.begin(), dblRange.end(), 0.0);

        auto task =
            ex::start_on(ctx, dblRange)
        |   ex::reduce(100100.0);  // biased sum
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(599600.0, epsilon));
    }


    SECTION( "vector of double; custom reduction op" ) {
        vector_t<double> dblRange (nelems);
        std::iota(dblRange.begin(), dblRange.end(), 0.0);

        auto task =
            ex::start_on(ctx, dblRange)
        |   ex::reduce(100.0, [](double acc, double x){ return acc + x; });
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(499600.0, epsilon));
    }


    SECTION( "vector of vec3d; sum" ) {
        vector_t<vec3d> vecRange (nelems);

        auto task =
            ex::start_on(ctx, vecRange)
        |   ex::generate_indexed([](std::size_t i){
                return vec3d{1.0 * i, 2.0 * i, 3.0 * i};
            })
        |   ex::reduce();  // sum
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result.x, WithinRel( 499500.0, epsilon));
        REQUIRE_THAT(result.y, WithinRel( 999000.0, epsilon));
        REQUIRE_THAT(result.z, WithinRel(1498500.0, epsilon));
    }


    SECTION( "vector of vec3d; biased sum" ) {
        vector_t<vec3d> vecRange (nelems);

        auto task =
            ex::start_on(ctx, vecRange)
        |   ex::generate_indexed([](std::size_t i){
                return vec3d{1.0 * i, 2.0 * i, 3.0 * i};
            })
        |   ex::reduce(vec3d{10.0, 100.0, 1000.0});  // sum
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result.x, WithinRel( 499510.0, epsilon));
        REQUIRE_THAT(result.y, WithinRel( 999100.0, epsilon));
        REQUIRE_THAT(result.z, WithinRel(1499500.0, epsilon));
    }


    SECTION( "vector of vec3d; custom reduction op" ) {
        vector_t<vec3d> vecRange (nelems);

        auto task =
            ex::start_on(ctx, vecRange)
        |   ex::generate_indexed([](std::size_t i){
                return vec3d{1.0 * i, 2.0 * i, 3.0 * i};
            })
        |   ex::reduce(
                vec3d{10.0, 100.0, 1000.0},
                [](vec3d const& a, vec3d const& b){
                    return vec3d{a.x + b.x, a.y + b.y, a.z + b.z};
                });
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result.x, WithinRel( 499510.0, epsilon));
        REQUIRE_THAT(result.y, WithinRel( 999100.0, epsilon));
        REQUIRE_THAT(result.z, WithinRel(1499500.0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "transform_reduce" )
{
    context_t ctx;

    SECTION( "vector of double" ) {
        vector_t<double> dblRange (nelems);

        auto task =
            // start chain on execution context or scheduler
            // attaches compute information (#threas,#devices,etc.)
            ex::start_on(ctx, dblRange)
            // async range algorithm: set range elements using index->value mapping
        |   ex::generate_indexed([](std::size_t i){ return double(i); })
            // another async range algorithm
        |   ex::transform_reduce(
                [](double a){ return 2.0 * a; },  // transformation
                [](double a, double b){ return a + b; }  // reduction
            );
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(999000.0, epsilon));
    }


    SECTION( "vector of vec3d" ) {
        vector_t<vec3d> vecRange (nelems);

        auto task =
            ex::start_on(ctx, vecRange)
        |   ex::generate_indexed([](std::size_t i){
                return vec3d{1.0 * i, 2.0 * i, 3.0 * i};
            })
        |   ex::transform_reduce(
                [](vec3d const& a){
                    return vec3d{2.0*a.x, 2.0*a.y, 2.0*a.z};
                },
                [](vec3d const& a, vec3d const& b){
                    return vec3d{a.x + b.x, a.y + b.y, a.z + b.z};
                }
            );
        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result.x, WithinRel( 999000.0, epsilon));
        REQUIRE_THAT(result.y, WithinRel(1998000.0, epsilon));
        REQUIRE_THAT(result.z, WithinRel(2997000.0, epsilon));
    }
}




//-----------------------------------------------------------------------------
TEST_CASE( "zip_reduce" )
{
    context_t ctx;

    vector_t<double> inRange1 (nelems);
    vector_t<double> inRange2 (nelems);


    SECTION( "2 ranges already in chain" ) {
        auto task =
            ex::start_on(ctx, inRange1, inRange2)
        |   ex::for_each_indexed(
                [](std::size_t i, double& a, double& b){
                    a = 1.0 * i;
                    b = 2.0 * i;
                })
        |   ex::zip_reduce(
                [](double acc, double v1, double v2){
                    return acc + v1 + v2;
                });

        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(1498500.0, epsilon));
    }


    SECTION( "put 2 ranges in chain" ) {
        // preparation
        auto initTask =
            ex::start_on(ctx, inRange1, inRange2)
        |   ex::for_each_indexed(
            [](std::size_t i, double& a, double& b){
                a = 1.0 * i;
                b = 2.0 * i;
            });
        stdexec::sync_wait(std::move(initTask)).value();

        auto task =
            ex::start_on(ctx)
        |   ex::zip_reduce(inRange1, inRange2,
                [](double acc, double v1, double v2){
                    return acc + v1 + v2;
                });

        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(1498500.0, epsilon));
    }


    vector_t<double> inRange3 (nelems);
}




//-----------------------------------------------------------------------------
TEST_CASE( "zip_transform_reduce" )
{
    context_t ctx;

    vector_t<vec3d> inVecRange1 (nelems);
    vector_t<vec3d> inVecRange2 (nelems);


    SECTION( "2 ranges already in chain" ) {
        auto task =
            ex::start_on(ctx, inVecRange1, inVecRange2)
        |   ex::for_each_indexed(
                [](std::size_t i, vec3d& a, vec3d& b){
                    a.x = 1.0 * i; a.y = 2.0 * i; a.z = 3.0 * i;
                    b.x = 2.0 * i; b.y = 4.0 * i; b.z = 6.0 * i;
                })
        |   ex::zip_transform_reduce(
                // transformation vec3d -> double
                [](vec3d const& v1, vec3d const& v2) {
                    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
                },
                // reduction
                [](double acc, double x) { return acc + x; });

        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(9319338000.0, epsilon));
    }


    SECTION( "put 2 ranges in chain" ) {
        // preparation
        auto initTask =
            ex::start_on(ctx, inVecRange1, inVecRange2)
        |   ex::for_each_indexed(
            [](std::size_t i, vec3d& a, vec3d& b){
                a.x = 1.0 * i; a.y = 2.0 * i; a.z = 3.0 * i;
                b.x = 2.0 * i; b.y = 4.0 * i; b.z = 6.0 * i;
            });
        stdexec::sync_wait(std::move(initTask)).value();

        auto task =
            ex::start_on(ctx)
        |   ex::zip_transform_reduce(inVecRange1, inVecRange2,
                [](vec3d const& v1, vec3d const& v2) {
                    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
                },
                [](double acc, double x) { return acc + x; });

        auto [result] = stdexec::sync_wait(std::move(task)).value();

        REQUIRE_THAT(result, WithinRel(9319338000.0, epsilon));
    }

}


#endif




//-----------------------------------------------------------------------------
// TEST_CASE( "if_then_else" )
// {
//     SECTION( "true branch") {
//         auto task =
//             stdexec::just(47)
//         |   ex::if_then_else(
//                 [](int i) { return i == 47; },
//                 stdexec::then( [](int) { return 1; } ),
//                 stdexec::then( [](int) { return 2; } )
//             );
//
//         auto [result] = stdexec::sync_wait(std::move(task)).value();
//
//         REQUIRE(result == 1);
//     }
//
//     SECTION( "false branch") {
//         auto task =
//             stdexec::just(1)
//         |   ex::if_then_else(
//                 [](int i) { return i == 47; },
//                 stdexec::then( [](int) { return 1; } ),
//                 stdexec::then( [](int) { return 2; } )
//             );
//
//         auto [result] = stdexec::sync_wait(std::move(task)).value();
//
//         REQUIRE(result == 2);
//     }
//
//
//     SECTION("with scheduler") {
//         context_t ctx;
//
//         vector_t<double> dblRange (100);
//         std::iota(dblRange.begin(), dblRange.end(), 0.0);
//
//         auto task =
//             ex::start_on(ctx, dblRange)
//         |   ex::if_then_else(
//                 [](ex::compute_tag auto, auto) {
//                     return true; 
//                 },
//                 stdexec::then([](ex::compute_tag auto, auto r) {
//                     return r.back();
//                 }),
//                 stdexec::then([](ex::compute_tag auto, auto r) {
//                     return r.front();
//                 })
//             );
//
//         auto [result] = stdexec::sync_wait(std::move(task)).value();
//
//         REQUIRE(result == dblRange.back());
//     }
// }




// //-----------------------------------------------------------------------------
// TEST_CASE( "for_each_grid_index" )
// {
//     // context_t ctx;
//     
//     // ex::for_each_grid_index(ctx, std::array{3,4,7,11,9},
//     //     [] (auto const idx)
//     //     {
//     //         auto const [a,b,c,d,e] = idx;
//     //         
//     //                     
//     //     });
//
// }
//
//

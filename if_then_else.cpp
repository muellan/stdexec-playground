#include <stdexec/execution.hpp>
#include <exec/variant_sender.hpp>

#include <cstdio>


template <class Fun, class... Args>
concept Predicate =
    std::invocable<Fun, Args...> &&
    std::same_as<std::invoke_result_t<Fun, Args...>, bool>;

template <class... Ts>
using just_sender_t = decltype(stdexec::just(std::declval<Ts>()...));

template <
    class Pred,
    stdexec::__sender_adaptor_closure Then,
    stdexec::__sender_adaptor_closure Else
>
auto if_then_else (Pred pred, Then then_, Else else_)
{
    return stdexec::let_value(
        [=]<class... Args>(Args&&... args) mutable
            -> exec::variant_sender<
                std::invoke_result_t<Then, just_sender_t<Args...>>,
                std::invoke_result_t<Else, just_sender_t<Args...>>>
            requires Predicate<Pred, Args&...>
        {
            if (pred(args...)) {
                return std::move(then_)(stdexec::just((Args&&)args...));
            }
            else {
                return std::move(else_)(stdexec::just((Args&&)args...));
            }
        }
    );
}


int main (int argc, char* argv[])
{
    using namespace std::literals;

    int const answer = (argc > 1) ? std::atoi(argv[1]) : 43;

    auto work =
        stdexec::just(answer)
    |   if_then_else(
            [](int i) { return i == 42; },
            stdexec::then( [](int) { return "correct"s; } ),
            stdexec::then( [](int) { return "incorrect"s; } )
        );

    auto [message] = stdexec::sync_wait(std::move(work)).value();

    std::printf("%s\n", message.c_str());
}

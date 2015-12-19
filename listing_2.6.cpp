#include <thread>
#include <utility>
#include <iostream>
#include <stdexcept>

class scoped_thread
{
    std::thread t;
public:
    explicit scoped_thread(std::thread t_):
        t(std::move(t_))
    {
      std::cout << "ctr" << std::endl;
        if(!t.joinable()) {
          std::cout << "error" << std::endl;
            throw std::out_of_range("No thread");
        }
    }
    ~scoped_thread()
    {
      std::cout << "join" << std::endl;
      std::flush(std::cout);
        t.join();
    }
    scoped_thread(scoped_thread const&)=delete;
    scoped_thread& operator=(scoped_thread const&)=delete;
};

void do_something(int& i)
{
    ++i;
}

struct func
{
    int& i;

    func(int& i_):i(i_){}

    void operator()()
    {
        for(unsigned j=0;j<1000000;++j)
        {
            do_something(i);
        }
        std::cout << "run once" << std::endl;
    }
};

void do_something_in_current_thread()
{}

void f()
{
    int some_local_state;
    scoped_thread t(std::thread(func(some_local_state)));
        
    do_something_in_current_thread();
}

int main()
{
    f();
}

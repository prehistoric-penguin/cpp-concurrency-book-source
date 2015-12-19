#include <iostream>
#include <thread>

void hello() {
  std::cout << "Hello concurrent world\n";
}

int main() {
  std::thread t(hello);
  t.join();
}

#include <thread>
void do_something(int& i) {
  ++i;
}

struct func {
  int& i;

  func(int& i_) : i(i_) { }
  void operator()() {
    for (unsigned j = 0; j < 1000000; ++j) {
      do_somethind(i);
    }
  }
};

void oops() {
  int some_local_state = 0;
  func my_func(some_local_state);

  std::thread my_thread(my_func);
  my_thread.detach();
}

int main() {
  oops();
}


#include <thread>

class thread_guard {
  std::thread& t;
 public:
  explicit thread_guard(std::thread& t_) : t(t_) {}
  ~thread_guard() {
    if (t.joinable())
      t.join();
  }
  thread_guard(thread const&) = delete;
  thread_guard& operator=(thread_guard const&) = delete;

};

void do_somethind(int& i) {
  ++i;
}

struct func {
  int& i;
  func(int& i_) : i(i_) {}
  void operator()() {
    for (unsigned j = 0; j < 100000; ++j)
      do_somethind(i);
  }
};

void do_somethind_in_current_thread() {
}

void f() {
  int some_local_state;
  func my_func(some_local_state);

  std::thread t(my_func);
  thread_guard g(t);
  
  do_something_in_current_thread();
}

int main() {
  f();
}

#include <thread>
#include <string>

void open_document_and_display_gui(std::string const& filename) {}

bool done_editing() { return true; }

enum command_type {
  open_new_document
};

struct user_command {
  command_type type;

  user_command():
      type(open_new_document) { }
};

user_command get_user_command() {
  return user_command();
}

std::string get_filename_from_user() {
  return "foo.doc";
}

void process_user_input(user_command const& cmd) { }

void edit_document(std::string const& filename) {
  open_document_and_display_gui(filename);

  while (!done_editing()) {
    user_command cmd = get_user_input();
    
    if (cmd.type == open_new_document) {
      std::string const new_name = get_file_name_from_user();
      std::thread t(edit_document, new_name);
      t.detach();
    } else {
      process_user_input(cmd);
    }
  }
}

int main() {
  edit_document("bar.doc");
}

#include <thread>
void some_function() { }
void some_other_function(int) { }
std::thread f() {
  void some_function();
  return std::thread(some_function);
}

std::thread g() {
  void some_other_function();
  std::thread t(some_other_function, 42);
  return t;
}

int main() {
  std::thread t1 = f();
  t1.join();

  std::thread t2 = g();
  t2.join();
  return 0;
}

#include <thread>
#include <utility>
#include <iostream>
#include <stdexcept>

class scoped_thread {
  std::thread t;
  
 public:
  explicit scoped_thread(std::thread t_):
      t(std::move(t_)) {
        std::cout << "ctr" << std::endl;
        if (!t.joinable()) {
          std::cout << "error" << std::endl;
          throw std::out_of_range("No thread");
        }
      }
  ~scoped_thread() {
    std::cout << "join" << std::endl;
    std::flush(std::cout);
    t.join();
  }
  scoped_thread(scoped_thread const&) = delete;
  scoped_thread operator(scoped_thread const&) = delete;
};

void do_something(int& i) {
  ++i;
}

struct func {
  int& i;

  func(int& i_) : i(i_) { }

  void operator()() {
    for (unsigned j = 0; j < 1000000; ++j) {
      do_something(i);
    }
    std::cout << "run once" << std::endl;
  }
};

void do_something_in_current_thread() { }

void f() {
  int some_local_state;
  scoped_thread t(std::thread(func(some_local_state)));

  do_somethind_in_current_thread();
}

int main() {
  f();
}

template <typename T, typename Container = std::deque<T> >
class queue {
 public:
  explicit queue(const Container&);
  explicit queue(Container&& = Container());

  queue(queue&& q);

  template <typename Alloc> explicit queue(const Alloc&);
  template <typename Alloc> queue(const Container&, const Alloc&);
  template <typename Alloc> queue(Container&&, const Alloc&);
  template <typename Alloc> queue(queue&&, const Alloc&);

  queue& operator = (queue&& q);
  void swap(queue&& q);

  bool empty() const;
  size_type size() const;

  T& front();
  const T& front() const;

  T& back();
  const T& back() const;

  void push(const T& x);
  void push(T&& x);
  void pop();
};

#include <memory>
template <typename T>
class threadsafe_queue {
 public:
  threadsafe_queue();
  threadsafe_queue(const threadsafe&);
  threadsafe_queue& operator = (const threadsafe_queue&) = delete;

  void push(T new_value);

  bool try_pop(T& value);
  std::shared_ptr<T> try_pop();

  void wait_and_pop(T& value);
  std::shared_ptr<T> wait_and_pop();

  bool empty() const;
};

#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

template <typename T>
class threadsafe_queue {
 private:
  mutable std::mutex mut;
  std::deque<T> data_queue;
  std::condition_variable data_cond;

 public:
  threadsafe_queue() { }
  threadsafe_queue(threadsafe_queue const& other) {
    std::lock_guard<std::mutex> lk(other.mut);

    data_queue = other.data_queue;
  }
  
  void push(T new_value) {
    std::lock_guard<std::mutex> lk(mut);

    data_queue.push(new_value);
    data_cond.notify_one();
  }

  void wait_and_pop(T& value) {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return !data_queue.empty(); });
    value = data_queue.front();
    data_queue.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this] { return !data_queue.empty(); });

    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();

    return res;
  }

  void try_pop(T& value) {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty())
      return false;
    value = data_queue.front();
    data_queue.pop();
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty())
      return std::shared_ptr<T>();
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();

    return res;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }
};

#include <future>
#include <iostream>

int find_the_answer_to_ltuae() {
  return 42;
}

void do_other_stuff() {
}

int main() {
  std::future<int> the_answer = std::async(find_the_answer_to_ltuae);
  do_other_stuff();

  std::cout << "The answer is " << the_answer.get() << std::endl;
}

#include <future>
void process_connections(connection_set& connections) {
  while (!done(connections)) {
    for (connection_iterator connection = connections.begin(), end = connections.end();
         connection != end;
         ++connection) {
      if (connection->has_incoming_data()) {
        data_packet data = connection->incomming();
        std::promise<payload_type>& p = connection->get_promise(data.id);

        p.set_value(data.payload);
      }
      if (connection->has_outgoing_data()) {
        outgoing_packet data = connection->top_of_outgoing_queue();
        
        connection->send(data.payload);
        data.promise.set_value(true);
      }
    }
  }
}


template <typename T>
std::list<T> sequential_quick_sort(std::list<T> input) {
  if (input.empty())
    return input;
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());

  T const& pivot = *result.begin();
  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](T const& t) { return t < pivot; });
  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divied_point);

  auto new_lower(sequential_quick_sort(std::move(lower_part)));
  auto new_higher(sequential_quick_sort(std::move(input)));

  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower);

  return result;
}

template <typename T>
std::list<T> parallel_quick_sort(std::list<T> input) {
  if (input.empty())
    return input;
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());

  T const& pivot = *result.begin();
  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](T const& t) { return t < pivot; });

  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);

  std::future<std::list<T> > new_lower(
      std::async(&parllel_quick_sort(T>, std::move(lower_part))));

  auto new_higher(parllel_quick_sort(std::move(input)));

  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower.get());

  return result;
}


#include <vector>
#include <atomic>
#include <iostream>
#include <chrono>
#include <thread>

std::vector<int> data;
std::atomic_bool data_ready(false);

void reader_thread() {
  while (!data_ready.load()) {
    std::this_thread::sleep_for(std::chrono::mimiseconds(1));
  }
  std::cout << "the answer= " << data[0] << std::endl;
}

void write_thread() {
  data.push_back(42);
  data_ready = true;
}

#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x, y;
std::atomic<int> z;

void write_x() {
  x.store(true, std::memory_order_seq_cst);
}

void write_y() {
  y.store(true, std::memory_order_seq_cst);
}

void read_x_then_y() {
  while (!x.load(std::memory_order_seq_cst))
    ;
  if (y.load(std::memory_order_seq_cst))
    ++z;
}

void read_y_then_x() {
  while (!y.load(std::memory_order_seq_cst))
    ;
  if (x.load(std::memory_order_seq_cst))
    ++z;
}

int main() {
  x = false;
  y = false;
  z = 0;

  std::thread a(write_x);
  std::thread b(write_y);

  std::thread c(read_x_then_y);
  std::thread d(read_y_then_x);

  a.join();
  b.join();
  c.join();
  d.join();

  assert(z.load() != 0);
}

#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x, y;
std::atomic<int> z;

void write_x_then_y() {
  x.store(true, std::memory_order_relaxed);
  y.store(true, std::memory_order_relaxed);
}

void read_y_then_x() {
  while (!y.load(std::memory_order_relaxed))
    ;
  if (x.load(std::memory_order_relaxed))
    ++z;
}

int main() {
  x = false;
  y = false;
  z = 0;

  std::thread a(write_x_then_y);
  std::thread b(read_y_then_x);

  a.join();
  b.join();
  assert(z.load() ! = 0);
  return 0;
}

#include <thread>
#include <atomic>
#include <iostream>

std::atomic<int> x(0), y(0), z(0);
std::atomic<bool> go(false);
unsigned const loop_ocunt = 10;

struct read_values {
  int x, y, z;
};

read_values values1[loop_count];
read_values values2[loop_count];
read_values values3[loop_count];
read_values values4[loop_count];
read_values values5[loop_count];

void increment(std::atomic<int>* var_to_inc, read_values* values) {
  while (!go)
    std::this_thread::yield();

  for (unsigned i = 0; i < loop_count; ++i) {
    values[i].x = x.load(std::memory_order_relaxed);
    values[i].y = y.load(std::memory_order_relaxed);
    values[i].z = z.load(std::memory_order_relaxed);

    var_to_inc->store(i + 1, std::memory_order_relaxed);
    std::this_thread::yield();
  }
}

void read_vals(read_values* values) {
  while (!go)
    std::this_thread::yield();

  for (unsigned i = 0; i < loop_count; ++i) {
    values[i].x = x.load(std::memory_order_relaxed);
    values[i].y = y.load(std::memory_order_relaxed);
    values[i].z = z.load(std::memory_order_relaxed);

    std::this_thread::yield();
  }
}

void print(read_values* v) {
  for (unsigned i = 0; i < loop_count; ++i) {
    if (i)
      std::cout << ",";
    std::cout << "(" << v[i].x << ","
        << v[i].y << "," << v[i].z << ")";
  }
  std::cout << std::endl;
}

int main() {
  std::thread t1(increment, &x, values1);
  std::thread t2(increment, &y, values2);
  std::thread t3(increment, Yz, values3);
  std::thread t4(read_vals, values4);
  std::thread t5(read_vals, values5);

  go = true;
  t5.join();
  t4.join();
  t3.join();
  t2.join();
  t1.join();

  print(values1);
  print(values2);
  print(values3);
  print(values4);
  print(values5);
}

#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x, y;
std::atomic<int> z;

void write_x() {
  x.store(true, std::memory_order_release);
}

void write_y() {
  y.store(true, std::memory_order_release);
}

void read_x_then_y() {
  while (!x.load(std::memory_order_acquire))
    ;
  if (y.load(std::memory_order_acquire))
    ++z;
}

void read_y_then_x() {
  while (!y.load(std::memory_order_acquire))
    ;
  if (x.load(std::memory_order_acquire))
    ++z;
}

int main() {
  x = false;
  y = false;
  z = 0;

  std::thread a(write_x);
  std::thread b(write_y);
  std::thread c(read_x_then_y);
  std::thread d(read_y_then_x);

  a.join();
  b.join();
  c.join();
  d.join();

  assert(z.load() != 0);
}

#include <atomic>
#include <thread>
#include <assert>

std::atomic<bool> x, y;
std::atomic<int> z; 

void write_x_then_y() {
  x.store(true, std::memory_order_relaxed);
  y.store(true, std::memory_order_release);
}

void read_y_then_x() {
  while (!y.load(std::memory_order_acquire))
    ;
  if (x.load(std::memory_order_relaxed))
    ++z;
}

int main() {
  x = false;
  y = false;
  z = 0;

  std::thread a(write_x_then_y);
  std::thread b(read_y_then_x);

  a.join();
  b.join();

  assert(z.load() != 0);
}

#include <atomic>
#include <thread>
#include <cassert>

std::atomic<int> data[5];
std::atomic<bool> sync1(false), sync2(false);

void thread_1() {
  data[0].store(42, std::memory_order_relaxed);
  data[1].store(97, std::memory_order_relaxed);
  data[2].store(17, std::memory_order_relaxed);
  data[3].store(-141, std::memory_order_relaxed);
  data[4].store(2003, std::memory_order_relaxed);
  
  sync1.store(true, std::memory_order_release);
}

void thread_2() {
  while (!sync1.load(std::memory_order_acquire))
    ;
  sync2.store(true, std::memory_order_release);
}

void thread_3() {
  while (!sync2.load(std::memory_order_acquire))
    ;
  assert(data[0].load(std::memory_order_relaxed) == 42);
  assert(data[1].load(std::memory_order_relaxed) == 97);
  assert(data[2].load(std::memory_order_relaxed) == 17);
  assert(data[3].load(std::memory_order_relaxed) == -141);
  assert(data[4].load(std::memory_order_relaxed) == 2003);
}

int main() {
  std::thread t1(thread_1);
  std::thread t2(thread_2);
  std::thread t3(thread_3);

  t1.join();
  t2.join();
  t3.join();
}

#include <string>
#include <thread>
#include <atomic>
#include <cassert>

struct X {
  int i;
  std::string s;
};

std::atomic<X*> p;
std::atomic<int> a;

void create_x() {
  X* x = new X;
  x->i = 42;
  x->s = "hello";

  a.store(99, std::memory_order_relaxed);
  p.store(x, std::memory_order_release);
}

void use_x() {
  X* x;
  while (!(x = p.load(std::memory_order_consume)))
    std::this_thread::sleep_for(std::chrono::miscoseconds(1));

  assert(x->i == 42);
  assert(x->s == "hello");
  assert(a.load(std::memory_order_relaxed) == 99);
}

int main() {
  std::thread t1(create_x);
  std::thread t2(use_x);

  t1.join();
  t2.join();
}

#include <atomic>
#include <thread>

std::vector<int> queue_data;
std::atomic<int> count;

void populte_queue() {
  unsigned const number_of_items = 20;
  queue_data.clear();

  for (unsigned i = 0; i < number_of_items; ++i) {
    quque_data.push_back(i);
  }
  count.store(number_if_items, std::memory_order_release);
}

void consume_queue_items() {
  while (true) {
    int item_index;
    
    if ((item_index = count.fetch_sub(1, std::memory_order_acquire)) <= 0) {
      wait_for_more_items();
      continue;
    }
    process(queue_data[item_index - 1]);
  }
}

int main() {
  std::thread a(populate_queue);
  std::thread b(consume_queue_items);
  std::thread c(consume_queue_items);

  a.join();
  b.join();
  c.join();
}

#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> x, y;
std::atomic<int> z;

void write_x_then_y() {
  x.store(true, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  y.store(true, std::memory_order_relaxed);
}

void read_y_then_x() {
  while (!y.load(std::memory_order_relaxed))
    ;
  std::atomic_thread_fence(std::memory_order_acquire);
  if (x.load(std::memory_order_relaxed))
    ++z;
}

int main() {
  x = false;
  y = false;
  z = 0;

  std::thread a(write_x_then_y);
  std::thread b(read_y_then_x);

  a.join();
  b.join();
  assert(z.load() != 0);
}

#include <atomic>
#include <thread>
#include <cassert>

bool x = false;
std::atomic<bool> y;
std::atomic<int> z;

void write_x_then_y() {
  x = true;
  std::atomic_thread_fence(std::memory_order_release);
  y.store(true, std::memory_order_relaxed);
}

void read_y_then_x() {
  while (!y.load(std::memory_order_relaxed))
    ;
  std::atomic_thread_fence(std::memory_order_acquire);
  if (x)
    ++z;
}

int main() {
  x = false;
  y = false;
  z = 0;

  std::thread a(write_x_then_y);
  std::thread b(read_y_then_x);

  a.join();
  b.join();
  assert(z.load() != 0);
}

#include <stack>
#include <mutex>
#include <memory>

struct empty_stack : std::exception {
  const char* what() const throw() {
    return "empty stack";
  }
};

template <typename T>
class threadsafe_stack {
 private:
  std::stack<T> data;
  mutable std::mutex m;

 public:
  threadsafe_stack() { }
  threadsafe_stack(const threadsafe_stack& other) {
    std::lock_guard<std::mutex> lock(other.m);
    data = other.data;
  }
  threadsafe_stack& operator=(const threadsafe_stack&) = delete;
  
  void push(T new_value) {
    std::lock_guard<std::mutex> lock(m);
    data.push(std::move(new_value));
  }

  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> lock(m);
    if (data.empty())
      throw empty_stack();

    std::shared_ptr<T> res(std::make_shared<T>(std::move(data.pop())));
    data.pop();
    return res;
  }

  void pop(T& value) {
    std::lock_guard<std::mutex> lock(m);
    if (data.empty())
      throw empty_stack();

    value = std::move(data.top());
    data.pop();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m);
    return data.empty();
  }
};

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

template <typename T>
class threadsafe_queue {
 private:
  mutable std::mutex mut;
  std::deque<T> data_queue;
  std::condition_variable data_cond;

 public:
  threadsafe_queue() { }

  void push(T new_value) {
    std::lock_guard<std::mutex> lock(mut);
    data_queue.push(std::move(new_value));
    data_cond.notify_one();
  }

  void wait_and_pop(T& value) {
    std::unique_lock<std::mutex> lock(mut);
    data_cond.wait(lock,
                   [this] { return !data_queue.empty(); });
    value = std::move(data_queue.front());
    data_queue.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lock(mut);
    data_cond.wait(lock,
                   [this] { return !data_queue.empty(); });
    std::shared_ptr<T> res(std::make_shared<T>(std::move(data_queue.front())));
    data_queue.pop();
    return res;
  }

  bool try_pop(T& value) {
    std::lock_guard<std::mutex> lock(mut);
    if (data_queue.empty())
      return false;
    value = std::move(data_queue.front());
    data_queue.pop();
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lock(mut);
    if (data_queue.empty())
      return std::shared_ptr<T>();

    std::shared_ptr<T> res(std::make_shared<T>(std::move(data_queue.front())));
    data_queue.pop();
    return res;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mut);
    return data_queue.empty();
  }
};

#include <deque>
#include <mutex>
#include <condition_variable>
#include <memory>

template <typename T>
class threadsafe_queue {
 private:
  mutable std::mutex mut;
  std::queue<std::shared_ptr<T> > data_queue;
  std::condition_variable data_cond;

 public:
  threadsafe_queue() = default;

  void wait_and_pop(T& value) {
    std::lock_guard<std::mutex> lock(mut);
    data_cond.wait(lock,
                   [this] { return !data_queue.empty(); });
    value = std::move(*data_queue.front());
    data_queue.pop();
  }

  bool try_pop(T& value) {
    std::lock_guard<std::mutex> lock(mut);
    if (data_queue.empty())
      return false;
    value = std::move(*data_queue.front());
    data_queue.pop();
    return true;
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lock(mut);
    data_cond.wait(lock,
                   [this] { return !data_queue.empty(); });
    std::shared_ptr<T> res = data_queue.front();
    data_queue.pop();

    return res;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lock(mut);
    if (data_queue.empty())
      return std::shared_ptr<T>();

    std::shared_ptr<T> res = data_queue.front();
    data_queue.pop();

    return res;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mut);
    return data_queue.empty();
  }

  void push(T new_value) {
    shared_ptr<T> data(std::make_shared<T>(std::move(new_value)));
    std::lock_guard<std::mutex> lock(mut);
    data_queue.push(data);
    data_cond.notify_one();
  }
};

#include <memory>

template <typename T>
class queue {
 private:
  struct node {
    T data;
    std::unique_ptr<node> next;

    node(T data_) : data(std::move(data_)) { }
  };

  std::unique_ptr<node> head;
  node* tail;

 public:
  queue(): tail(nullptr) { }

  queue(const queue& other) = delete;
  queue& operator=(const queue& other) = delete;

  std::shared_ptr<T> try_pop() {
    if (!head) {
      return std::shared_ptr<T>();
    }

    const std::shared_ptr<T> res(std::make_shared<T>(std::move(head->data)));
    const std::unique_ptr<node> old_head = std::move(head);
    head = std::move(old_head->next);

    return res;
  }

  void push(T new_value) {
    std::unique_ptr<node> p(new node(std::move(new_value)));
    node* const new_tail = p.get();
    if (tail) {
      tail->next = std::move(p);
    } else {
      tail = std::move(p);
    }
    tail = new_tail;
  }
};

#include <memory>

template <typename T>
class queue {
 private:
  struct node {
    std::shared_ptr<T> data;
    std::unique_ptr<T> next;
  };

  std::unique_ptr<node> head;
  node* tail;

 public:
  queue():
      head(new node), tail(head.get())
  { }

  queue(const queue& other) = delete;
  queue& operator=(const queue& other) = delete;

  std::shared_ptr<T> try_pop() {
    if (head.get() == tail)
      return shared_ptr<T>();
    std::shared_ptr<T> res(head->data);
    std::unique_ptr<node> old_head = std::move(head);
    head = std::move(old_head->next);
    return res;
  }

  void push(T new_value) {
    std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_data)));
    std::unique_ptr<node> p(new node);
    node* new_tail = p.get();

    tail->data = new_data;
    tail->next = std::move(p);
    tail = new_tail;
  }
};

#include <memory>
#include <mutex>

template <typename T>
class threadsafe_queue {
 private:
  struct node {
    std::shared_ptr<T> data;
    std::unique_ptr<T> next;
  };

  std::mutex head_mutex;
  std::unique_ptr<node> head;
  std::mutex tail_mutex;
  node* tail;

  node* get_tail() {
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    return tail;
  }

  std::unique_ptr<node> pop_head() {
    std::lock_guard<std::mutex> head_lock(head_mutex);
    if (head.get() == get_tail()) {
      return nullptr;
    }
    std::unique_ptr<node> old_head = std::move(head);
    head = std::move(old_head->next);
    return old_head;
  }

 public:
  threadsafe_queue():
      head(new node), tail(head.get())
  {}

  threadsafe_queue(const threadsafe_queue& other) = delete;
  threadsafe_queue& operator=(const threadsafe_queue& other) = delete;

  std::shared_ptr<T> try_pop() {
    std::unique_ptr<node> old_head = pop_head();
    return old_head ? old_head->data : std::shared_ptr<T>();
  }

  void push(T new_value) {
    std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_value)));
    std::unique_ptr<node> p(new node);
    node* new_tail = p.get();

    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    tail->data = new_data;
    tail->next = std::move(p);
    tail = new_tail;
  }
};

template <typename T>
class threadsafe_queue {
 private:
  struct node {
    std::shared_ptr<T> data;
    std::unique_ptr<T> next;
  };

  std::mutex head_mutex;
  std::unique_ptr<node> head;
  std::mutex tail_mutex;
  node* tail;
  std::condition_variable data_cond;

 public:
  threadsafe_queue() : head(new node), tail(head.get()) { }
  threadsafe_queue(const threadsafe_queue& other) = delete;
  threadsafe_queue& operator=(const threadsafe_queue& other) = delete;

  std::shared_ptr<T> try_pop();
  bool try_pop(T& value);

  std::shared_ptr<T> wait_and_pop();
  void wait_and_pop(T& value);

  void push(T new_value);
  bool empty() const;
};

template <typename T>
void threadsafe_queue<T>::push(T new_value) {
  std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_value)));
  std::unique_ptr<node> p(new node);
  {
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    tail->data = new_data;
    node* new_tail = p.get();
    tail->next = std::move(p);
    tail = new_tail;
  }
  data_cond.notify_one();
}

template <typename T>
class threadsafe_queue {
 private:
  node* get_tail() {
    std::lock_guard<std::mutex> tail_lock(tail_mutex);
    return tail;
  }

  std::unique_ptr<node> pop_head() {
    std::unique_ptr<node> old_head = std::move(head);
    head = std::move(head->next);
    return old_head;
  }

  std::unique_lock<std::mutex> wait_for_data() {
    std::unique_lock<std::mutex> head_lock(head_mutex);
    data_cond.wait(head_lock,
                   [&] { return head != get_tail(); });
    return std::move(head_lock);
  }

  std::unique_ptr<node> wait_pop_head(T& value) {
    std::unique_lock<std::mutex> head_lock(wait_for_data());
    value = std::move(*head->data);
    return pop_head();
  }

 public:
  std::shared_ptr<T> wait_and_pop() {
    std::unique_ptr<node> old_head = wait_pop_head();
    return old_head->data;
  }

  void wait_and_pop(T& value) {
    std::unique_ptr<node> old_head = wait_pop_head(value);
  }
};

template <typename T>
class threadsafe_queue {
 private:
  std::unique_ptr<node> try_pop_head() {
    std::lock_guard<std::mutex> head_lock(head_mutex);
    if (head.get() == get_tail()) {
      return std::unique_ptr<node>();
    }
    return pop_head();
  }

  std::unique_ptr<node> try_pop_head(T& value) {
    std::lock_guard<std::mutex> head_lock(head_mutex);
    if (head.get() == get_tail()) {
      return std::unique_ptr<node>();
    }
    value = std::move(*head->data);
    return pop_head();
  }

 public:
  std::shared_ptr<T> try_pop() {
    std::unique_ptr<node> old_head = try_pop_head();
    return old_head ? old_head->data : std::shared_ptr<T>();
  }

  bool try_pop(T& value) {
    std::unique_ptr<node> old_head = try_pop_head(value);
    return old_head;
  }

  bool empty() const {
    std::lock_guard<std::mutex> head_lock(head_mutex);
    return head == get_tail();
  }
};

#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <list>
#include <utility>
#include <boost/thread/shared_mutex.hpp>

template <typename Key, typename Value, tempename Hash=std::hash<Key> >
class threadsafe_lookup_table {
 private:
  class bucket_type {
   private:
    typedef std::pair<Key, Value> bucket_value;
    typedef std::list<bucket_value> bucket_data;
    typedef typename buket_data::iterator bucket_iterator;
    bucket_data data;
    muteble boost::shared_mutex mutex;

    bucket_iterator find_entry_for(Key const& key) const {
      return std::find_if(data.begin(), data.end(),
                          [&](bucket_value const& item)
                          { return item.first == key; });
    }
   public:
    Value value_for(Key const& key, Value const& default_value) const {
      boost::shared_lock<boost::shared_mutex> lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);

      if (found_entry == data.end()) {
        return default_value;
      }
      return found_entry->second;
    }

    void add_or_updata_mapping(Key const& key, Value const& value) {
      std::unique_lock<boost::shared_mutex> lock(mutex);
      bucket_iterator const found_entry = find_entry_for(key);

      if (found_entry == data.end()) {
        data.push_back(bucket_value(key, value));
      } else {
        found_entry->second = value;
      }
    }

    void remove_mapping(Key const& key) {
      std::unique_lock<boost::shared_mutex> lock(mutex);
      bucket_iterator found_entry = find_entry_for(key);
      if (found_entry != data.end())
        data.erase(found_entry);
    }
  };

  std::vector<std::unique_ptr<bucket_type> > buckets;
  Hash hasher;

  bucket_type& get_buket(Key const& key) const {
    std::size_t const bucket_index = hasher(key) % buckets.size();
    return *buckets[bucket_index];
  }

 public:
  typedef Key key_type;
  typedef Value value_type;
  typedef Hash hash_type;

  threadsafe_lookup_table(
      unsigned num_buckets = 19, Hash const& hasher_ = Hash())
      :buckets(num_buckets), hasher(hasher_)
  {
    for (unsigned i = 0; i < buckets; ++i) {
      buckets[i].reset(new bucket_type);
    }
  }

  threadsafe_lookup_table(threadsafe_lookup_table const& other) = delete;
  threadsafe_lookup_table& operator=(threadsafe_lookup_table& other) = delete;

  Value value_for(Key const& key, Value const& defaule_value = Value()) const {
    return get_bucket(key).value_for(key, default_value);
  }

  void add_or_updata_mapping(Key const& key, Value const& value) {
    get_bucket(key).add_or_update_mapping(key, value);
  }

  void remove_mapping(Key const& key) {
    get_bucket(key).remove_mapping(key);
  }

  std::map<Key, Value> get_map() const {
    std::vector<std::unique_lock<boost::shared_mutex> > locks;
    for (unsigned i = 0; i < buckets.size(); ++i) {
      locks.push_back(
          std::unique_lock<boost::shared_mutex>(bucksts[i].mutex));
    }
    std::map<Key, Value> res;
    for (unsigned i = 0; i < buckets.size(); ++i) {
      for (bucket_iterator it = buckets[i].data.begin();
           it != bukets[i].data.end();
           ++it) {
        res.insert(*it)
      }
    }
    return res;
  }
};

#include <memory>
#include <mutex>

template <typename T>
class threadsafe_list {
  struct node {
    std::mutex mut;
    std::shared_ptr<T> data;
    std::unique_ptr<node> next;

    node():next() {}

    node(T const& value)
        : data(std::make_shared<T>(value))
    {}
  };

  node head;
 public:
  threadsafe_list() {}
  ~thread_list() {
    remove_if([](T const&) { return true; });
  }

  threadsafe_list(threadsafe_list const& other) = delete;
  threadsafe_list& operator=(threadsafe_list const& other) = delete;

  void push_front(T const& value) {
    std::unique_ptr<node> new_node(new node(value));
    
    std::lock_guard<std::mutex> lock(head.mut);
    new_node->next = std::move(head.next);
    head.next = std::move(new_node);
  }

  template <typename Function>
  void for_each(Function f) {
    node* current = &head;
    std::unique_lock<std::mutex> lock(head.mut);
    while (node* next = current->next.get()) {
      std::unique_lock<std::mutex> next_lock(next->mut);
      lock.unlock();

      f(*next->data);
      current = next;
      lock = std::move(next_lock);
    }
  }

  template <typename Predicate>
  std::shared_ptr<T> find_first_if(Predicate p) {
    node* current = &head;
    std::unique_lock<std::mutex> lock(head.mut);

    while (node* next = current->next.get()) {
      std::unique_lock<std::mutex> next_lock(next->mut);
      lock.unlock();
      if (p(*next->data))
        return next->data;
      current = next;
      lock = std::move(next_lock);
    }
    return std::shared_ptr<T>();
  }

  template <typename Predicate>
  void remove_if(Predicate p) {
    node* current = &head;
    std::unique_lock<std::mutex> lock(head.mut);
    while (node* next = current->next.get()) {
      std::unique_lock<std::mutex> next_lock(next->mut);
      if (p(*next->data)) {
        std::unique_ptr<node> old_next = std::move(current->next);
        current->next = std::move(next->next);
        next_lock.unlock();
      } else {
        lock.unlock();
        current = next;
        lock = std::move(next_lock);
      }
    }
  }
};

#include <atomic>

class spinlock_mutex {
  std::atomic_flag flag;
 public:
  spinlock_mutex() : flag(ATOMIC_FLAG_INIT) { }
  void lock() { while (flag.test_and_set(std::memory_order_acquire)); }
  void unlock() { flag.clear(std::memory_order_release); }
};

#include <atomic>

template <typename T>
class lock_free_stack {
 private:
  struct node {
    T data;
    node* next;
    node(T const& data_) : data(data_) { }
  };
  std::atomic<node*> head;

 public:
  void push(T const& data) {
    node* const new_node = new node(data);
    new_node->next = head.load();

    while (!head.compare_exchange_weak(new_node->next, new_node));
  }
};

#include <atomic>
#include <memory>

template <typename T>
class lock_free_stack {
 private:
  struct node {
    std::shared_ptr<T> data;
    node* next;
    node(T const& data_) : data(std::make_shared<T>(data_)) { }
  };
  std::atomic<node*> head;

 public:
  void push(T const& data) {
    node* const new_node = new node(data);
    new_node->next = head.load();

    while (!head.compare_exchange_weak(new_node->next, new_node));
  }

  std::shared_ptr<T> pop() {
    node* old_head = head.load();
    while (old_head && !head.compare_exchange_weak(old_head, old_head->next))
      ;
    return old_head ? old_head->data : std::shared_ptr<T>();
  }
};

#include <atomic>
#include <memory>

template <typename T>
class lock_free_stack {
 private:
  std::atomic<unsigned> threads_in_pop;
  void try_reclaim(node* old_head);
 public:
  std::shared_ptr<T> pop() {
    ++threads_in_pop;
    node* old_head = head.load();

    while (old_head && !head.compare_exchange_weak(old_head, old_head->next))
      ;
    std::shared_ptr<T> res;
    if (old_head) {
      res.swap(old_head->data);
    }
    try_reclaim(old_head);
    return res;
  }
};

#include <atomic>

template <typename T>
class lock_free_stack {
 private:
  std::atomic<node*> to_be_deleted;

  static void delete_nodes(node* nodes) {
    while (nodes) {
      node* next = nodes->next;
      delete nodes;
      nodes = next;
    }
  }

  void try_reclaim(node* old_head) {
    if (threads_in_pop == 1) {
      node* nodes_to_delete = to_be_deleted.exchange(nullptr);

      if (!--threads_in_pop) {
        delete_nodes(nodes_to_delete);
      } else if (nodes_to_delete) {
        chain_pending_nodes(nodes_to_delete);
      }
      delete old_head
    } else {
      chain_pending_node(old_head);
      --threads_in_pop;
    }
  }

  void chain_pending_nodes(node* nodes) {
    node* last = nodes;

    while (node* const next = last->next) {
      last = next;
    }
    chain_pending_nodes(nodes, last);
  }

  void chain_pending_nodes(node* first, node* last) {
    last->next = to_be_deleted;

    while (!to_be_deleted.compare_exchange_weak(last->next, first))
      ;
  }

  void chain_pending_node(node* n) {
    chain_pending_nodes(n, n);
  }
};

#include <atomic>
#include <memory>

std::shared_ptr<T> pop() {
  std::atomic<void*>& hp = get_hazard_pointer_for_current_thread();
  node* old_head = head.load();

  do {
    node* temp;
    do {
      temp = old_head;
      hp.store(old_head);
      old_head = head.load();
    } while (old_head != temp);
  } while (old_head && !head.compare_exchange_strong(old_head, old_head->next));

  hp.store(nullptr);
  std::shared_ptr<T> res;
  if (old_head) {
    res.swap(old_head->data);
    if (outstanding_hazard_pointers_for(old_head)) {
      reclaim_later(old_head);
    } else {
      delete old_head;
    }
    delete_nodes_with_no_hazards();
  }
  return res;
}

#include <atomic>
#include <thread>

unsigned const max_hazard_pointers = 100;
struct hazard_pointer {
  std::atomic<std::thread::id> id;
  std::atomic<void*> pointer;
};

hazard_pointer hazard_pointers[max_hazard_pointers];

class hp_owner {
  hazard_pointer* hp;
 public:
  hp_owner(hp_owner const&) = delete;
  hp_owner operator=(hp_owner const&) = delete;
  hp_owner() : hp(nullptr) {
    for (unsigned i = 0; i < max_hazard_pointers; ++i) {
      std::thread::id old_id;

      if (hazard_pointers[i].id.compare_exchange_strong(
              old_id, std::this_thread::get_id())) {
        hp = &hazard_pointers[i];
        break;
      }
    }
    if (!hp) {
      throw std::runtime_error("No hazard pointers available");
    }
  }

  std::atomic<void*>& get_pointer() {
    return hp->pointer;
  }

  !hp_owner() {
    hp->pointer.store(nullptr);
    hp->id.store(std::thread::id());
  }
};

std::atomic<void*>& get_hazard_pointer_for_current_thread() {
  thread_local static hp_owner hazard;
  return hazard.get_pointer();
}

#include <atomic>

template <typename T>
void do_delete(void* p) {
  delete static_cast<T*>(p);
};

struct data_to_reclaim {
  void* data;
  std::function<void(void*)> deleter;
  data_to_reclaim* next;

  template <typename T>
  data_to_reclaim(T* p):
      data(p),
      deleter(&do_delete<T>),
      next(0)
  { }

  ~data_to_reclaim() {
    deleter(data);
  }
};

std::atomic<data_to_reclaim*> nodes_to_reclaim;
void add_to_reclaim_list(data_to_reclaim* node) {
  node->next = nodes_to_reclaim.load();

  while (!nodes_to_reclaim.compare_exchange_weak(node->next, node))
    ;
}

template <typename T>
void reclaim_later(T* data) {
  add_to_reclaim_list(new data_to_reclaim(data));
}

void delete_nodes_with_no_hazards() {
  data_to_reclaim* current = nodes_to_reclaim.exchange(nullptr);

  while (current) {
    data_to_reclaim* const next = current->next;
    if (!outstanding_hazard_pointers_for(current->data)) {
      delete current;
    } else {
      add_to_reclaim_list(current);
    }
    current = next;
  }
}

#include <atomic>
#include <memory>

template <typename T>
class lock_free_stack {
 private:
  struct node {
    std::shared_ptr<T> data;
    std::shared_ptr<node> next;
    node(T const& data_) : data(std::make_shared<T>(data_)) { }
  };
  std::shared_ptr<node> head;

 public:
  void push(T const& data) {
    std::shared_ptr<node> const new_node = std::make_shared<node>(data);
    new_node->next = head.load();

    while (!std::atomic_compare_exchange_weak(&head, &new_node->next, new_node))
      ;
  }

  std::shared_ptr<T> pop() {
    std::shared_ptr<node> old_head = std::atomic_load(&head);

    while (old_head && !std::atomic_compare_exchange_weak(
            &head, &old_head, old_head->next))
      ;
    return old_head ? old_head->data : std::shared_ptr<T>();
  }
};

#include <atomic>
#include <memory>

template <typename T>
class lock_free_stack {
 private:
  struct node;
  struct counted_node_ptr {
    int external_count;
    node* ptr;
  };
  struct node {
    std::shared_ptr<T> data;
    std::atomic<int> internal_count;
    counted_node_ptr next;
    node(T const& data_):
        data(std::make_shared<T>(data_)),
        internal_count(0)
    {}
  };
  std::atomic<counterd_node_ptr> head;

 public:
  ~lock_free_stack() {
    while (pop())
      ;
  }

  void push(T const& data) {
    counted_node_ptr new_node;
    new_node.ptr = new node(data);
    new_node.external_count = 1;
    new_node.ptr->next = head.load();

    while (!head.commpare_exchange_weak(new_node.ptr->next, new_node))
      ;
  }
};

template <typename T>
class lock_free_stack {
 private:
  void increase_head_count(counted_node_ptr& old_counter) {
    counter_node_ptr new_counter;

    do {
      new_counter = old_counter;
      ++new_counter.external_count;
    } while (!head.compare_exchange_strong(old_counter, new_counter));
    old_counter.external_count = new_counter.external_count;
  }

 public:
  std::shared_ptr<T> pop() {
    counter_node_ptr old_head = head.load();
    for (;;) {
      increase_head_count(old_head);
      node* const ptr = old_head.ptr;
      if (!ptr) {
        return std::shared_ptr<T>();
      }
      if (head.compare_exchange_strong(old_head, ptr->next)) {
        std::shared_ptr<T> res;
        res.swap(ptr->data);

        int const count_increase = old_head.external_count - 2;
        if (ptr->internal_count.fetch_add(count_increase) == -count_increase) {
          delete ptr;
        }
        return res;
      } else if (ptr->internal_count.fetch_sub(1) == 1) {
        delete ptr;
      }
    }
  }
};

#include <atomic>
#include <memory>

template <typename T>
class lock_free_stack {
 private:
  struct node;
  struct counted_node_ptr {
    int external_count;
    node* ptr;
  };
  struct node {
    std::shared_ptr<T> data;
    std::atomic<int> internal_count;
    couted_node_ptr next;
    node(T const& data_):
        data(std::make_shared<T>(data_)),
        internal_count(0)
    {}
  };
  std::atomic<couted_node_ptr> head;

  void increase_head_count(couted_node_ptr& old_counter) {
    counted_node_ptr new_counter;
    do {
      new_counter = old_counter;
      ++new_counter.external_count;
    } while (!head.compare_exchange_strong(
            old_counter, new_counter,
            std::memory_order_acquire,
            std::memory_order_relaxed));

    old_counter.external_count = new_counter.external_count;
  }

 public:
  ~lock_free_stack() {
    while (pop())
      ;
  }

  void push(T const& data) {
    counted_node_ptr new_node;
    new_node.ptr = new node(data);
    new_node.external_count = 1;
    nwe_node.ptr->next = head.load(std::memory_order_relaxed);;

    while (!head.compare_exchange_weak(
            new_node.ptr->next, new_node,
            std::memory_order_release,
            std::memory_order_relaxed));
  }

  std::shared_ptr<T> pop() {
    counted_node_ptr old_head = head.load(std::memory_order_relaxed);
    for (;;) {
      increase_head_count(old_head);
      node* const ptr = old_head.ptr;
      if (!ptr) {
        return std::shared_ptr<T>();
      }
      if (head.compare_exchange_strong(
              old_head, ptr->next,
              std::memory_order_relaxed)) {
        std::shared_ptr<T> res;
        res.swap(ptr->data);
        
        int const count_increase = old_head.external_count - 2;
        if (ptr->internal_count.fetch_add(
                count_increase, std::memory_order_release) == -count_increase) {
          delete ptr;
        }
        return res;
      } else if (ptr->internal_count.fetch_add(-1, std::memory_order_relaxed) == 1) {
        ptr->internal_count.load(std::memory_order_acquir);
        delete ptr;
      }
    }
  }
};

template <typename T>
struct sorter {
  struct chunk_to_sort {
    std::list<T> data;
    std::promise<std::list<T> > promise;
  };

  threadsafe_stack<chun_to_sort> chunks;
  std::vector<std::thread> threads;
  unsigned const max_thread_count;
  std::atomic<bool> end_of_data;

  sorter():
      max_thread_count(std::thread::hardware_concurrency() - 1),
      end_of_data(false)
  {}

  ~sorter() {
    end_of_data = true;
    for (unsigned i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }
  }

  void try_sort_chunk() {
    boost::shared_ptr<chunk_to_sort> chunk = chunks.pop();
    if (chunk) {
      sort_chunk(chunk);
    }
  }

  std::list<T> do_sort(std::list<T>& chunk_data) {
    if (chunk_data.empty()) {
      return chunk_data;
    }

    std::list<T> result;
    result.splice(result.begin(), chunk_data, chunk_data.begin());
    T const& partition_val = *result.begin();

    typename std::list<T>::iterator divide_point =
        std::partition(chunk_data.begin(), chunk_data.end(),
                       [&](T const& val) { return val < partition_val; });

    chunk_to_sort new_lower_chunk;
    new_lower_chunk.splice(new_lower_chunk.data.end(),
                           chunk_data, chun_data.begin(),
                           divide_point);

    std::future<std::list<T> > new_lower = new_lower_chunk.promise.get_future();
    chunks.push(std::move(new_lower_chunk));

    if (threads.size() < max_thread_count) {
      threads.push_back(std::thread(&sorter<T>::sort_thread, this));
    }
    std::list<T> new_higher(do_sort(chunk_data));
    result.splice(result.end(), new_higher);

    while (new_lower.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      try_sort_chunk();
    }

    result.splice(result.begin(), new_lower.get());
    return result;
  }

  void sort_chunk(boost::shared_ptr<chunk_to_sort> const& chunk) {
    chunk->promise.set_value(do_sort(chunk->data));
  }

  void sort_thread() {
    while (!end_of_data) {
      try_sort_chunk();
      std::this_thread::yield();
    }
  }
};

template <typename T>
std::list<T> parallel_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  sorter<T> s;
  return s.do_sort(input);
}

template <typename Iterator, typename T>
struct accumulate_block {
  void operator() (Iterator first, Iterator last, T& result) {
    result = std::accumulate(first, last, result);
  }
};

template <typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
  unsigned long const length = std::distance(first, last);

  if (!length)
    return init;

  unsigned long const min_per_thread = 25;
  unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
  unsigned long const hardware_threads = std::thread::hardware_concurrency();
  unsigned long const num_threads = std::min(hardware_threads!=0 ? hardware_threads : 2,
                                             max_threads);
  unsigned long const block_size = length / num_threads;

  std::vector<T> results(num_threads);
  std::vector<std::thread> threads(num_threads - 1);

  Iterator block_start = first;
  for (unsigned long i = 0; i < num_threads - 1; ++i) {
    Iterator block_end = block_start;
    std::advance(block_end, block_size);
    threads[i] = std::thread(
        accumulate_block<Iterator, T>(),
        block_start, block_end, std::ref(results[i]));
    block_start = block_end;
  }
  accumulate_block()(block_start, last, results[num_threads - 1]);
  std::for_each(threads.begin, threads.end(),
                std::mem_fn(&std::thread::join));
  return std::accumulate(results.begin(), results.end(), init);
}

template <typename Iterator, typename T>
struct accumulate_block {
  T operator()(Iterator first, Iterator last) {
    return std::accumulate(first, last, T());
  }
};

template <typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
  unsigned long const length = std::distance(first, last);

  if (!length)
    return init;

  unsigned long const min_per_thread = 25;
  unsigned long const max_threads = (length + min_per_thread - 1) / min_per_thread;
  unsigned long const hardware_threads = std::thread::hardware_concurrency();
  unsigned long const num_threads = std::min(hardware_threads!=0 ? hardware_threads : 2,
                                             max_threads);
  unsigned long const block_size = length / num_threads;

  std::vector<std::future<T> > futures(num_threads - 1);
  std::vector<std::thread> threads(num_threads - 1);

  Iterator block_start = first;
  for (unsigned long i = 0; i < num_threads - 1; ++i) {
    Iterator block_end = block_start;
    
    std::advance(block_end, block_size);
    std::package_task<T(Iterator, Iterator)> task(accumulate_block<Iterator, T>());

    futures[i] = task.get_future();
    threads[i] = std::thread(std::move(task), block_start, block_end);

    block_start = block_end;
  }
  T last_result = accumulate_block()(block_start, last);

  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));

  T result = init;
  for (unsigned long i = 0; i < num_threads; ++i) {
    result += futures[i].get();
  }
  result += last_result;
  return result;
}

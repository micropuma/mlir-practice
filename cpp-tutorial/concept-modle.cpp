#include <memory>
#include <iostream>
#include <vector>

// Cannot change that! library!
struct library_task{};

struct some_library_task : library_task {
    void process() {
        std::cout << "Library Task" << std::endl;
    }
};

// some of our tasks.

struct print_task {
    void process() {
        std::cout << "Print Task" << std::endl;
    }
};

struct special_task {
    int process(bool more_stuff = false) const {
        std::cout << "Special Task" << std::endl;

        return 1;
    }
};

struct some_task {
    void process() const {
        std::cout << "Some Task" << std::endl;
    }

    void stuff() {
        std::cout << "Some Task Stuff" << std::endl;
    }
};

// Our wrapper. A polymorphic container for all tasks.
struct task {
    template<typename T>
    task(T t) noexcept : self{std::make_unique<model_t<T>>(std::move(t))} {}
    
    void process() {
        self->process();
    }
    
private:
    // concept is the interface representing the concept
    struct concept_t {
        virtual ~concept_t() = default;
        virtual void process() = 0;
    };
    
    // This is the implementation of the concept.
    template<typename T>
    struct model_t : concept_t {
        model_t(T s) noexcept : self{std::move(s)} {}
        void process() override { self.process(); }
        T self;
    };

    // the contained erased type
    std::unique_ptr<concept_t> self;
};

// our tasks. all tasks are processed inside run()
std::vector<task> tasks;

// add a new task into the vector
void push(task t) {
    tasks.emplace_back(std::move(t));
}

// Runs all the task and clear the vector
void run() {
    for(auto&& task : tasks) {
        task.process();
    }
    
    tasks.clear();
}

void do_stuff(bool condition) {
    some_task t;
    
    // do some stuff with task
    t.stuff();
    
    // maybe push the task
    if (condition) {
        push(std::move(t));
    }
}

int main() {

    do_stuff(true);
    do_stuff(false);
    
    // natural syntax for object construction! Yay!
    push(some_library_task{});
    push(special_task{});
    push(print_task{});

    run();
}
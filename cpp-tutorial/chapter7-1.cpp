#include <iostream>

class TimeKeeper {
public:
    virtual ~TimeKeeper() = 0;  // 纯虚析构函数
    virtual void displayTime() const = 0;
};

// error：必须定义
// // 必须定义纯虚析构函数的实现
// TimeKeeper::~TimeKeeper() {
//     std::cout << "TimeKeeper destructor\n";
// }

class AtomicClock : public TimeKeeper {
public:
    ~AtomicClock() {
        std::cout << "AtomicClock destructor\n";
    }
    void displayTime() const override {
        std::cout << "AtomicClock: Showing time from atomic mechanism.\n";
    }
};

int main() {
    TimeKeeper* clock = new AtomicClock();
    clock->displayTime();
    delete clock;  // 会依次调用 AtomicClock 和 TimeKeeper 析构函数
    return 0;
}
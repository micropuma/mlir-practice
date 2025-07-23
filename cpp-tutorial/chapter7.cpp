#include <iostream>
#include <string>

// 抽象基类
class TimeKeeper {
public:
    ~TimeKeeper() = default;

    // 纯虚函数：子类必须实现
    virtual void displayTime() const = 0;
};

// 派生类 AtomicClock
class AtomicClock : public TimeKeeper {
public:
    void displayTime() const override {
        std::cout << "AtomicClock: Showing time from atomic mechanism.\n";
    }
};

// 派生类 WaterClock
class WaterClock : public TimeKeeper {
public:
    void displayTime() const override {
        std::cout << "WaterClock: Showing time based on water flow.\n";
    }
};

// 工厂函数：根据输入返回不同的 TimeKeeper 实例（裸指针）
TimeKeeper* getTimeKeeper(const std::string& clockType) {
    if (clockType == "AtomicClock") {
        return new AtomicClock();
    } else if (clockType == "WaterClock") {
        return new WaterClock();
    } else {
        return nullptr;
    }
}

// 测试程序
int main() {
    TimeKeeper* tk1 = getTimeKeeper("AtomicClock");
    TimeKeeper* tk2 = getTimeKeeper("WaterClock");

    if (tk1) tk1->displayTime();
    if (tk2) tk2->displayTime();

    // 手动释放内存，防止内存泄漏
    delete tk1;
    delete tk2;

    return 0;
}

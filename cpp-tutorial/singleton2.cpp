#include <iostream>

class Singleton {
public:
    // 返回单例实例
    static Singleton* getInstance() {
        return &instance; // 返回静态实例
    }

    // 打印实例创建信息
    void printMessage() {
        std::cout << "Singleton instance created!" << std::endl;
    }

private:
    // 私有构造函数，确保外部无法创建实例
    Singleton() {
        std::cout << "Singleton Constructor Called" << std::endl;
    }

    // 静态成员变量，在类加载时初始化
    static Singleton instance; // 唯一实例
};

// 定义静态成员变量，初始化实例
Singleton Singleton::instance;

int main() {
    // 通过 getInstance 获取单例实例
    Singleton* singleton1 = Singleton::getInstance();
    Singleton* singleton2 = Singleton::getInstance();

    // 打印消息，验证同一个实例
    singleton1->printMessage();
    singleton2->printMessage();

    // 检查两个指针是否指向相同实例
    if (singleton1 == singleton2) {
        std::cout << "Both pointers point to the same instance." << std::endl;
    } else {
        std::cout << "Different instances." << std::endl;
    }

    return 0;
}
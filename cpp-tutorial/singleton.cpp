#include <iostream>

class Singleton {
public:
    Singleton(const Singleton&) = delete; // Prevent copy construction
    Singleton& operator=(const Singleton&) = delete; // Prevent assignment

    static Singleton* getInstance() {
        if (instance == nullptr) {
            instance = new Singleton();
        }
        return instance;
    }

    void printMessage() {
        std::cout << "Hello from Singleton!" << std::endl;
    }
    
private:
    // 构造函数私有化，确保外部无法直接创建对象
    Singleton() {
        std::cout << "Singleton created!" << std::endl;
    }

    static Singleton* instance;
};

// 一定要定义
Singleton* Singleton::instance = nullptr;

int main() {
    Singleton* s1 = Singleton::getInstance();
    Singleton* s2 = Singleton::getInstance();

    if (s1 == s2) {
        std::cout << "Both pointers point to the same instance." << std::endl;
    }

    s1->printMessage();
    s2->printMessage(); 
    return 0;
}
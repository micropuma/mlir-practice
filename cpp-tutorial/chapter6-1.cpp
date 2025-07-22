#include <iostream>

using namespace std;

class Uncopyable {
protected:
    Uncopyable() = default; // 默认构造函数
    ~Uncopyable() = default; // 默认析构函数
private:
    // 禁止拷贝构造函数和赋值运算符
    Uncopyable(const Uncopyable&);
    Uncopyable& operator=(const Uncopyable&);
};

class HomeForSale : public Uncopyable {
public:
    HomeForSale() {
        cout << "HomeForSale constructor called." << endl;
    }
    ~HomeForSale() {
        cout << "HomeForSale destructor called." << endl;
    }

    // friend可以调用 private 成员！
    friend HomeForSale makeCopy(const HomeForSale& obj) {
        return HomeForSale(obj);           // 明确的编译器报错
    }
};

int main() {
    HomeForSale home1;
    HomeForSale home2 = makeCopy(home1);
    return 0;
}

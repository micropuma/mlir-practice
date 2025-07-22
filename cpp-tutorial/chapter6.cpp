#include <iostream>

using namespace std;

class HomeForSale {
private:
    // 在private中声明
    HomeForSale(const HomeForSale& other);
    HomeForSale& operator=(const HomeForSale& other);

public:
    HomeForSale() {
        cout << "HomeForSale constructor called." << endl;
    }
    ~HomeForSale() {
        cout << "HomeForSale destructor called." << endl;
    }

    // friend可以调用 private 成员！
    friend HomeForSale makeCopy(const HomeForSale& obj) {
        return HomeForSale(obj);
    }
};

int main() {
    HomeForSale home1;
    HomeForSale home2 = makeCopy(home1);
    return 0;
}

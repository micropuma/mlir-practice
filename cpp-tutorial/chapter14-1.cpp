#include <future>
#include <iostream>
#include <mutex>
#include <tr1/memory>

using namespace std;

class Lock {
public:
    explicit Lock(mutex* m) 
        : mtx(m, [](mutex* mtx_ptr){ mtx_ptr->unlock(); }) // 自定义删除器
    {
        mtx->lock();  // 构造时加锁
    }

private:
    tr1::shared_ptr<mutex> mtx;
};

int main() {
    mutex m;
    {
        Lock lock(&m);  // 构造时加锁
        cout << "Mutex locked" << endl;
        // 离开作用域时 lock 对象析构，mutex 自动 unlock
    }
    cout << "Mutex unlocked" << endl;
    return 0;
}

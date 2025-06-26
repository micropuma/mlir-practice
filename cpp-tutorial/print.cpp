#include <iostream>
#include <ostream>


using namespace std;

// 收录一个triton中看到的log轮子（print函数）

static void print(ostream& o) {
    auto print = [&](const string& s)     {
        o << "[";
        o << s;
        o << "]\n";
    };

    print("leon");
    print("dou");
}

int main() {
    print(cout);

    return 0;
}
#include <iostream>

using namespace std;

template <typename T>
class NamedObject {
public:
    NamedObject(string& name, const T& value) : name_(name), value_(value) {}

    void print() const {
        cout << "Name: " << name_ << ", Value: " << value_ << endl;
    }

private:
    string& name_;
    const T value_;
};

int main() {
    std::string name1("Leon Dou");
    std::string name2("Dou Leon");
    NamedObject<int> p(name1, 1);
    NamedObject<int> q(name2, 2);
    p = q;

    return 0;
}
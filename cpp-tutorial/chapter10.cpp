#include <iostream>
using namespace std;

class Complex {
public:

    Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}
    
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }

    void display() const {
        cout << real << " + " << imag << "i" << endl;
    }
private:
    double real;
    double imag;
};

int main() {
    Complex c1(3.0, 4.0);
    Complex c2(1.0, 2.0);
    
    Complex c3 = c1 + c2;
    c3.display();  // 输出: 4.0 + 6.0i
    
    Complex c4 = c1 - c2;
    c4.display();  // 输出: 2.0 + 2.0i
    
    c1 += c2 += c2 += c2;      // 等同于 c1 = c1 + c2
    c1.display();  // 输出: 4.0 + 6.0i
    
    c1 -= c2;      // 等同于 c1 = c1 - c2
    c1.display();  // 输出: 3.0 + 4.0i

    return 0;
}
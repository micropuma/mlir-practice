#include <iostream>

using namespace std;

int main() {
    char greeting[] = "Hello, World!";
    cout << greeting << endl; // Output: Hello, World!
    const char * p = greeting; // const pointer to a const literal
    cout << p << endl; // Output: Hello, World!
    p = "Goodbye!"; // Pointing to a string literal
    cout << p << endl; // Output: Goodbye!
    
    const char * q = "Hello, Universe!"; // const pointer to a string literal
    const char * const r = "Hello, Galaxy!"; // const pointer to a const literal
    // r = "Hello, Solar System!"; // Error: cannot change the pointer r
    cout << q << endl; // Output: Hello, Universe!
}
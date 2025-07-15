#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5};

    const vector<int>::iterator it = numbers.begin(); // const iterator, 类似T* const
    *it = 10;
    // ++it;
    vector<int>::const_iterator cit = numbers.cbegin(); // const iterator, 类似const T*
    ++cit;
    // *cit = 10;

    return 0;
}
#include <iostream>
#include <cstring>
#include <cstddef>

using namespace std;

class TextBook {
public:
    TextBook(const char* title) {
        this->title = new char[strlen(title) + 1];
        strcpy(this->title, title);
    }

    ~TextBook() {
        delete[] title;
    }

    void printTitle() const {
        cout << "Title: " << title << endl;
    }   

    char& operator[](size_t index) const {
        return title[index];
    }

    char& operator[](size_t index) {
        return title[index];
    }

private:
    char* title;
};

int main() {
    TextBook book("C++ Programming");

    book.printTitle();

    cout << "First character: " << book[0] << endl;
    book[0] = 'c'; // 修改第一个字符
    cout << "Modified first character: " << book[0] << endl;

    book.printTitle();

    const TextBook ctb("World");
    ctb.printTitle();
    char* pc = &ctb[0];
    *pc = 'w'; // 修改 const 对象的字符

    cout << "First character of const TextBook: " << ctb[0] << endl;
    // ctb[0] = 'w'; // 错误，不能修改 const

    return 0;
}
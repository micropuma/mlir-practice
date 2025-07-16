#include <iostream>
#include <cstring>
#include <cstddef>

class TextBook {
public:
    TextBook(const char* title) {
        size_t len = std::strlen(title);
        this->title = new char[len + 1];
        std::strcpy(this->title, title);
    }

    ~TextBook() {
        delete[] title;
    }

    void printTitle() const {
        std::cout << "Title: " << title << std::endl;
    }

    // const 版本：只能读取，不能修改
    const char& operator[](std::size_t position) const {
        // 可加越界检查（这里只做简单实现）
        return title[position];
    }

    // 非 const 版本：可修改，复用 const 版本
    char& operator[](std::size_t position) {
        return const_cast<char&>(
            static_cast<const TextBook&>(*this)[position]
        );
    }

private:
    char* title;
};

int main() {
    TextBook book("C++ Programming");

    book.printTitle();

    std::cout << "First character: " << book[0] << std::endl;
    book[0] = 'c';  // 修改首字母
    std::cout << "Modified first character: " << book[0] << std::endl;

    book.printTitle();

    const TextBook ctb("World");
    ctb.printTitle();
    // const TextBook 对象只能读，不能改
    std::cout << "First character of const book: " << ctb[0] << std::endl;
    // ctb[0] = 'w'; // ❌ 编译错误，不能对 const 对象赋值

    return 0;
}

#include <iostream>
#include <memory>

using namespace std;

class Investment {
public:
    Investment(int x, int y) : x(x), y(y) {
        cout << "Investment created with x: " << x << " and y: " << y << endl;
    }

    ~Investment() {
        cout << "Investment destroyed" << endl;
    }

    void display() const {
        cout << "Investment values - x: " << x << ", y: " << y << endl;
    }

private:
    int x;
    int y;
};

Investment* createInvestment() {
    return new Investment(10, 20);
}

int main() {
    // Investment* inv = createInvestment();
    // inv->display();
    // delete inv;
    {
        // RAII机制
        auto_ptr<Investment> pInv(createInvestment());
        pInv->display();

        auto_ptr<Investment> pInv2(pInv);
        // pInv->display();
        pInv2->display();

        pInv = pInv2; // pInv2的所有权转移到pInv
    }
    
    return 0;
}
#include <iostream>
#include <functional>

using namespace std;

int main() {
	auto func_wrapper = [](int a, int b) {
		std::function<int(int, int)> fn = [&](int a, int b) {
			return a + b;
		};	

		return fn(a, b);
	};

	int a = 3;
	int b = 4;

	int c = func_wrapper(a, b);
	cout << c;

	return 0;
}


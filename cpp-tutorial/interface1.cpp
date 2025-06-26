#include <iostream>
#include <vector>
#include <memory>

using namespace std;

struct AbstractClass {
	virtual void process() = 0;
	virtual	~AbstractClass() = default;
};

struct PrintClass : AbstractClass{
	void process() override {
		cout << "Hello\n";
	}
};

vector<unique_ptr<AbstractClass>> tasks;

static void push(unique_ptr<AbstractClass> task) {
	tasks.emplace_back(std::move(task));
}

static void run() {
	for (auto &&task : tasks) {
		task->process();
	}
	tasks.clear();
}

int main() {
	auto task1 = make_unique<PrintClass>();
	auto task2 = make_unique<PrintClass>();

	push(std::move(task1));
	push(std::move(task2));

	run();	
	return 0;
}

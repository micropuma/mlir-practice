#include <iostream>
#include <vector>
#include <memory>

// 前向声明
class AddOp;
class SubOp;
class MulOp;

class AxisInfo {
private:
    std::string axisName_; 
public:
    AxisInfo() : axisName_("Unknown") {}  // ✅ 默认构造
    AxisInfo(const std::string& name) : axisName_(name) {}
    friend std::ostream& operator<<(std::ostream& os, const AxisInfo& axisInfo) {
        os << axisInfo.axisName_ << "\n";
        return os;
    }
};

class AxisInfoVisitor {
public:
    virtual ~AxisInfoVisitor() = default;

    virtual void visit(const AddOp& op) = 0;
    virtual void visit(const SubOp& op) = 0;
    virtual void visit(const MulOp& op) = 0;

    AxisInfo result;
};

class Operation {
public:
    virtual ~Operation() = default;
    virtual void accept(AxisInfoVisitor& visitor) const  = 0;
    virtual std::string getName() = 0;
};

class AddOp : public Operation {
public:
    std::string getName() override {
        return "AddOp";
    }

    void accept(AxisInfoVisitor& visitor) const override {
        visitor.visit(*this);
    }
};

class SubOp : public Operation {
public:
    std::string getName() override {
        return "SubOp";
    }

    void accept(AxisInfoVisitor& visitor) const override {
        visitor.visit(*this);
    }
};

class MulOp : public Operation {
public:
    std::string getName() override {
        return "MulOp";
    }

    void accept(AxisInfoVisitor& visitor) const override {
        visitor.visit(*this);
    }
};

// ======================
// 3. 访问者具体实现
// ======================
class ConcreteAxisInfoVisitor : public AxisInfoVisitor {
public:
    void visit(const AddOp& op) override {
        result = AxisInfo("AddOp requires contiguous memory access");
    }

    void visit(const SubOp& op) override {
        result = AxisInfo("SubOp allows strided memory access");
    }

    void visit(const MulOp& op) override {
        result = AxisInfo("MulOp is element-wise and preserves sparsity");
    }
};

int main() {
    std::vector<std::unique_ptr<Operation>> tasks;
    tasks.emplace_back(std::make_unique<AddOp>());
    tasks.emplace_back(std::make_unique<SubOp>());
    tasks.emplace_back(std::make_unique<MulOp>());

    ConcreteAxisInfoVisitor visitor;
    for (const auto& task : tasks) {
        task->accept(visitor);
        std::cout << visitor.result;
    }

    return 0;
}





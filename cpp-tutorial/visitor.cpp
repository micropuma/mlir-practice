#include <iostream>
#include <array>
#include <memory>  // 添加智能指针支持

/**
 * 组件类前置声明
 */
class ConcreteComponentA;
class ConcreteComponentB;

/**
 * 访问者接口
 */
class Visitor {
public:
    virtual ~Visitor() = default;
    virtual void VisitConcreteComponentA(const ConcreteComponentA* element) const = 0;
    virtual void VisitConcreteComponentB(const ConcreteComponentB* element) const = 0;
};

/**
 * 组件接口
 */
class Component {
public:
    virtual ~Component() = default;
    virtual void Accept(Visitor* visitor) const = 0;
};

/**
 * 具体组件A
 */
class ConcreteComponentA : public Component {
public:
    void Accept(Visitor* visitor) const override;  // 实现延迟声明
    
    std::string ExclusiveMethodOfConcreteComponentA() const {
        return "A";
    }
};

/**
 * 具体组件B
 */
class ConcreteComponentB : public Component {
public:
    void Accept(Visitor* visitor) const override;  // 实现延迟声明
    
    std::string SpecialMethodOfConcreteComponentB() const {
        return "B";
    }
};

// 实现组件A的Accept方法
void ConcreteComponentA::Accept(Visitor* visitor) const {
    visitor->VisitConcreteComponentA(this);
}

// 实现组件B的Accept方法
void ConcreteComponentB::Accept(Visitor* visitor) const {
    visitor->VisitConcreteComponentB(this);
}

/**
 * 具体访问者1
 */
class ConcreteVisitor1 : public Visitor {
public:
    void VisitConcreteComponentA(const ConcreteComponentA* element) const override {
        std::cout << element->ExclusiveMethodOfConcreteComponentA() 
                  << " + ConcreteVisitor1\n";
    }

    void VisitConcreteComponentB(const ConcreteComponentB* element) const override {
        std::cout << element->SpecialMethodOfConcreteComponentB() 
                  << " + ConcreteVisitor1\n";
    }
};

/**
 * 具体访问者2
 */
class ConcreteVisitor2 : public Visitor {
public:
    void VisitConcreteComponentA(const ConcreteComponentA* element) const override {
        std::cout << element->ExclusiveMethodOfConcreteComponentA() 
                  << " + ConcreteVisitor2\n";
    }
    
    void VisitConcreteComponentB(const ConcreteComponentB* element) const override {
        std::cout << element->SpecialMethodOfConcreteComponentB() 
                  << " + ConcreteVisitor2\n";
    }
};

/**
 * 客户端操作函数
 */
void ClientCode(std::array<Component*, 2> components, Visitor* visitor) {
    for (Component* comp : components) {
        comp->Accept(visitor);
    }
}

int main() {
    // 使用智能指针自动管理内存
    auto compA = std::make_unique<ConcreteComponentA>();
    auto compB = std::make_unique<ConcreteComponentB>();
    
    std::array<Component*, 2> components = {compA.get(), compB.get()};
    
    std::cout << "基础访问者操作:\n";
    ConcreteVisitor1 visitor1;
    ClientCode(components, &visitor1);
    
    std::cout << "\n不同访问者类型:\n";
    ConcreteVisitor2 visitor2;
    ClientCode(components, &visitor2);

    return 0;
}
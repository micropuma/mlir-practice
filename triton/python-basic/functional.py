# 支持type hints
from typing import Callable, Iterable

# 定义一个函数，返回一个函数，该函数将传入的filter function参数应用到参数数组上
def apply_filter(filter: Callable[[float], bool]) -> Callable[[Iterable[float]], Iterable[float]]:
    def func(ls: Iterable[float]):
        res = []
        for a in ls:
            if filter(a):
                res.append(a)

        return res
    return func

def more_than_4(a: float) -> bool:
    return a > 4

array = [1, 3, 4, 5, 6, 7]
filter_fn = apply_filter(more_than_4) 
print(filter_fn(array))

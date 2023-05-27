from abc import ABC, abstractmethod

from sympy import latex, diff, sin, exp, Symbol
from prettytable import PrettyTable


class Equation:
    """
    Класс обертка для уравнений
    """
    def __init__(self, equation_func, symbol: Symbol) -> None:
        self.equation_func = equation_func
        self.symbol = symbol

    def get_string(self) -> str:
        return latex(self.equation_func)

    def get_diff(self):
        return diff(self.equation_func)


class SolutionMethod(ABC):
    """
    Базовый абстрактный класс для классов реализаций методов решения диф. уравнений
    """
    def __init__(self,
                 equation_diff: Equation,
                 equation_solution: Equation,
                 y_zero: float,
                 x_zero: float,
                 x_n: float,
                 h: float,
                 epsilon: float,
                 header_table) -> None:
        assert x_zero != x_n, "Значения x0 и xn должны быть различны"
        assert x_zero < x_n, "Значение x0 должно быть меньше xn"
        assert h > 0, "Значение шага h должно быть больше нуля"
        assert epsilon > 0, "Значение эпсилон должно быть больше нуля"
        self._equation_diff = equation_diff
        self._equation_solution = equation_solution
        self._y_zero = y_zero
        self._x_zero = x_zero
        self._x_n = x_n
        self._h = h
        self._epsilon = epsilon
        self._header_table = header_table

    @abstractmethod
    def calc(self) -> PrettyTable:
        pass

    @abstractmethod
    def draw(self) -> None:
        pass


class EulerMethod(SolutionMethod):
    """
    Класс метода Эйлера
    """
    name: str = 'метод Эйлера'

    def __init__(self, equation: Equation, a: float, b: float, n: int, epsilon: float = 0.01) -> None:
        super().__init__(equation, a, b, n, 4, epsilon)

    def calc(self) -> tuple[float, int]:
        func = self._equation.equation_func
        x = self._equation.symbol
        integral_value_first: float = 0.0
        n: int = self._n
        while True:
            integral_value_zero = integral_value_first
            integral_value_first = 0.0
            h = (self._b - self._a) / n
            for i in range(1, n):
                x_i = self._a + h * i
                if i % 2 == 0:
                    integral_value_first += 2 * func.subs(x, x_i).evalf()
                    continue
                integral_value_first += 4 * func.subs(x, x_i).evalf()
            integral_value_first += func.subs(x, self._a).evalf() + func.subs(x, self._b).evalf()
            integral_value_first *= h / 3
            if abs((integral_value_first - integral_value_zero) / (2 ** self._k - 1)) < self._epsilon:
                break
            n *= 2
        return integral_value_first, n


def input_data(equations, solution_methods) -> SolutionMethod:
    equation = None
    while True:
        print("Выберите функцию, интеграл которой требуется вычислить:")
        [print(f"{i + 1}. {equation_iter.get_string()}") for i, equation_iter in enumerate(equations)]
        equation_num = int(input("Введите номер выбранной функции...\n"))
        if equation_num < 1 or equation_num > len(equations):
            print("Номер функции не найден, повторите ввод")
            continue
        equation = equations[equation_num - 1]
        break
    while True:
        print("Задайте пределы интегрирования:")
        a, b = (float(i) for i in input("Введите значения a и b через пробел...\n").split())
        if a == b:
            print("Значения должны быть различны")
            continue
        elif a > b:
            print("Значение a должно быть меньше b")
            continue
        break
    solution_method = None
    while True:
        print("Выберите метод решения")
        [print(f"{i + 1}. {solution_method_iter.name}") for i, solution_method_iter in enumerate(solution_methods)]
        solution_num = int(input("Введите номер выбранного метода решения...\n"))
        if solution_num < 1 or solution_num > len(solution_methods):
            print("Номер метода не найден, повторите ввод")
            continue
        solution_method = solution_methods[solution_num - 1]
        break
    while True:
        n = input(
            "Введите значение числа разбиения интервала интегрирования (чтобы оставить значение по умолчанию 4 нажмите Enter)...\n")
        if n == '':
            n = 4
            break
        n = int(n)
        if n <= 0:
            print("Значение должно быть больше нуля")
            continue
        break
    while True:
        epsilon = input(
            "Введите погрешность вычислений (чтобы оставить значение по умолчанию - 0,01 нажмите Enter)...\n")
        if epsilon == '':
            solution_method = solution_method(equation, a, b, n)
            break
        epsilon = float(epsilon)
        if epsilon <= 0:
            print("Значение погрешности должно быть больше нуля")
            continue
        solution_method = solution_method(equation, a, b, n, epsilon)
        break
    return solution_method


def main():
    x = Symbol('x')
    equations = (
        Equation(x ** 3 - 2 * x ** 2 - 5 * x + 24, x),
        Equation(x ** 2, x),
        Equation(sin(x * 2) + 2 * x ** 3 - 1.3 * x + 5.14, x),
        Equation(exp(x) - 1.12 * x ** 2 - 3.14, x),
        Equation(x ** 5 - 1.18, x)
    )
    solution_methods = (
        SimpsonMethod,
    )
    solution_method = input_data(equations, solution_methods)
    if solution_method is None:
        return
    integral_value, n = solution_method.calc()
    print(f"Вычисленное значение интеграла {integral_value}")
    print(f"Число разбиения интервала интегрирования для достижения требуемой точности {n}")


if __name__ == '__main__':
    main()

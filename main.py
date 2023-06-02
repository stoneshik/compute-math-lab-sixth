from abc import ABC, abstractmethod

import numpy
import matplotlib
import matplotlib.pyplot as plt
from sympy import latex, diff, sin, exp, Symbol
from prettytable import PrettyTable


class Equation:
    """
    Класс обертка для уравнений
    """
    def __init__(self, equation_func, first_symbol: Symbol, second_symbol: Symbol = Symbol('y')) -> None:
        self.equation_func = equation_func
        self.first_symbol = first_symbol
        self.second_symbol = second_symbol

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
                 p: int,
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
        self._p = p
        self._header_table = header_table
        self._solution = []

    @abstractmethod
    def calc(self) -> PrettyTable:
        pass

    def draw(self) -> None:
        plt.figure()
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$F(x)$', fontsize=14)
        plt.title(r'Графики точного и приближённого решения$')
        x = Symbol('x')
        x_values = numpy.arange(self._x_zero - self._h * 2, self._x_n + self._h * 2, 0.01)
        y_values = [self._equation_solution.equation_func.subs(x, x_iter) for x_iter in x_values]
        plt.plot(x_values, y_values, color='red')
        x_values = []
        y_values = []
        for i in self._solution:
            x_values.append(i[0])
            y_values.append(i[1])
        plt.plot(x_values, y_values, color='blue', marker='o')
        plt.show()


class EulerMethod(SolutionMethod):
    """
    Класс метода Эйлера
    """
    name: str = 'метод Эйлера'

    def __init__(self,
                 equation_diff: Equation,
                 equation_solution: Equation,
                 y_zero: float,
                 x_zero: float,
                 x_n: float,
                 h: float,
                 epsilon: float = 0.01) -> None:
        super().__init__(
            equation_diff, equation_solution, y_zero, x_zero, x_n, h, epsilon, 1,
            ['i', 'xi', 'yi^h', 'yi^{h/2}', 'f(xi, yi)^h', 'f(xi, yi)^{h/2}', 'R']
        )

    def calc(self) -> PrettyTable:
        func = self._equation_diff.equation_func
        x: Symbol = self._equation_diff.first_symbol
        y: Symbol = self._equation_diff.second_symbol
        table: PrettyTable = PrettyTable()
        table.field_names = self._header_table
        table.add_row([
            0, self._x_zero, self._y_zero, self._y_zero, func.subs({x: self._x_zero, y: self._y_zero}),
            func.subs({x: self._x_zero, y: self._y_zero}), 0
        ])
        rows: list = []
        solution: list = []
        h: float = self._h
        x_i: float = self._x_zero
        y_i: float = self._y_zero
        y_i_h_divide_2: float = self._y_zero
        i: int = 0
        while x_i < self._x_n:
            y_i_plus_1: float = y_i + h * func.subs({x: x_i, y: y_i})
            y_i_h_divide_2_plus_half: float = y_i_h_divide_2 + h/2 * func.subs({x: x_i, y: y_i_h_divide_2})
            y_i_h_divide_2_plus_1: float = y_i_h_divide_2_plus_half + h/2 * func.subs(
                {x: x_i + h/2, y: y_i_h_divide_2_plus_half})
            r: float = abs(y_i_plus_1 - y_i_h_divide_2_plus_1) / (2 ** self._p - 1)
            if r > self._epsilon:
                rows = []
                solution = []
                h /= 4
                x_i = self._x_zero
                y_i = self._y_zero
                y_i_h_divide_2 = self._y_zero
                i = 0
                continue
            y_i = y_i_plus_1
            y_i_h_divide_2 = y_i_h_divide_2_plus_1
            x_i += h
            i += 1
            solution.append((x_i, y_i))
            rows.append([
                i, x_i, y_i, y_i_h_divide_2, func.subs({x: x_i, y: y_i}),
                func.subs({x: x_i + h / 2, y: y_i_h_divide_2}), r
            ])
        self._solution = solution
        table.add_rows(rows)
        return table


class RungeKuttaMethod(SolutionMethod):
    """
    Класс метода Рунге-Кутта 4-го порядка
    """
    name: str = 'метод Рунге-Кутта 4-го порядка'

    def __init__(self,
                 equation_diff: Equation,
                 equation_solution: Equation,
                 y_zero: float,
                 x_zero: float,
                 x_n: float,
                 h: float,
                 epsilon: float = 0.01) -> None:
        super().__init__(
            equation_diff, equation_solution, y_zero, x_zero, x_n, h, epsilon, 4,
            ['i', 'xi', 'yi^h', 'yi^{h/2}', 'R']
        )

    def calc_iter(self, x_i: float, y_i: float, h: float) -> float:
        func = self._equation_diff.equation_func
        x: Symbol = self._equation_diff.first_symbol
        y: Symbol = self._equation_diff.second_symbol
        k1: float = h * func.subs({x: x_i, y: y_i})
        k2: float = h * func.subs({x: x_i + h / 2, y: y_i + k1 / 2})
        k3: float = h * func.subs({x: x_i + h / 2, y: y_i + k2 / 2})
        k4: float = h * func.subs({x: x_i + h, y: y_i + k3})
        return y_i + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def calc(self) -> PrettyTable:
        table: PrettyTable = PrettyTable()
        table.field_names = self._header_table
        table.add_row([0, self._x_zero, self._y_zero, self._y_zero, 0])
        rows: list = []
        solution: list = []
        h: float = self._h
        x_i: float = self._x_zero
        y_i: float = self._y_zero
        y_i_h_divide_2: float = self._y_zero
        i: int = 0
        while x_i < self._x_n:
            y_i_plus_1: float = self.calc_iter(x_i, y_i, h)
            y_i_h_divide_2_plus_half: float = self.calc_iter(x_i, y_i_h_divide_2, h/2)
            y_i_h_divide_2_plus_1: float = self.calc_iter(x_i + h/2, y_i_h_divide_2_plus_half, h/2)
            r: float = abs(y_i_plus_1 - y_i_h_divide_2_plus_1) / (2 ** self._p - 1)
            if r > self._epsilon:
                rows = []
                solution = []
                h /= 4
                x_i = self._x_zero
                y_i = self._y_zero
                y_i_h_divide_2 = self._y_zero
                i = 0
                continue
            y_i = y_i_plus_1
            y_i_h_divide_2 = y_i_h_divide_2_plus_1
            x_i += h
            i += 1
            solution.append((x_i, y_i))
            rows.append([i, x_i, y_i, y_i_h_divide_2, r])
        self._solution = solution
        table.add_rows(rows)
        return table


class AdamsMethod(SolutionMethod):
    """
    Класс метода Адамса
    """
    name: str = 'метод Адамса'

    def __init__(self,
                 equation_diff: Equation,
                 equation_solution: Equation,
                 y_zero: float,
                 x_zero: float,
                 x_n: float,
                 h: float,
                 epsilon: float = 0.01) -> None:
        super().__init__(
            equation_diff, equation_solution, y_zero, x_zero, x_n, h, epsilon, 4,
            ['i', 'xi', 'yi', 'Точное значение', 'R']
        )

    def _calc_iter(self,
                   x_i_minus_3: float,
                   y_i_minus_3: float,
                   y_i_minus_2: float,
                   y_i_minus_1: float,
                   y_i: float,
                   h: float) -> float:
        func = self._equation_diff.equation_func
        x: Symbol = self._equation_diff.first_symbol
        y: Symbol = self._equation_diff.second_symbol
        f_minus_3: float = func.subs({x: x_i_minus_3, y: y_i_minus_3})
        f_minus_2: float = func.subs({x: x_i_minus_3 + h, y: y_i_minus_2})
        f_minus_1: float = func.subs({x: x_i_minus_3 + 2*h, y: y_i_minus_1})
        f_i: float = func.subs({x: x_i_minus_3 + 3*h, y: y_i})
        delta_f_i: float = f_i - f_minus_1
        delta_2_f_i: float = f_i - 2*f_minus_1 + f_minus_2
        delta_3_f_i: float = f_i - 3*f_minus_1 + 3*f_minus_2 - f_minus_3
        return y_i + h * f_i + (h**2 / 2) * delta_f_i + (5*h**3 / 12) * delta_2_f_i + (3*h**4 / 8) * delta_3_f_i

    def calc(self) -> PrettyTable:
        func_solution = self._equation_solution.equation_func
        x: Symbol = self._equation_solution.first_symbol
        table: PrettyTable = PrettyTable()
        table.field_names = self._header_table
        table.add_row([0, self._x_zero, self._y_zero, self._y_zero, 0])
        runge_kutta_method: RungeKuttaMethod = RungeKuttaMethod(
            self._equation_diff, self._equation_solution,
            self._y_zero, self._x_zero, self._x_n, self._h, self._epsilon
        )
        h: float = self._h
        x_i_minus_3: float = self._x_zero
        y_i_minus_3: float = self._y_zero
        y_i_minus_2: float = runge_kutta_method.calc_iter(x_i_minus_3 + h, y_i_minus_3, h)
        y_i_minus_1: float = runge_kutta_method.calc_iter(x_i_minus_3 + 2*h, y_i_minus_2, h)
        y_i: float = runge_kutta_method.calc_iter(x_i_minus_3 + 3*h, y_i_minus_1, h)
        rows: list = [
            [1, x_i_minus_3 + h, y_i_minus_2, func_solution.subs(x, x_i_minus_3 + h),
             abs(func_solution.subs(x, x_i_minus_3 + h) - y_i_minus_2)],
            [2, x_i_minus_3 + 2*h, y_i_minus_1, func_solution.subs(x, x_i_minus_3 + 2*h),
             abs(func_solution.subs(x, x_i_minus_3 + 2*h) - y_i_minus_1)],
            [3, x_i_minus_3 + 3*h, y_i, func_solution.subs(x, x_i_minus_3 + 3*h),
             abs(func_solution.subs(x, x_i_minus_3 + 3*h) - y_i)],
        ]
        solution: list = [
            (x_i_minus_3 + h, y_i_minus_2),
            (x_i_minus_3 + 2*h, y_i_minus_1),
            (x_i_minus_3 + 3*h, y_i)
        ]
        i: int = 3
        while x_i_minus_3 + 4*h < self._x_n:
            y_i_plus_1: float = self._calc_iter(
                x_i_minus_3 + 4*h,
                y_i_minus_3,
                y_i_minus_2,
                y_i_minus_1,
                y_i,
                h
            )
            r: float = abs(func_solution.subs(x, x_i_minus_3 + 4*h) - y_i_plus_1)
            if r > self._epsilon:
                h: float = h / 2
                x_i_minus_3 = self._x_zero
                y_i_minus_3 = self._y_zero
                y_i_minus_2 = runge_kutta_method.calc_iter(x_i_minus_3 + h, y_i_minus_3, h)
                y_i_minus_1 = runge_kutta_method.calc_iter(x_i_minus_3 + 2 * h, y_i_minus_2, h)
                y_i = runge_kutta_method.calc_iter(x_i_minus_3 + 3 * h, y_i_minus_1, h)
                rows: list = [
                    [1, x_i_minus_3 + h, y_i_minus_2, func_solution.subs(x, x_i_minus_3 + h),
                     abs(func_solution.subs(x, x_i_minus_3 + h) - y_i_minus_2)],
                    [2, x_i_minus_3 + 2 * h, y_i_minus_1, func_solution.subs(x, x_i_minus_3 + 2 * h),
                     abs(func_solution.subs(x, x_i_minus_3 + 2 * h) - y_i_minus_1)],
                    [3, x_i_minus_3 + 3 * h, y_i, func_solution.subs(x, x_i_minus_3 + 3 * h),
                     abs(func_solution.subs(x, x_i_minus_3 + 3 * h) - y_i)],
                ]
                solution: list = [
                    (x_i_minus_3 + h, y_i_minus_2),
                    (x_i_minus_3 + 2 * h, y_i_minus_1),
                    (x_i_minus_3 + 3 * h, y_i)
                ]
                i: int = 3
                continue
            y_i_minus_3 = y_i_minus_2
            y_i_minus_2 = y_i_minus_1
            y_i_minus_1 = y_i
            y_i = y_i_plus_1
            x_i_minus_3 += h
            i += 1
            solution.append((x_i_minus_3 + 4*h, y_i))
            rows.append([i, x_i_minus_3 + 4*h, y_i, func_solution.subs(x, x_i_minus_3 + 4*h), r])
        self._solution = solution
        table.add_rows(rows)
        return table


def input_data(equations, solution_methods) -> SolutionMethod:
    equation = None
    while True:
        print("Выберите диф. уравнение, которое требуется решить:")
        [print(f"{i + 1}. y' = {equation_iter[0].get_string()}") for i, equation_iter in enumerate(equations)]
        equation_num = int(input("Введите номер выбранного диф. уравнения...\n"))
        if equation_num < 1 or equation_num > len(equations):
            print("Номер диф. уравнения не найден, повторите ввод")
            continue
        equation = equations[equation_num - 1]
        break
    y_zero: float = float(input("Введите начальное условие y0 = y(x0)...\n"))
    while True:
        print("Задайте пределы дифференцирования [x0, xn]:")
        x_zero, x_n = (float(i) for i in input("Введите значения x0 и xn через пробел...\n").split())
        if x_zero == x_n:
            print("Значения должны быть различны")
            continue
        elif x_zero > x_n:
            print("Значение x0 должно быть меньше xn")
            continue
        break
    while True:
        h: float = float(input("Введите начальное значение для шага h...\n"))
        if h <= 0:
            print("Значение должно быть больше нуля")
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
        epsilon = input(
            "Введите погрешность вычислений (чтобы оставить значение по умолчанию - 0,01 нажмите Enter)...\n")
        if epsilon == '':
            solution_method = solution_method(equation[0], equation[1], y_zero, x_zero, x_n, h)
            break
        epsilon = float(epsilon)
        if epsilon <= 0:
            print("Значение погрешности должно быть больше нуля")
            continue
        solution_method = solution_method(equation[0], equation[1], y_zero, x_zero, x_n, h, epsilon)
        break
    return solution_method


def main():
    x = Symbol('x')
    y = Symbol('y')
    equations = (
        (Equation(y + (1 + x) * y ** 2, x, y), Equation(-1/x, x, y)),
    )
    solution_methods = (
        EulerMethod,
        RungeKuttaMethod,
        AdamsMethod
    )
    solution_method = input_data(equations, solution_methods)
    if solution_method is None:
        return
    table: PrettyTable = solution_method.calc()
    print(table)
    solution_method.draw()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    main()

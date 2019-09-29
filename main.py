import numpy as np
import random
from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import GrayCode
import matplotlib.pyplot as plt


class Population:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        xd = gray_to_bin(self.chromosome)
        xd = int(xd, 2)
        h = 10 / (2 ** 20)
        start = -5
        self.x = start + h * xd
        self.fc = self.function_c()
        self.fp = self.function_c(p=True)

    def function_c(self, p=False):
        if p:  # Функция приспособленности
            f = -1 * (0.4 * np.sin(0.8 * self.x + 0.9) + 0.7 * np.cos(0.8 * self.x - 1) * np.sin(7.3 * self.x)) + 6
        else:  # Целевая функция
            f = 0.4 * np.sin(0.8 * self.x + 0.9) + 0.7 * np.cos(0.8 * self.x - 1) * np.sin(7.3 * self.x)
        return f


def create_population(population):
    p = 0.8
    pm = 0.2
    par = []
    stack = []
    child = []
    population.sort(key=lambda x: x.fp, reverse=True)
    for i in range(len(population) - 25):
        stack.append(population[i])
        if len(stack) == 2:
            par.append(stack)
            stack = []
    # -------- Скрешивение ----- #
    for element in par:
        px = random.random()
        if px < p:
            # --- Формирования потомства у пары ----- #
            k = random.randint(0, len(population[0].chromosome) - 1)  # место разрыва
            piece = element[0].chromosome[0:k + 1]
            piece += element[1].chromosome[k + 1:]
            child.append(Population(piece))
            piece = element[1].chromosome[0:k + 1]
            piece += element[0].chromosome[k + 1:]
            child.append(Population(piece))
    # ---- Мутация --- #
    for i in range(len(child)):
        px = random.random()
        if px < pm:
            tmp = ''
            mutation = []
            mutation_count = 20
            for k in range(mutation_count):
                mutation.append(k)
            for l in range(len(child[i].chromosome)):

                if l in mutation:
                    if child[i].chromosome[l] == '0':
                        tmp += '1'
                    else:
                        tmp += '0'
                else:
                    if child[i].chromosome[l] == '0':
                        tmp += '0'
                    else:
                        tmp += '1'
            child[i].chromosome = tmp
    population.extend(child)
    return population


def roll(population):  # игра в рулетку на вылет
    s = 0
    new_population = []
    for someone in population:
        s += someone.fp
    while len(new_population) < 50:
        sp = 0
        rand = random.uniform(0, int(s))
        for i in range(len(population)):
            sp += population[i].fp
            if sp >= rand:
                if population[i] not in new_population:
                    new_population.append(population[i])
                break
    return new_population


def ygr(xx):
    yy = []
    for x in xx:
        y = 0.4 * np.sin(0.8 * x + 0.9) + 0.7 * np.cos(0.8 * x - 1) * np.sin(7.3 * x)
        yy.append(y)
    return yy


def plot_some(population, ax, start=-5, h=10 / (2 ** 20)):
    xn = []
    for elem in population:
        x = int(gray_to_bin(elem.chromosome), 2)
        x = start + h * x
        xn.append(x)
    y1 = ygr(xn)
    ax.scatter(xn, y1, color='red')
    ax.grid()


def main():
    count = 50
    xg = np.linspace(-5, 5, num=10000)
    yg = 0.4 * np.sin(0.8 * xg + 0.9) + 0.7 * np.cos(0.8 * xg - 1) * np.sin(7.3 * xg)
    population = []
    population1 = []
    population3 = []
    g = GrayCode(20)
    g = list(g.generate_gray())
    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 1, 1)
    ax_2 = fig.add_subplot(2, 2, 3)
    ax_3 = fig.add_subplot(2, 2, 4)
    for i in range(count):
        r = random.randint(0, len(g) - 1)
        population.append(Population(g[r]))
    for i in range(50):  # чисто поколений
        tmp_pop = create_population(population)
        population = []
        population = tmp_pop
        tmp_pop = []
        tmp_pop = roll(population)
        population = []
        population = tmp_pop
        tmp_pop = []
        if i == 0:
            ax_1.plot(xg, yg)
            for el in population:
                population1.append(el)
            plot_some(population, ax_1)
        if i == 3:
            ax_2.plot(xg, yg)
            for el in population:
                population3.append(el)
            plot_some(population, ax_2)
    ax_3.plot(xg, yg)
    plot_some(population, ax_3)
    plt.show()
    sum1 = 0
    sum3 = 0
    sum50 = 0
    for i in range(len(population)):
        sum1 += population1[i].fp
        sum3 += population3[i].fp
        sum50 += population[i].fp
    print("Средние приспособленности 1, 3 и 50 поколений:")
    print(str(sum1 / 50))
    print(str(sum3 / 50))
    print(str(sum50 / 50))

    print("1 популяция:")
    print("{0:^2} | {1:^6} | {2:^20} | {3:^6} | {4:^6}".format("№", "X", "Хромосома", "Fc(x)", "Fp(x)"))
    for i in range(len(population1)):
        print("{0:2} | {1:=6} | {2} | {3:=6} | {4:^6}".format(i, round(population1[i].x, 3), population1[i].chromosome,
                                                              round(population1[i].fc, 3), round(population1[i].fp, 3)))
    print("")
    print("3 популяция:")
    print("{0:^2} | {1:^6} | {2:^20} | {3:^6} | {4:^6}".format("№", "X", "Хромосома", "Fc(x)", "Fp(x)"))
    for i in range(len(population3)):
        print("{0:2} | {1:=6} | {2} | {3:=6} | {4:^6}".format(i, round(population3[i].x, 3), population3[i].chromosome,
                                                              round(population3[i].fc, 3), round(population3[i].fp, 3)))
    print("")
    print("50 популяция:")
    print("{0:^2} | {1:^6} | {2:^20} | {3:^6} | {4:^6}".format("№", "X", "Хромосома", "Fc(x)", "Fp(x)"))
    for i in range(len(population)):
        print("{0:2} | {1:=6} | {2} | {3:=6} | {4:^6}".format(i, round(population[i].x, 3), population[i].chromosome,
                                                              round(population[i].fc, 3), round(population[i].fp, 3)))


main()

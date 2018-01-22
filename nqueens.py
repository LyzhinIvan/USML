import random
import numpy as np

class Solver_8_queens:

    def __init__(self, pop_size=500, cross_prob=0.5, mut_prob=0.05):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.rnd = random.Random(42)
        pass

    def solve(self, min_fitness=0.9, max_epochs=100, verbose=False):
        population = [self.gen_chromosome() for _ in range(self.pop_size)]
        epoch = 0
        while True:
            epoch += 1
            if max_epochs is not None and epoch > max_epochs:
                break
            fits = list(map(self.fitness, population))
            max_fit = max(fits)
            if verbose:
                print('Epoch {}\tMax fitness: {:.2f}'.format(epoch, max_fit))
            if min_fitness is not None and max_fit >= min_fitness:
                break

            # selection
            parent_pool = [self.roulette_select(population, fits) for _ in range(self.pop_size)]

            # crossingover
            population = []
            for i in range(0, self.pop_size, 2):
                if self.rnd.random() < self.cross_prob:
                    ch1 = parent_pool[i]
                    ch2 = parent_pool[i+1]
                    new_ch1 = Chromosome()
                    new_ch2 = Chromosome()
                    k = self.rnd.randint(1, 7)
                    for index in range(k):
                        new_ch1.set(index, ch1.get(index))
                        new_ch2.set(index, ch2.get(index))
                    for index in range(k, 8):
                        new_ch1.set(index, ch2.get(index))
                        new_ch2.set(index, ch1.get(index))
                    population.append(new_ch1)
                    population.append(new_ch2)
                else:
                    population.append(parent_pool[i])
                    population.append(parent_pool[i+1])

            # mutation
            for ch in population:
                if self.rnd.random() < self.mut_prob:
                    index = self.rnd.randint(0, 7)
                    value = self.rnd.randint(0, 7)
                    ch.set(index, value)

        fits = list(map(self.fitness, population))
        best_index = np.argmax(fits)
        return fits[best_index], epoch, population[best_index].get_board_string()

    def gen_chromosome(self):
        chromosome = Chromosome()
        for i in range(8):
            chromosome.set(i, self.rnd.randint(0, 7))
        return chromosome

    def roulette_select(self, population, fits):
        total_fit = sum(fits)
        sum_fit = 0
        number = self.rnd.random() * total_fit
        for i in range(self.pop_size):
            if sum_fit < number <= sum_fit + fits[i]:
                return population[i]
            sum_fit += fits[i]
        return None

    @staticmethod
    def fitness(chromosome):
        conflicts = 0
        for row1 in range(8):
            col1 = chromosome.get(row1)
            for row2 in range(row1+1, 8):
                col2 = chromosome.get(row2)
                if col1 == col2 or row2 - row1 == abs(col2 - col1):
                    conflicts += 1
        return 1 - conflicts / 28


class Chromosome:

    def __init__(self, value=0):
        self.value = value
        pass

    def __str__(self):
        return '{:024b} ({})'.format(self.value, ''.join([str(self.get(i)) for i in range(8)]))

    def clear(self, index):
        self.value = self.value ^ (self.get(index) << (index * 3))

    def set(self, index, value):
        self.clear(index)
        self.value = self.value | (value << (index * 3))

    def get(self, index):
        return (self.value >> (index * 3)) & 7

    def get_board_string(self):
        board = ['+' * 8] * 8
        for row in range(8):
            col = self.get(row)
            board[row] = board[row][:col] + 'Q' + board[row][col+1:]
        return '\n'.join(board)


if __name__ == "__main__":
    solver = Solver_8_queens(pop_size=500)
    best_fit, epoch, board = solver.solve(min_fitness=1.0, max_epochs=500)
    print('Best solution')
    print('Fitness:', best_fit)
    print('Iterations:', epoch)
    print(board)

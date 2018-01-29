import random
import numpy as np
from collections import defaultdict

class Solver_8_queens:

    def __init__(self, pop_size=500, cross_prob=0.7, mut_prob=0.2, n=8):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.rnd = random.Random(42)
        self.n = n

        self.selection_functions = defaultdict(lambda : self.roulette_select)
        self.selection_functions['tournament'] = self.tournament_select
        self.selection_functions['rank'] = self.rank_select

        self.crossing_functions = defaultdict(lambda : self.one_point_cross)
        self.crossing_functions['two-point'] = self.two_point_cross

        self.mutation_functions = defaultdict(lambda : self.change_gen_mutate)
        self.mutation_functions['swap-gens'] = self.swap_gens_mutate
        self.mutation_functions['gray-code'] = self.gray_code_mutate

    def solve(self, min_fitness=0.9, max_epochs=100, verbose=False,
              select_method='tournament', cross_method='one-point', mutate_method='swap-gens'):
        population = [self.gen_chromosome() for _ in range(self.pop_size)]
        fits = list(map(self.fitness, population))
        epoch = 0

        while (min_fitness is None or max(fits) < min_fitness) and (max_epochs is None or epoch < max_epochs):
            epoch += 1
            if verbose:
                print('Epoch {}\tMax fitness: {:.5f}'.format(epoch, max(fits)))

            parent_pool = self.selection(population, fits, select_method)
            population = self.crossing(parent_pool, cross_method)
            population = self.mutation(population, mutate_method)

            fits = list(map(self.fitness, population))

        best_index = np.argmax(fits)
        return fits[best_index], epoch, population[best_index].get_board_string()

    def gen_chromosome(self):
        gens = list(range(self.n))
        self.rnd.shuffle(gens)
        return Chromosome(gens)

    def fitness(self, chromosome):
        conflicts = 0
        for row1 in range(self.n):
            col1 = chromosome[row1]
            for row2 in range(row1 + 1, self.n):
                col2 = chromosome[row2]
                if col1 == col2 or row2 - row1 == abs(col2 - col1):
                    conflicts += 1
        max_conflicts = self.n * (self.n - 1) / 2
        return 1 - conflicts / max_conflicts

    # selection methods

    def selection(self, population, fits, select_method):
        select_function = self.selection_functions[select_method]
        return [select_function(population, fits) for _ in range(self.pop_size)]

    def roulette_select(self, population, fits):
        total_fit = sum(fits)
        sum_fit = 0
        number = self.rnd.random() * total_fit
        for i in range(self.pop_size):
            if sum_fit < number <= sum_fit + fits[i]:
                return population[i]
            sum_fit += fits[i]
        return None

    def tournament_select(self, population, fits, group_size=2):
        group = self.rnd.choices(list(zip(population, fits)), k=group_size)
        group.sort(key=lambda pair: pair[1])
        winner = group[-1][0]
        return winner

    def rank_select(self, population, fits):
        sorted_population = sorted(list(zip(population, fits)), key=lambda pair: pair[1])
        weights = list(range(1, self.pop_size + 1))
        return self.rnd.choices(sorted_population, weights=weights, k=1)[0][0]

    # crossing methods

    def crossing(self, parent_pool, cross_method):
        cross_function = self.crossing_functions[cross_method]
        new_population = []
        for i in range(1, self.pop_size, 2):
            ch1, ch2 = parent_pool[i-1], parent_pool[i]
            if self.rnd.random() < self.cross_prob:
                ch1, ch2 = cross_function(ch1, ch2)
            new_population.append(ch1)
            new_population.append(ch2)
        return new_population

    def one_point_cross(self, parent1, parent2):
        cross_point = self.rnd.randrange(self.n)
        child1 = Chromosome(parent1[:cross_point] + parent2[cross_point:])
        child2 = Chromosome(parent2[:cross_point] + parent1[cross_point:])
        return child1, child2

    def two_point_cross(self, parent1, parent2):
        cross_point1 = self.rnd.randrange(self.n-1)
        cross_point2 = self.rnd.randrange(cross_point1, self.n)
        child1 = Chromosome(parent1[:cross_point1] + parent2[cross_point1:cross_point2] + parent1[cross_point2:])
        child2 = Chromosome(parent2[:cross_point1] + parent1[cross_point1:cross_point2] + parent2[cross_point2:])
        return child1, child2

    # mutation methods

    def mutation(self, population, mutate_method):
        mutation_function = self.mutation_functions[mutate_method]
        mutated_population = []
        for chromosome in population:
            if self.rnd.random() < self.mut_prob:
                chromosome = mutation_function(chromosome)
            mutated_population.append(chromosome)
        return mutated_population

    def change_gen_mutate(self, chromosome):
        mut_index = self.rnd.randrange(self.n)
        new_gene = self.rnd.randrange(self.n)
        return Chromosome(chromosome[:mut_index] + [new_gene] + chromosome[mut_index+1:])

    def swap_gens_mutate(self, chromosome):
        index1 = self.rnd.randrange(self.n - 1)
        index2 = self.rnd.randrange(index1 + 1, self.n)
        mutated_chromosome = chromosome.copy()
        mutated_chromosome[index1] = chromosome[index2]
        mutated_chromosome[index2] = chromosome[index1]
        return mutated_chromosome

    def gray_code_mutate(self, chromosome):
        """ Corresponds to change one bit in gray code, i.e. increment or decrement gene by one """
        mut_index = self.rnd.randrange(self.n)
        delta = self.rnd.choice([-1, 1])
        mutated_chromosome = chromosome.copy()
        mutated_chromosome[mut_index] = (chromosome[mut_index] + delta) % self.n
        return mutated_chromosome


class Chromosome:

    def __init__(self, gens):
        self.gens = gens
        pass

    def __str__(self):
        return '[{}]'.format(', '.join(map(str, self.gens)))

    def __setitem__(self, index, value):
        self.gens[index] = value

    def __getitem__(self, index):
        return self.gens[index]

    def copy(self):
        return Chromosome(self.gens[:])

    def get_board_string(self):
        n = len(self.gens)
        board = ['+' * n] * n
        for row in range(n):
            col = self[row]
            board[row] = board[row][:col] + 'Q' + board[row][col+1:]
        return '\n'.join(board)


if __name__ == "__main__":
    solver = Solver_8_queens(pop_size=500, n=20)
    best_fit, epoch, board = solver.solve(min_fitness=1.0, max_epochs=100, verbose=True,
                                          select_method='tournament', cross_method='one-point', mutate_method='swap-gens')
    print('Best solution')
    print('Fitness:', best_fit)
    print('Iterations:', epoch)
    print(board)

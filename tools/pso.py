"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

MAX_CORE = multiprocessing.cpu_count()
NCORE = int(MAX_CORE*0.6)
fitness_function_ = None


class PSO:
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(1.0, 0.05, 0.2, .5)):
        self.fitness_function = func
        self.dim = None
        self.position = np.zeros(0)
        self.historical_position = []
        self.velocity = None
        self.max_iteration = n_steps
        # self.w = parameters[0]
        self.w1 = parameters[0]
        self.w2 = parameters[1]
        self.c1 = parameters[2]
        self.c2 = parameters[3]
        self.npar = n_particles
        self.pbest_position = self.position  # particle best position
        self.pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])
        self.gbest_fitness_value = float('inf')
        self.gbest_position = None  # gloal best position
        self.historical_g_position = []
        self.gbest_correlation = 0.0
        self.gbest_r2_score = 0.0
        self.gbest_mse = 0.0
        self.range_var = None
        self.evol_best_fitness = np.zeros(self.max_iteration)
        self.evol_p_fitness = np.zeros((self.npar, self.max_iteration))
        self.gbest_position = np.array([np.inf for _ in range(self.npar)])
        self.historical_fitness = []

    def iterative_evaluation(self):
        return [self.fitness_function(pos) for pos in self.position]

    def initialize(self, range_var):
        self.range_var = range_var
        self.position = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])

        self.velocity = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])
        self.pbest_position = self.position

    @staticmethod
    def create_random_vector(range_var: list, vel=False) -> list:
        var = []
        for elem in range_var:
            temp = np.random.uniform(elem[0], elem[1])
            var.append(temp)
        return var


class PSOStandard(PSO):
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(0.8, 0.01, 1.5, 1.1)):
        super().__init__(func, n_particles, n_steps, parameters)

    def optimize(self, clip=True, args=()):
        iteration = 0
        W = self.w1
        min_state = None
        while iteration < self.max_iteration:
            self.historical_position.append(self.position.copy())
            # pool = multiprocessing.Pool(processes=NCORE)
            # zip_var = [[p_, args] for p_ in self.position]
            # result = pool.map(self.fitness_function, zip_var)
            # pool.close()
            # result = self.iterative_evaluation()
            # fitness = np.array([elem[0] for elem in result])
            fitness = np.array([self.fitness_function(p_) for p_ in self.position])
            self.historical_fitness.append(fitness)
            # result = [elem[1] for elem in result]
            self.pbest_position[fitness < self.pbest_fitness_value] = self.position[fitness < self.pbest_fitness_value]
            self.pbest_fitness_value[fitness < self.pbest_fitness_value] = fitness[fitness < self.pbest_fitness_value]
            best_particle_idx = np.argmin(fitness)
            best_fitness = fitness[best_particle_idx]

            # print("BEST: ", best_fitness, self.position[best_particle_idx])
            if best_fitness < self.gbest_fitness_value:
                self.gbest_fitness_value = best_fitness
                self.gbest_position = self.position[best_particle_idx]
                #min_state = result[best_particle_idx]

            self.historical_g_position.append(self.gbest_position)
            gbest = np.tile(self.gbest_position, (self.npar, 1))
            r = np.random.uniform(size=2)
            cognitive_comp = self.c1 * r[0] * (self.pbest_position - self.position)
            social_comp = self.c2 * r[1] * (gbest - self.position)
            self.velocity = W * self.velocity + cognitive_comp + social_comp
            self.position = self.velocity + self.position
            if clip:
                self.position = np.clip(self.position, np.array(self.range_var)[:, 0], np.array(self.range_var)[:, 1])

            W = self.w1 - (self.w1 - self.w2) * (iteration + 1) / self.max_iteration
            self.evol_best_fitness[iteration] = self.gbest_fitness_value
            self.evol_p_fitness[:, iteration] = self.pbest_fitness_value
            print("Train: ", iteration, "Fitness: ", self.gbest_fitness_value, "Worst: ", max(self.pbest_fitness_value),
                  "Best:", self.gbest_position)
            iteration += 1
        print("Finished")
        return (self.historical_g_position[-1], min_state, self.historical_position, self.historical_g_position,
                self.evol_p_fitness, self.evol_best_fitness)

    def get_gains(self):
        return self.gbest_position

    def plot(self):
        plt.figure()
        plt.plot(self.evol_best_fitness, color='red')
        plt.plot(self.evol_p_fitness, '.', color='blue')
        plt.show()


class PSOMagCalibration(PSO):
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(0.8, 0.01, 1.2, 1.5)):
        super().__init__(func, n_particles, n_steps, parameters)
        self.evol_best_fitness = []
        self.evol_p_fitness = []
        self.gbest_position = []
        self.historical_fitness = []
        self.current_iteration = 0

    def optimize(self, mag_sensor, mag_true, clip=True):
        dw = (self.w2 - self.w1) / 500
        W = np.max([self.w2 - dw * self.current_iteration, self.w1])

        self.historical_position.append(self.position.copy())
        fitness = np.array([self.fitness_function(pos, mag_sensor, mag_true) for pos in self.position])
        self.historical_fitness.append(fitness)
        self.pbest_position[fitness < self.pbest_fitness_value] = self.position[fitness < self.pbest_fitness_value]
        self.pbest_fitness_value[fitness < self.pbest_fitness_value] = fitness[fitness < self.pbest_fitness_value]
        best_particle_idx = np.argmin(fitness)
        best_fitness = fitness[best_particle_idx]

        # print("BEST: ", best_fitness, self.position[best_particle_idx])
        if best_fitness < self.gbest_fitness_value:
            self.gbest_fitness_value = best_fitness
            self.gbest_position = self.position[best_particle_idx]

        gbest = np.tile(self.gbest_position, (self.npar, 1))
        r = np.random.uniform(size=2)
        cognitive_comp = self.c1 * r[0] * (self.pbest_position - self.position)
        social_comp = self.c2 * r[1] * (gbest - self.position)
        self.velocity = W * self.velocity + cognitive_comp + social_comp
        self.position = self.velocity + self.position
        if clip:
            self.position = np.clip(self.position, np.array(self.range_var)[:, 0], np.array(self.range_var)[:, 1])

        self.evol_best_fitness.append(self.gbest_fitness_value)
        self.evol_p_fitness.append(self.pbest_fitness_value)
        self.historical_g_position.append(self.gbest_position)

        print("Train: ", self.current_iteration, "Fitness: ", self.gbest_fitness_value, "Worst: ", max(self.pbest_fitness_value),
              "Best:", self.gbest_position)
        self.current_iteration += 1

        return (self.historical_g_position[-1], self.historical_position, self.historical_g_position,
                self.evol_p_fitness, self.evol_best_fitness)

    def get_gains(self):
        return self.gbest_position

    @staticmethod
    def get_full_D(d_vector):
        d_ = np.zeros((3, 3))
        d_[0, 0] = d_vector[0]
        d_[1, 1] = d_vector[1]
        d_[2, 2] = d_vector[2]

        d_[0, 1] = d_vector[3]
        d_[0, 2] = d_vector[4]
        d_[1, 0] = d_vector[3]
        d_[2, 0] = d_vector[4]

        d_[1, 2] = d_vector[5]
        d_[2, 1] = d_vector[5]
        return d_

    def get_calibration(self):
        bias_ = self.gbest_position[:3]
        d_ = self.get_full_D(self.gbest_position[3:9])
        return bias_, d_
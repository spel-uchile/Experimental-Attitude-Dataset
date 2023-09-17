"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 24-08-2022
"""
import numpy as np
from sklearn.metrics import r2_score
import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class PSO:
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(1.5, 0.05, 0.2, .5)):
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

    def initialize(self, range_var):
        self.range_var = range_var
        self.position = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])

        self.velocity = np.array(
            [self.create_random_vector(range_var) for _ in range(self.npar)])
        self.pbest_position = self.position

    def plot_state_solution(self, min_state_full):
        fig_pso, ax_pso = plt.subplots(2, 2, figsize=(10, 8))
        ax_pso = ax_pso.flatten()
        ax_pso[0].set_ylabel("X-Position [km]")
        ax_pso[0].grid()
        ax_pso[1].set_ylabel("Y-Position [km]")
        ax_pso[1].grid()
        ax_pso[2].set_ylabel("Mass [kg]")
        ax_pso[2].grid()

        for min_state in min_state_full:
            ax_pso[0].plot(min_state[-1], [elem[0] for elem in min_state[0]])
            ax_pso[1].plot(min_state[-1], [elem[1] for elem in min_state[0]])
            ax_pso[2].plot(min_state[-1], min_state[2])
            ax_pso[3].plot([elem[0] * 1e-3 for elem in min_state[0]], [elem[1] * 1e-3 for elem in min_state[0]])

    def plot_historical_position(self):
        fig, axes = plt.subplots(len(self.range_var), 1)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            ax.plot(np.arange(1, self.max_iteration + 1),
                    np.array(self.historical_position).T[i].T, lw=0.7, color='b')
            ax.plot(np.arange(1, self.max_iteration + 1),
                              np.array(self.historical_g_position).T[i], lw=1.2, color='r')
            ax.grid()

    @staticmethod
    def create_random_vector(range_var: list, vel=False) -> list:
        var = []
        for elem in range_var:
            temp = np.random.uniform(elem[0], elem[1])
            var.append(temp)
        return var

    def show_map2(self):
        xi, yi, zi = self.get_gridmap_approximation()
        # Crear una figura 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar la superficie
        surf = ax.plot_surface(xi, yi, zi, cmap='jet')

        # Agregar puntos originales
        ax.scatter(np.array(self.historical_position).T[0].flatten(),
                   np.array(self.historical_position).T[1].flatten(),
                   np.log10(self.historical_fitness).flatten(),
                   color='red', s=15, label='Positions')
        plt.colorbar(surf, ax=ax, label='fitness (Log10)')
        # Personalizar la apariencia
        ax.set_xlabel('Angular ignition [rad]')
        ax.set_ylabel('Engine Diameter [m]')
        ax.set_zlabel('fitness')
        ax.set_title('PSO evaluation')
        ax.legend()

    def show_map(self):
        cmap = plt.get_cmap('jet').reversed()  # I only want 4 colors from this cmap
        x = np.array(self.historical_position).T[0]
        y = np.array(self.historical_position).T[1]
        value = np.log10(self.historical_fitness)
        plt.figure()
        plt.scatter(x.T, y.T, c=value, cmap=cmap)
        plt.colorbar(label='Log10')

    def get_gridmap_approximation(self, res=100):
        x = np.array(self.historical_position).T[0].flatten()
        y = np.array(self.historical_position).T[1].flatten()
        value = np.log10(self.historical_fitness).flatten()
        xi = np.linspace(min(x), max(x), res)
        yi = np.linspace(min(y), max(y), res)
        xi, yi = np.meshgrid(xi, yi)
        # Interpolación de los puntos para obtener valores z en la superficie
        zi = griddata((x, y), value, (xi, yi), method='cubic', fill_value=np.max(value))
        return xi, yi, zi


class PSOStandard(PSO):
    def __init__(self, func, n_particles=100, n_steps=200, parameters=(1.0, 0.01, 1.2, 1.5)):
        super().__init__(func, n_particles, n_steps, parameters)

    def optimize(self):
        iteration = 0
        W = self.w1
        min_state = None
        while iteration < self.max_iteration:
            self.historical_position.append(self.position.copy())
            pool = multiprocessing.Pool(processes=2)
            result = pool.map(self.fitness_function, self.position)
            pool.close()
            # result = [self.fitness_function(pos) for pos in self.position]
            fitness = np.array([elem[0] for elem in result])
            self.historical_fitness.append(fitness)
            result = [elem[1] for elem in result]
            self.pbest_position[fitness < self.pbest_fitness_value] = self.position[fitness < self.pbest_fitness_value]
            self.pbest_fitness_value[fitness < self.pbest_fitness_value] = fitness[fitness < self.pbest_fitness_value]
            best_particle_idx = np.argmin(fitness)
            best_fitness = fitness[best_particle_idx]

            # print("BEST: ", best_fitness, self.position[best_particle_idx])
            if best_fitness < self.gbest_fitness_value:
                self.gbest_fitness_value = best_fitness
                self.gbest_position = self.position[best_particle_idx]
                min_state = result[best_particle_idx]

            self.historical_g_position.append(self.gbest_position)
            gbest = np.tile(self.gbest_position, (self.npar, 1))
            r = np.random.uniform(size=(self.npar, 2))
            cognitive_comp = self.c1 * np.diag(r[:, 0]) @ (self.pbest_position - self.position)
            social_comp = self.c2 * np.diag(r[:, 1]) @ (gbest - self.position)
            self.velocity = W * self.velocity + cognitive_comp + social_comp
            self.position = self.velocity + self.position
            self.position[:, 0] = self.position[:, 0] % (2 * np.pi)
            self.position[:, 1] = np.clip(self.position[:, 1], np.array(self.range_var)[1, 0], np.array(self.range_var)[1, 1])

            W = self.w1 - (self.w1 - self.w2) * (iteration + 1) / self.max_iteration
            self.evol_best_fitness[iteration] = self.gbest_fitness_value
            self.evol_p_fitness[:, iteration] = self.pbest_fitness_value
            print("Train: ", iteration, "Fitness: ", self.gbest_fitness_value, "Worst: ", max(self.pbest_fitness_value), "Best:", self.gbest_position)
            iteration += 1
        print("Finished")

        self.plot_state_solution(min_state)
        self.plot_historical_position()
        return self.historical_g_position[-1]

    def get_gains(self):
        return self.gbest_position
"""
Created by Elias Obreque
Date: 30-03-2024
email: els.obrq@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

# Definición de funciones de movimiento y observación
def move(x, u):
    # Simulación de movimiento: Agregamos ruido gaussiano al movimiento
    return x + u + np.random.normal(0, 1, x.shape)

def observe(x):
    # Simulación de la observación: Agregamos ruido gaussiano a la observación
    return x + np.random.normal(0, 1, x.shape)

# Función para inicializar partículas aleatorias
def initialize_particles(num_particles, space_range):
    particles = []
    for _ in range(num_particles):
        particle = np.random.uniform(space_range[0], space_range[1], size=(2,))
        particles.append(particle)
    return np.array(particles)

# Función para calcular el peso de las partículas basado en la observación
def compute_weights(particles, observation):
    weights = []
    for particle in particles:
        observation_prob = np.prod(np.exp(-(particle - observation)**2 / 2))
        weights.append(observation_prob)
    weights /= np.sum(weights)
    return weights

# Función para resamplear partículas basado en los pesos
def resample_particles(particles, weights):
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

# Configuración inicial
true_position = np.array([5, 5])  # Posición verdadera del robot
initial_guess = np.array([0, 0])  # Suposición inicial del filtro
num_particles = 100  # Número de partículas
space_range = [-10, 10]  # Rango del espacio

# Inicialización de partículas
particles = initialize_particles(num_particles, space_range)

# Visualización de la posición verdadera y las partículas iniciales
plt.scatter(true_position[0], true_position[1], color='red', label='True Position')
plt.scatter(particles[:, 0], particles[:, 1], color='blue', alpha=0.5, label='Particles')
plt.legend()
plt.title('Initial State')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(space_range)
plt.ylim(space_range)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Movimiento y observación
movement = np.array([1, 1])  # Movimiento simulado
observation = observe(true_position)  # Observación simulada

# Actualización de partículas y pesos
particles = move(particles, movement)
weights = compute_weights(particles, observation)

# Resampleo de partículas
particles = resample_particles(particles, weights)

# Visualización después de la actualización
plt.scatter(true_position[0], true_position[1], color='red', label='True Position')
plt.scatter(particles[:, 0], particles[:, 1], color='blue', alpha=0.5, label='Particles')
plt.legend()
plt.title('After Movement and Observation Update')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(space_range)
plt.ylim(space_range)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


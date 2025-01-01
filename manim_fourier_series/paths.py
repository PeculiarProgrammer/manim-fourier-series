import platform

import numpy as np
from tqdm import trange


# Adapted from https://github.com/diego-vicente/som-tsp
def self_organising_maps(
    points: np.ndarray, iterations: int = 0, learning_rate: float = 0.8
) -> np.ndarray:
    # Make sure that all points are unique and that the population size is 8 times the number of cities
    points = np.unique(points)
    n = len(points) * 8

    # Generate an adequate network of neurons
    network = np.random.uniform(
        min(points.real), max(points.real), n
    ) + 1j * np.random.uniform(min(points.imag), max(points.imag), n)

    if iterations == 0:
        iterations = min(100000, int(np.log(n) / -np.log(0.9997)))

    for _ in trange(
        iterations,
        desc="Optimising shape",
        ascii=True if platform.system() == "Windows" else None,
        leave=False,
    ):
        # Choose a random city
        point = np.random.choice(points)
        idx = abs(network - point).argmin()

        # Generate a filter that applies changes to the winner's gaussian
        # Compute the circular network distance to the center
        deltas = np.absolute(idx - np.arange(len(network)))
        distances = np.minimum(deltas, len(network) - deltas)
        # Impose an upper bound on the radix to prevent NaN and blocks
        radix = max(n / 10, 1)
        # Compute Gaussian distribution around the given center
        gaussian = np.exp(-(distances**2) / (2 * radix**2))

        # Update the network's weights (closer to the city)
        network += gaussian * learning_rate * (point - network)

        # Decay the variables
        learning_rate *= 0.99997
        n *= 0.9997

    # Find the route from the network
    route = points[np.argsort([np.argmin(abs(network - point)) for point in points])]
    return route


def greedy_shortest_path(points: np.ndarray) -> np.ndarray:
    # Make sure that all the points are unique
    points = np.unique(points)

    # Initialise an empty path
    path = np.ndarray(len(points), dtype=complex)
    points *= 1j
    path[-1] = min(points[points.imag == min(points.imag)]) / 1j
    points /= 1j

    for i in trange(
        len(points),
        desc="Optimising shape",
        ascii=True if platform.system() == "Windows" else None,
        leave=False,
    ):
        # Find the nearest point
        nearest = np.abs(points - path[i - 1]).argmin()
        # Set the next element in the path to this value
        path[i] = points[nearest]
        # Delete that point
        points = np.delete(points, nearest)

    return path


def optimized_shortest_path(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    # Precompute greedy path
    points = greedy_shortest_path(points)

    for _ in range(iterations):
        # Start with points that are already in place
        order = np.argsort(
            abs(points - np.roll(points, 1)) + abs(points - np.roll(points, -1))
        )
        for i in order:
            point = points[i]
            # Remove the point
            points = np.delete(points, i)
            # Find the best place to insert
            distances = abs(points - point)
            heuristic = (
                distances + np.roll(distances, 1) - abs(points - np.roll(points, 1))
            )
            nearest = heuristic.argmin()
            # Insert back in the shortest place
            points = np.insert(points, nearest, point)

    return points

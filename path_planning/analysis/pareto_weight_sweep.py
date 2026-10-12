"""
Approximate-Pareto-front experiment: re-run the weighted-sum GA (differential
evolution over the coverage-planning fitness) at several DISTINCT weight
vectors (not the +/-20% single-weight sensitivity sweep already in the
manuscript's script), and collect the resulting (coverage, info_gain,
path_length, smoothness) point for each run. Plotting these points is the
standard cheap way to approximate a Pareto front from a weighted-sum method
without switching to a true multi-objective GA (e.g., NSGA-II).

NOTE ON DATA: the original sample_area.txt / covplan field data used to
produce the manuscript's numbers (36.3% mean improvement, etc.) was not
available in this environment. This script uses a synthetic hexagonal AoI
of comparable size instead, built with the 'covplan' package (the same one
imported in the user's script). Absolute numbers here will NOT match the
manuscript -- only the shape of the trade-off (spread of points across
weight vectors) is meaningful as a demonstration of the method.
"""
import sys
sys.path.insert(0, '.')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

from covplan.lines import getPoly, numPoly
from covplan.field import Field
from shapely.geometry import Polygon


class FastGaussianProcess:
    def __init__(self, length_scale=10.0, signal_variance=1.0, noise_variance=0.1):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.X_train = []
        self.y_train = []
        self._K_inv = None
        self._K = None

    def kernel_vectorized(self, X1, X2):
        X1 = np.array(X1)
        X2 = np.array(X2)
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        return self.signal_variance * np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def predict(self, X_test):
        if len(self.X_train) == 0:
            return np.zeros(len(X_test)), self.signal_variance * np.ones(len(X_test))
        X_test = np.array(X_test)
        K_s = self.kernel_vectorized(X_test, self.X_train)
        mean = K_s @ self._K_inv @ self.y_train
        K_ss = self.kernel_vectorized(X_test, X_test)
        variance = np.diag(K_ss) - np.sum(K_s @ self._K_inv * K_s, axis=1)
        variance = np.maximum(variance, 0)
        return mean, variance

    def update(self, X_new, y_new):
        self.X_train.append(X_new)
        self.y_train.append(y_new)
        n = len(self.X_train)
        if n == 1:
            self._K = np.array([[self.kernel_vectorized([X_new], [X_new])[0, 0]]])
        else:
            new_K = np.zeros((n, n))
            new_K[:n - 1, :n - 1] = self._K
            for i in range(n):
                k_val = self.kernel_vectorized([self.X_train[i]], [X_new])[0, 0]
                new_K[i, n - 1] = k_val
                new_K[n - 1, i] = k_val
            self._K = new_K
        K_noisy = self._K + self.noise_variance * np.eye(n)
        self._K_inv = np.linalg.inv(K_noisy)


class WeightVectorOptimizer:
    """Same fitness/DE machinery as the user's FastGeneticAIOptimizer, trimmed
    to what the Pareto sweep needs, and parameterised by an arbitrary weight
    vector rather than a single +/-20% perturbation of one weight."""

    def __init__(self, input_file, population_size=15, max_generations=25,
                 mutation_factor=0.8, crossover_prob=0.7, weights=(0.35, 0.25, 0.25, 0.15)):
        self.input_file = input_file
        self.pop_size = population_size
        self.max_gen = max_generations
        self.F = mutation_factor
        self.CR = crossover_prob
        self.current_weights = weights
        self.bounds = [(0, 90), (8, 15), (2, 6), (-20, 20), (-20, 20)]
        for _ in range(3):
            self.bounds.append((-20, 20))
        self._cached_area = None
        self._cached_boundary = None
        self._compute_field_area()

    def _compute_field_area(self):
        field = Field(self.input_file, 10, 1, 45)
        field.headlandGen()
        boundary = getPoly(field.data, 1)
        self._cached_boundary = boundary
        self._cached_area = Polygon(boundary).area

    def decode_parameters(self, x):
        return {'theta': x[0], 'width': x[1], 'num_clusters': int(round(x[2])),
                'offset_x': x[3], 'offset_y': x[4], 'phi_values': x[5:]}

    @lru_cache(maxsize=256)
    def _cached_generate_field(self, theta, width, num_clusters):
        try:
            field = Field(self.input_file, width, 1, theta)
            field.headlandGen()
            field.trackGen()
            field.cluster(num_clusters)
            field.tsp_opt()
            field.trajGen(turning_radius=5.0)
            return field
        except Exception:
            return None

    def generate_field(self, theta, width, num_clusters):
        return self._cached_generate_field(round(theta, 1), round(width, 1), num_clusters)

    def _point_in_polygon(self, point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if (y1 == y2) and (y == y1):
                if min(x1, x2) <= x <= max(x1, x2):
                    return True
            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                if x_intersect > x:
                    inside = not inside
        return inside

    def calculate_coverage_fast(self, field):
        total_area = self._cached_area
        if total_area <= 0:
            return 0.0
        covered_points, total_samples = 0, 0
        for track in field.tracks:
            track_vector = track[1] - track[0]
            track_length = np.linalg.norm(track_vector)
            if track_length == 0:
                continue
            n_samples = max(5, int(track_length / 10))
            for i in range(n_samples):
                t = i / (n_samples - 1) if n_samples > 1 else 0.5
                point = track[0] + t * track_vector
                if self._point_in_polygon(point, self._cached_boundary):
                    covered_points += 1
                total_samples += 1
        return min(1.0, covered_points / total_samples) if total_samples > 0 else 0.0

    def calculate_information_gain_fast(self, path, gp_model):
        if len(path) < 10:
            return 0.0
        total_variance = 0.0
        step = max(1, len(path) // 30)
        rng = np.random.RandomState(0)
        for i in range(0, len(path), step):
            point = path[i][:2]
            _, variance = gp_model.predict([point])
            total_variance += variance[0]
            simulated_measurement = 5.0 + 0.1 * rng.randn()
            gp_model.update(point, simulated_measurement)
        return total_variance / (1.0 + len(path))

    def calculate_smoothness_fast(self, path):
        if len(path) < 5:
            return 1.0
        step = max(1, len(path) // 50)
        sampled_path = path[::step]
        if len(sampled_path) < 3:
            return 0.8
        curvatures = []
        for i in range(1, len(sampled_path) - 1):
            p1 = np.array(sampled_path[i - 1][:2])
            p2 = np.array(sampled_path[i][:2])
            p3 = np.array(sampled_path[i + 1][:2])
            v1, v2 = p2 - p1, p3 - p2
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            curvatures.append(angle / (n1 + n2))
        if not curvatures:
            return 0.8
        kappa_max = 0.5
        return 1.0 / (1.0 + max(0.0, max(curvatures) - kappa_max))

    def calculate_metrics(self, field):
        if field is None or not hasattr(field, 'tracks'):
            return None
        tracks = field.tracks
        track_length = sum(np.linalg.norm(t[1] - t[0]) for t in tracks)
        turn_dist = getattr(field, 'turn_dist', 0)
        total_distance = track_length + turn_dist
        coverage = self.calculate_coverage_fast(field)
        gp_model = FastGaussianProcess(length_scale=10.0, signal_variance=1.0)
        info_gain = self.calculate_information_gain_fast(field.traj, gp_model) if len(field.traj) > 50 else 0.0
        smoothness = self.calculate_smoothness_fast(field.traj) if len(field.traj) > 10 else 0.5
        return {'total_distance': total_distance, 'coverage': coverage,
                'info_gain': info_gain, 'smoothness': smoothness, 'num_tracks': len(tracks)}

    def fitness_function(self, x):
        params = self.decode_parameters(x)
        field = self.generate_field(params['theta'], params['width'], params['num_clusters'])
        if field is None:
            return 1e6
        metrics = self.calculate_metrics(field)
        if metrics is None:
            return 1e6
        w1, w2, w3, w4 = self.current_weights
        phi_diversity = np.std(params['phi_values']) if len(params['phi_values']) > 0 else 0
        total_info_gain = metrics['info_gain'] + 0.01 * phi_diversity
        fitness = (w1 * metrics['coverage'] + w2 * total_info_gain
                   - w3 * metrics['total_distance'] + w4 * metrics['smoothness'])
        return -fitness

    def optimize(self):
        result = differential_evolution(
            func=self.fitness_function, bounds=self.bounds, popsize=self.pop_size,
            maxiter=self.max_gen, mutation=self.F, recombination=self.CR,
            disp=False, polish=False, seed=42)
        best_params = self.decode_parameters(result.x)
        field = self.generate_field(best_params['theta'], best_params['width'], best_params['num_clusters'])
        metrics = self.calculate_metrics(field)
        return best_params, metrics, -result.fun


def run_pareto_sweep(input_file):
    # Distinct weight vectors spanning different planner "personalities":
    # balanced, coverage-first, info-gain-first, length-first, smoothness-first,
    # and the manuscript's own base weights, all normalised to sum to 1.
    raw_vectors = {
        'Balanced':          (0.25, 0.25, 0.25, 0.25),
        'Manuscript base':   (0.35, 0.25, 0.25, 0.15),
        'Coverage-first':    (0.55, 0.15, 0.15, 0.15),
        'Info-gain-first':   (0.15, 0.55, 0.15, 0.15),
        'Path-length-first': (0.15, 0.15, 0.55, 0.15),
        'Smoothness-first':  (0.15, 0.15, 0.15, 0.55),
    }
    results = []
    for name, w in raw_vectors.items():
        w = tuple(np.array(w) / sum(w))
        print(f"--- Running DE for weight vector '{name}': "
              f"w1={w[0]:.2f}, w2={w[1]:.2f}, w3={w[2]:.2f}, w4={w[3]:.2f} ---")
        opt = WeightVectorOptimizer(input_file, weights=w)
        best_params, metrics, fitness = opt.optimize()
        row = {'name': name, 'w1': w[0], 'w2': w[1], 'w3': w[2], 'w4': w[3],
               'theta': best_params['theta'], 'spacing': best_params['width'],
               'clusters': best_params['num_clusters'], 'fitness': fitness, **metrics}
        results.append(row)
        print(f"    -> coverage={metrics['coverage']:.3f}, info_gain={metrics['info_gain']:.4f}, "
              f"path_length={metrics['total_distance']:.1f} m, smoothness={metrics['smoothness']:.3f}")
    return results


def plot_pareto_approx(results, out_path='pareto_front_approx.png'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    xs = [r['total_distance'] for r in results]
    ys = [r['coverage'] for r in results]
    sizes = [200 + 3000 * r['info_gain'] for r in results]
    sc = ax.scatter(xs, ys, s=sizes, c=[r['w3'] for r in results], cmap='viridis',
                     edgecolor='black', linewidth=1.2, zorder=3)
    for r in results:
        ax.annotate(r['name'], (r['total_distance'], r['coverage']),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel('Path length (m)  [minimize]')
    ax.set_ylabel('Coverage completeness  [maximize]')
    ax.set_title('Approximate Pareto front: path length vs. coverage\n(marker size = information gain, color = w3)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('w3 (path-length weight)')

    ax2 = axes[1]
    xs2 = [r['total_distance'] for r in results]
    ys2 = [r['smoothness'] for r in results]
    sc2 = ax2.scatter(xs2, ys2, s=200, c=[r['w4'] for r in results], cmap='plasma',
                       edgecolor='black', linewidth=1.2, zorder=3)
    for r in results:
        ax2.annotate(r['name'], (r['total_distance'], r['smoothness']),
                     textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax2.set_xlabel('Path length (m)  [minimize]')
    ax2.set_ylabel('Smoothness score  [maximize]')
    ax2.set_title('Approximate Pareto front: path length vs. smoothness\n(color = w4, smoothness weight)')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('w4 (smoothness weight)')

    plt.suptitle('Approximate Pareto front from re-running the weighted-sum GA\n'
                  'at distinct weight vectors (synthetic AoI -- demo, not manuscript data)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    input_file = 'sample_area_L.txt'
    results = run_pareto_sweep(input_file)
    with open('pareto_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    plot_pareto_approx(results)
    print("\nSummary table:")
    print(f"{'Weight vector':<18}{'theta':>7}{'spacing':>9}{'n':>4}{'coverage':>10}{'info_gain':>11}{'length(m)':>11}{'smooth':>8}")
    for r in results:
        print(f"{r['name']:<18}{r['theta']:>7.1f}{r['spacing']:>9.1f}{r['clusters']:>4d}"
              f"{r['coverage']:>10.3f}{r['info_gain']:>11.4f}{r['total_distance']:>11.1f}{r['smoothness']:>8.3f}")

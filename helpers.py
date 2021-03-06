import numpy as np
from plot_params import *
from scipy.spatial.distance import cdist
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation

class RunSimulation():
    """
    Class that handles running a single simulation.
    """
    def __init__(
        self, 
        n_charges, 
        circle_radius, 
        n_steps, 
        stepsize_init, 
        stepsize_final, 
        random_influence, 
        force_influence, 
        temperature
    ):
        self.n_charges = n_charges
        self.circle_radius = circle_radius
        self.n_steps  = n_steps
        self.stepsize_init = stepsize_init
        self.stepsize_final = stepsize_final
        self.random_influence = random_influence
        self.force_influence = force_influence
    
         # initialize charges
        r = np.random.rand(n_charges)
        theta = np.random.rand(n_charges) * 2 * np.pi
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        self.pos = np.stack((x, y), axis=1)

        # initialize loop arrays
        self.pos_history = np.zeros((n_steps, n_charges, 2))
        self.stepsize = np.linspace(stepsize_init, stepsize_final, n_steps)
        self.temperature = temperature


    def run(self):
        """
        Main loop of the simulation.
        """
        # loop over temperatures
        for j in tqdm(range(self.n_steps)):
            self.pos_history[j] = self.pos

            # loop charge
            for i in range(self.n_charges):
                energy_before = self.total_energy(self.pos)
                possible_pos = self.random_move_with_drift(
                    deepcopy(self.pos), i, j
                )
                energy_after = self.total_energy(possible_pos)
                if self.accept_move(energy_after, energy_before, j):
                    self.pos = possible_pos

        return self.pos

    def total_energy(self, pos):
        """
        Returns the total energy of a given configuration.
        """
        dist = cdist(pos, pos)
        return (1 / dist[dist != 0]).sum() / 2


    def accept_move(self, energy_after, energy_before, j):
        """
        Determines wheter or not to accept a move given the energy before and after making the move.
        """
        delta_energy = energy_after - energy_before
        if delta_energy < 0:
            return True
        alpha = np.exp(- delta_energy / self.temperature[j]) 
        if np.random.rand() <= alpha:
            return True
        return False

    def get_force(self, pos, i):
        """
        Return the force acting on the i'th particle.
        """

        # total distance to charge i
        r = cdist(pos, pos)
        r = r[i]
        r = r[r != 0]

        # x distance
        r_x = pos[:, 0].reshape(-1 , 1) - pos[:, 0]
        r_x = r_x[i]
        r_x = r_x[r_x != 0]

        # y distance
        r_y = pos[:, 1].reshape(-1 , 1) - pos[:, 1]
        r_y = r_y[i]
        r_y = r_y[r_y != 0]

        # total force 
        force = np.stack((r_x / np.abs(r ** 3), r_y / np.abs(r ** 3)), axis=1)
        total_force = force.sum(axis=0)

        # normalized size 1
        normal_force_move = total_force / np.linalg.norm(total_force)

        return normal_force_move


    def get_random(self):
        """ 
        Generate a random move.
        """
        random_move = np.random.uniform(-1, 1, 2)
        normal_random_move = random_move / np.linalg.norm(random_move)
        return normal_random_move


    def get_step(self, normal_force_move, normal_random_move, j):
        """
        Combine the influence of the random and force moves into a single move.
        """

        combined_step = (normal_random_move * self.random_influence 
                      + normal_force_move * self.force_influence)
        normal_step =  combined_step / np.linalg.norm(combined_step)
        step =  normal_step * self.stepsize[j]
        return step
    
    def move_across_circle(self, pos, i, step):
        """
        Move a particle across the circle if it wants to exit the circle. 

        """
        step_size = np.linalg.norm(step)
        dist_to_circle = (
            self.circle_radius - np.sqrt(pos[i, :][0] ** 2 + pos[i, :][1] ** 2)
        )
        step_angle = np.arctan2(step[1], step[0])
        point_angle = np.arctan2(pos[i, 1], pos[i, 0])
        angle_dif = step_angle - point_angle
        arc_length = step_size - dist_to_circle
        arc_angle = np.copysign(arc_length / self.circle_radius,angle_dif)
        pos[i, 0] = np.cos(point_angle+arc_angle) * self.circle_radius
        pos[i, 1] = np.sin(point_angle+arc_angle) * self.circle_radius
        return pos

    def random_move_with_drift(self, pos, i, j):
        """
        Returns a position array where the i'th partcicle has made its move.
        """
        og = deepcopy(pos)

        if self.force_influence > 0:
            normal_force_move = self.get_force(pos, i)
        else:
            normal_force_move = 0
            
        if self.random_influence > 0:
            normal_random_move = self.get_random()
        else:
            normal_random_move = 0

        step = self.get_step(normal_force_move, normal_random_move, j)
        pos[i] = og[i] + step
        
        # check if outside of circle
        if np.sqrt(pos[i][0] ** 2 + pos[i][1] ** 2) < self.circle_radius:
            return pos
        return self.move_across_circle(pos, i, step)


class PostProcess():
    """
    Class that handles the creating of figures and scores after the simulation. 
    """
    def __init__(self, best_runs, best_of_best_index=None):
        self.best_runs = best_runs
        self.best_of_best_index = best_of_best_index

    
    def make_figure(self, title, xlabel, ylabel, savepath, show=True):
        """
        Creates a figure of the resulting configurations. 
        """
        standard_pos = self.best_runs[self.best_of_best_index]
        best_runs_without_best = np.delete(
            self.best_runs, self.best_of_best_index, axis=0
        )
        plt.figure(figsize=(6, 6))
        plt.box(False)

        plt.scatter(
            standard_pos[:, 0], 
            standard_pos[:, 1], 
            s=300, zorder=-100, 
            color='tab:orange', 
            edgecolors='darkgoldenrod'
        )

        if len(best_runs_without_best) > 0:
            energies = [self.total_energy(br) for br in best_runs_without_best]
            energies = (energies - min(energies)) / (max(energies) - min(energies))

            for e, br in zip(energies, best_runs_without_best):
                plot_br = self.get_minimized_config(standard_pos, br)
                plt.scatter(
                    plot_br[:, 0], 
                    plot_br[:, 1], 
                    color=plt.cm.get_cmap('Blues')(e),
                    edgecolors=plt.cm.get_cmap('Blues')(e + 0.3), 
                    s=150*e, 
                    zorder=-e
                )

        circle = plt.Circle((0, 0), 1, fill=False)
        plt.gca().add_patch(circle)
        plt.yticks(np.linspace(-1, 1, 5))
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
    
    
    def energy_overview(self):
        """ 
        Gives information about the energy of several runs. 
        """
        energies = [self.total_energy(br) for br in self.best_runs]
        lowest = min(energies)
        highest = max(energies)
        mean = np.mean(energies)
        std = np.std(energies)
        return lowest, highest, mean, std
        

    def variability_score(self):
        """
        Gives a score based on how much the input configurations are alike. 
        """
        
        scores_mean = np.zeros((len(self.best_runs)))
        # for i, br in enumerate(self.best_runs):
        for i in tqdm(range(len(self.best_runs)), desc='variability'):
            br = self.best_runs[i]
            optimized_configs = [
                self.get_minimized_config(br, i) for i in self.best_runs
            ]
            difference_scores = [
                self.difference_score(0, br, i) for i in optimized_configs
            ]
            scores_mean[i] = np.mean(difference_scores)
        score = scores_mean.mean()
        return score
    
    def difference_score(self, theta_rotation, standard_pos, pos):
        """
        Returns a measure of how much two configurations differ from each other. 
        """
        pos = self.rotation(pos, theta_rotation)
        return cdist(standard_pos, pos).min(axis=1).sum()

    def get_minimized_config(self, standard_pos, pos, precision=3600):
        """
        Finds the rotation between two configuration that minimized the difference between them.
        """

        radians = np.linspace(0, 2 * np.pi, precision)
        scores = [self.difference_score(r, standard_pos, pos) for r in radians]
        best_radian = radians[np.argmin(scores)]
        return self.rotation(pos, best_radian)   
    
    def rotation(self, pos, theta_rotation):
        """
        Rotation a configuration theta radians.
        """
        x = pos[:, 0]
        y = pos[:, 1]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        theta = theta + theta_rotation
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        pos = np.stack((x, y), axis=1)
        return pos
    
    def total_energy(self, pos):
        """
        Return the total energy in a configuration. 
        """
        dist = cdist(pos, pos)
        return (1 / dist[dist != 0]).sum() / 2
    

class CircleCharges():
    """ 
    Class that handles running multiple configurations. 
    """
    def __init__(
        self, 
        n_charges, 
        n_attemps_per_run, 
        n_runs, 
        circle_radius, 
        n_steps, 
        stepsize_init, 
        stepsize_final, 
        random_influence, 
        force_influence, 
        temperature
    ):

        self.n_charges = n_charges
        self.n_attemps_per_run = n_attemps_per_run
        self.n_runs = n_runs
        self.circle_radius = circle_radius
        self.n_steps  = n_steps
        self.stepsize_init = stepsize_init
        self.stepsize_final = stepsize_final
        self.random_influence = random_influence
        self.force_influence = force_influence
        self.temperature = temperature

        self.best_runs = np.zeros((self.n_runs, self.n_charges, 2))


    def produce_figure(self, title=None, xlabel=None, ylabel=None, savepath=None):
        """
        Create a figure of the result of the simulation.
        """
        best_of_best_index = self.choose_best_run(self.best_runs)[1]
        pp = PostProcess(self.best_runs, best_of_best_index)
        pp.make_figure(title, xlabel, ylabel, savepath)


    def get_results(self, variability=False):
        """
        Returns some metrics regarding the result of the run.
        """
        pp = PostProcess(self.best_runs)
        if variability:
            var_score = pp.variability_score()
        else:
            var_score = 1
        lowest, highest, mean, std = pp.energy_overview()
        dct = {
            'variability': var_score,
            'min_energy': lowest, 
            'max_energy': highest,
            'mean_energy': mean, 
            'std_energy': std
        }
        return dct


    def get_variability_score(self):
        """ 
        Obtains and return a socre that indicates how much several configurations are alike.
        """
        pp = PostProcess(self.best_runs)
        score = pp.variability_score()
        return score
    

    def run(self):
        """
        Main loop of the simulation. Run the simulation multiple times and keep the best results.
        """
        for i in tqdm(range(self.n_runs), desc='simulation'):
            many_attemps = (
                Parallel(n_jobs=-1, verbose=0)
                (delayed(self.single_run)
                () for _ in range(self.n_attemps_per_run))
            )
            many_attemps = np.array(many_attemps)
            best = self.choose_best_run(many_attemps)[0]
            self.best_runs[i] = best
    
    def run_without_selection(self):
        """
        Run the simulation but do not discard any of the results.
        """
        many_attemps = (
            Parallel(n_jobs=-1, verbose=0)
            (delayed(self.single_run)
            () for _ in range(self.n_runs))
        )
        self.best_runs = np.array(many_attemps)

        
    def total_energy(self, pos):
        """
        Returns the total energy of a given configuration.
        """
        dist = cdist(pos, pos)
        return (1 / dist[dist != 0]).sum() / 2

    def choose_best_run(self, runs):
        """ 
        Returns the configuration with the lowest total energy.
        """
        energies = np.zeros(len(runs))
        for i, r in enumerate(runs):
            e = self.total_energy(r)
            energies[i] = e
        index_best = np.argmin(energies)
        best = runs[index_best]
        return best, index_best

    def single_run(self):
        """
        Runs the simulation just one time. 
        """
        rs = RunSimulation(
            self.n_charges,
            self.circle_radius,
            self.n_steps, 
            self.stepsize_init,
            self.stepsize_final,
            self.random_influence, 
            self.force_influence,
            self.temperature
        )
        final_pos =  rs.run()
        return final_pos
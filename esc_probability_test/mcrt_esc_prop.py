#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2018 Ulrich Noebauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Python module containing tools to perform simple MCRT simulations to determine
the escape probability from a homogeneous sphere. This test is presented in the
MCRT review.
"""
from __future__ import print_function
import numpy as np
import numpy.random as random
# from pynverse import inversefunc
import logging as log
log.basicConfig(level=log.WARNING)


def p_esc_analytic(t):
    """Calculate the escape probability analytically

    Note: it is assumed that there is no scattering within the sphere, but
    that photons/packets can only be absorbed.

    Parameter
    ---------
    t : float, np.ndarray
        total optical depth of the sphere

    Returns
    -------
    p : float, np.ndarray
        escape probability
    """
    return (3. / (4. * t) * (1. - 1. / (2. * t**2) +
                             (1. / t + 1. / (2. * t**2)) * np.exp(-2. * t)))


class homogeneous_sphere_esc_abs(object):
    """Homogeneous Sphere class

    This class defines a homogeneous sphere with a specified total optical
    depth and performs a simple MCRT simulation to determine the escape
    probability. This is done automatically during the initialization step. The
    escape probability can be accessed via the class attribute p_esc. The
    effect of isotropic scattering can be included by setting albedo > 0. This
    parameter describes the scattering probability with respect to the total
    (i.e. scattering + absorption) interaction probability.

    Note: the analytic prediction p_esc_analytic only applies for albedo = 0,
    i.e. in the absence of scattering

    Parameters
    ----------
    tau : float
        total optical depth of the homogeneous sphere
    albedo : float
        ratio of scattering to total interaction probability (default 0.1)
    N : int
        number of MC packets that are setup up (default 10000)

    Attributes
    ----------
    p_esc : float
        escape probability as determined by MCRT

    """

    def __init__(self, tau, albedo=0.1, N=10000):

        self.RNG = random.RandomState(seed=None)
        self.N = N
        self.tau_sphere = tau
        self.albedo = albedo

        # initial position of packets in optical depth space
        # question: Why are they using tau instead of r?
        self.tau_i = self.tau_sphere * (self.RNG.rand(self.N))**(1./3.)
        # initial propagation direction
        self.mu_i = 2 * self.RNG.rand(self.N) - 1.

        # number of escaping packets
        self.N_esc = 0
        # number of active packets
        self.N_active = self.N

        # perform propagation
        self._propagated = False
        self._propagate()

    @property
    def p_esc(self):
        """escape probability"""
        return self.N_esc / float(self.N)

    def _propagate(self):
        """Perform propagation of MC packets

        All packets are followed until they are absorbed or escape from the
        sphere.
        """

        if self._propagated:

            print("Propagation has already been performed!")
            print("_propagate call will have no effect")
            return False

        i = 0
        while self.N_active > 0:
            self._propagate_step()
            i = i + 1
            if i > 1e6:
                print("Safety exit")
                print("Propagation steps limit of {:d} exceeded".format(i))
                return False
        print("Performed {:d} propagation steps".format(i))
        return True

    def _propagate_step(self):
        """Perform one propagation step

        All active packets are propagated to the next event which can either be
        a physical interaction or escaping from the sphere. If scatterings are
        active, it is decided for each interacting packet whether it is
        absorbed or scattered. All packets that are absorbed or escape during
        the current step are removed from the active pool.
        """

        # optical depth to next interaction
        self.tau = -np.log(self.RNG.rand(self.N_active))
        # optical depth to sphere edge
        # question: where does this formula come from?
        self.tau_edge = np.sqrt(
            self.tau_sphere**2 - self.tau_i**2*(1. - self.mu_i**2)) - self.tau_i * self.mu_i

        # identify packets that escape
        self.esc_mask = self.tau_edge < self.tau
        # update number of escaping packets
        self.N_esc += self.esc_mask.sum()

        # identify interacting packets
        self.nesc_mask = np.logical_not(self.esc_mask)

        # decide which interacting packets scatter and which get absorbed
        self.abs_mask = self.RNG.rand(self.nesc_mask.sum()) >= self.albedo
        self.scat_mask = np.logical_not(self.abs_mask)

        # select properties of scattering packets
        self.tau = self.tau[self.nesc_mask][self.scat_mask]
        self.tau_i = self.tau_i[self.nesc_mask][self.scat_mask]
        self.mu_i = self.mu_i[self.nesc_mask][self.scat_mask]

        # update number of active packets
        self.N_active = self.scat_mask.sum()

        # update properties (position in optical depth space, propagation
        # direction) of scattering packets
        # question: Where does this formula come from?
        self.tau_i = np.sqrt(self.tau_i**2 + self.tau **
                             2 + 2. * self.tau * self.tau_i * self.mu_i)
        self.mu_i = 2 * self.RNG.rand(self.N_active) - 1.


class non_homogeneous_sphere_esc_abs(object):
    """Non-Homogeneous Sphere class

    Matter density: rho(r) = m0 / (r + epsilon)**2

    This class defines a sphere with a non homogenous matter distribution and a specified total optical
    depth and performs a simple MCRT simulation to determine the escape
    probability. This is done automatically during the initialization step. The
    escape probability can be accessed via the class attribute p_esc. The
    effect of isotropic scattering can be included by setting albedo > 0. This
    parameter describes the scattering probability with respect to the total
    (i.e. scattering + absorption) interaction probability.

    Note: the analytic prediction p_esc_analytic_non_homogenous only applies for albedo = 0,
    i.e. in the absence of scattering

    Note: Since we are defining tau_sphere instead of r_sphere the mass constant m_0 has no impact on the final result.
    It is only added for completeness.

    Parameters
    ----------
    tau_sphere : float
        total optical depth of the homogeneous sphere
    epsilon : float
        Smoothing parameter avoiding matter singularity at center of sphere.
    albedo : float
        ratio of scattering to total interaction probability (default 0.1)
    N : int
        number of MC packets that are setup up (default 10000)

    Attributes
    ----------
    p_esc : float
        escape probability as determined by MCRT

    """

    def __init__(self, tau_sphere, epsilon=1e-5, albedo=0.1, N=10000):

        self.RNG = random.RandomState(seed=None)
        self.N = N
        self.epsilon = epsilon
        # The choice of m0 is arbitrary since we only define tau this only changes the radius of the sphere but not the physics in tau space.
        self.m0 = 1
        self.r_sphere = self._radius_at_optical_depth(tau_sphere)
        self.albedo = albedo

        # initial distance of photon packet from center of sphere.
        # Homogeneous distribution:
        self.r_i = self.r_sphere * (self.RNG.rand(self.N))**(1./3.)
        # Inhomogenous distribution:
        # inv_cdf = self._initialize_inverse_cdf()
        # self.r_i = inv_cdf(self.RNG.rand(self.N))

        # initial propagation direction
        self.mu_i = 2 * self.RNG.rand(self.N) - 1.

        # number of escaping packets
        self.N_esc = 0
        # number of active packets
        self.N_active = self.N

        # perform propagation
        self._propagated = False
        self._propagate()

    def _optical_depth_at_radius(self, r):
        eps = self.epsilon
        m0 = self.m0
        return m0*r / (eps*(r + eps))

    def _radius_at_optical_depth(self, tau):
        eps = self.epsilon
        m0 = self.m0
        tau = tau / m0
        return -tau * eps**2 / (tau*eps - 1)

    def _initialize_inverse_cdf(self):
        """Returns the inverse of the commulative distribution function.

        This function inverts the cdf of the photon packet distribution
        numerically to sample r from a random number.
        f(r) = 1 / (r+eps)**2
        F(r) = F(r)-F(0) = r*(r + 2*eps)/(r + eps) + 2*eps*log(eps/(r + eps))
        r = F^-1(rand)
        """
        eps = self.epsilon

        def cdf(r):
            return r*(r + 2*eps)/(r + eps) + 2*eps*np.log(eps/(r + eps))
        # Normalization constant
        c = cdf(self.r_sphere)

        breakpoint()
        inv_cdf = inversefunc(
            lambda r: cdf(r)/c,
            domain=(0, self.r_sphere),
            open_domain=(False, False)
        )
        return inv_cdf

    @property
    def p_esc(self):
        """escape probability"""
        return self.N_esc / float(self.N)

    def _propagate(self):
        """Perform propagation of MC packets

        All packets are followed until they are absorbed or escape from the
        sphere.
        """

        if self._propagated:

            print("Propagation has already been performed!")
            print("_propagate call will have no effect")
            return False

        i = 0
        while self.N_active > 0:
            self._propagate_step()
            i = i + 1
            if i > 1e6:
                print("Safety exit")
                print("Propagation steps limit of {:d} exceeded".format(i))
                return False
        print("Performed {:d} propagation steps".format(i))
        return True

    def _propagate_step(self):
        """Perform one propagation step

        All active packets are propagated to the next event which can either be
        a physical interaction or escaping from the sphere. If scatterings are
        active, it is decided for each interacting packet whether it is
        absorbed or scattered. All packets that are absorbed or escape during
        the current step are removed from the active pool.
        """

        # optical depth to next interaction
        self.tau = -np.log(self.RNG.rand(self.N_active))

        # distance in real space corresponding to optical depth
        va = self.r_i
        eps = self.epsilon
        tau = self.tau
        mu = self.mu_i
        m0 = self.m0
        vb = -(m0*va + va*eps*tau-eps**2*tau) / (-m0 + va*tau + eps*tau)
        self.dl = -va*mu + np.sqrt(-va**2 + vb**2 + va**2*mu**2)

        # Distance to sphere edge
        self.r_edge = np.sqrt(
            self.r_sphere**2 - self.r_i**2*(1. - self.mu_i**2)) - self.r_i * self.mu_i

        # identify packets that escape
        self.esc_mask = self.r_edge < self.dl
        # update number of escaping packets
        self.N_esc += self.esc_mask.sum()

        # identify interacting packets
        self.nesc_mask = np.logical_not(self.esc_mask)

        # decide which interacting packets scatter and which get absorbed
        self.abs_mask = self.RNG.rand(self.nesc_mask.sum()) >= self.albedo
        self.scat_mask = np.logical_not(self.abs_mask)

        # select properties of scattering packets
        self.dl = self.dl[self.nesc_mask][self.scat_mask]
        self.r_i = self.r_i[self.nesc_mask][self.scat_mask]
        self.mu_i = self.mu_i[self.nesc_mask][self.scat_mask]

        # update number of active packets
        self.N_active = self.scat_mask.sum()

        # update properties (position in optical depth space, propagation
        # direction) of scattering packets
        # question: Where does this formula come from?
        self.r_i = np.sqrt(self.r_i**2 + self.dl **
                           2 + 2. * self.dl * self.r_i * self.mu_i)
        self.mu_i = 2 * self.RNG.rand(self.N_active) - 1.
        log.info(f'{self.p_esc = }')
        log.info(f'{self.N_esc = }')
        log.info(f'{self.N = }')


def main():

    mcrt_esc_prop = homogeneous_sphere_esc_abs(2)
    print("tau: {:.4e}, escape probability: {:.4e}".format(
        mcrt_esc_prop.tau_sphere, mcrt_esc_prop.p_esc))


def task1_esc_prob(n_packets, ax):

    tau_values_sim = np.logspace(-2, 2, 10)
    tau_values_analytic = np.logspace(-2, 2, 200)
    albedos = [0, 0.10, 0.50, 0.95]

    # Simulate escape probability for a homogeneous sphere.
    sims = {}
    for albedo in albedos:
        print(f"Simulating escape probability for albedo {albedo:.2f}")
        p_esc_sim = [
            homogeneous_sphere_esc_abs(
                tau_value, albedo=albedo, N=n_packets).p_esc
            for tau_value in tau_values_sim
        ]
        sims[albedo] = p_esc_sim

    # Analytical solution for albedo = 0.
    p_esc_a = p_esc_analytic(tau_values_analytic)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = iter(prop_cycle.by_key()['color'])
    ax.plot(tau_values_analytic, p_esc_a, '-',
            color=next(colors), label='Analytic')
    for albedo, p_esc_sim in sims.items():
        label = fr'$\chi_S / \chi_{{tot}} = {albedo}$'
        color = next(colors)
        ax.scatter(tau_values_sim, p_esc_sim, facecolor='none',
                   edgecolor=color, label=label)
        ax.semilogx(tau_values_sim, p_esc_sim, '--', color=color)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau_{sphere}$')
    ax.set_ylabel('escape probability')
    ax.legend()


def task2_specific_intensity(n_packets, ax):

    n_repetitions = 1
    albedo = 0
    tau_sphere_list = np.logspace(-2, 2, 100)
    albedos = [0, 0.10, 0.50, 0.95]

    sims = {}
    for albedo in albedos:
        print(f"Simulating escape probability for albedo {albedo:.2f}")
        multiple_sims = [
            [
                homogeneous_sphere_esc_abs(
                    tau_sphere, albedo=albedo, N=n_packets).p_esc
                for tau_sphere in tau_sphere_list
            ]
            for _ in range(n_repetitions)
        ]
        p_esc_sim = np.mean(multiple_sims, axis=0)
        sims[albedo] = p_esc_sim

    # Choose source function
    S = 1

    # Analytical solution
    I_ana = [S*(1-np.exp(-tau_sphere)) for tau_sphere in tau_sphere_list]
    ax.semilogx(tau_sphere_list, I_ana, '-', label='analytic')

    # Numeric solution
    for albedo, p_esc_sim in sims.items():
        packet_energy = tau_sphere_list * S / n_packets
        I_num = n_packets * p_esc_sim * packet_energy

        ax.semilogx(tau_sphere_list, I_num, '.--', label=fr'{albedo = }')
        ax.set_xlabel(r'$\tau_{sphere}$')
        ax.set_ylabel(r'$I_\nu(\tau_{sphere})$')
    ax.legend()


def task3_esc_prob_non_homogeneous_sphere(n_packets, ax):

    tau_values_sim = np.logspace(-2, 2, 10)
    tau_values_analytic = np.logspace(-2, 2, 200)
    albedos = [0, 0.10, 0.50, 0.95]

    # Simulate escape probability for a non-homogeneous sphere.
    sims = {}
    for albedo in albedos:
        print(f"Simulating escape probability for albedo {albedo:.2f}")
        p_esc_sim = [
            non_homogeneous_sphere_esc_abs(
                tau_value, albedo=albedo, N=n_packets).p_esc
            for tau_value in tau_values_sim
        ]
        sims[albedo] = p_esc_sim

    log.info(f'{sims = }')

    # Analytical solution for albedo = 0.
    p_esc_a = p_esc_analytic(tau_values_analytic)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = iter(prop_cycle.by_key()['color'])
    ax.plot(tau_values_analytic, p_esc_a, '-',
            color=next(colors), label=r'Analytic $\rho = 1$')
    for albedo, p_esc_sim in sims.items():
        label = fr'$\chi_S / \chi_{{tot}} = {albedo}$'
        color = next(colors)
        ax.scatter(tau_values_sim, p_esc_sim, facecolor='none',
                   edgecolor=color, label=label)
        ax.semilogx(tau_values_sim, p_esc_sim, '--', color=color)
    ax.set_xlabel(r'$\tau_{sphere}$')
    ax.set_ylabel('escape probability')
    ax.legend()


def task3_non_homogeneous_sphere_specific_intensity(n_packets, ax):

    n_repetitions = 1
    albedo = 0
    tau_sphere_list = np.logspace(-2, 2, 100)
    albedos = [0, 0.10, 0.50, 0.95]

    sims = {}
    for albedo in albedos:
        print(f"Simulating escape probability for albedo {albedo:.2f}")
        multiple_sims = [
            [
                non_homogeneous_sphere_esc_abs(
                    tau_sphere, albedo=albedo, N=n_packets).p_esc
                for tau_sphere in tau_sphere_list
            ]
            for _ in range(n_repetitions)
        ]
        p_esc_sim = np.mean(multiple_sims, axis=0)
        sims[albedo] = p_esc_sim

    # Choose source function
    S = 1

    # Analytical solution
    I_ana = [S*(1-np.exp(-tau_sphere)) for tau_sphere in tau_sphere_list]
    ax.semilogx(tau_sphere_list, I_ana, '-', label='analytic')

    # Numeric solution
    for albedo, p_esc_sim in sims.items():
        packet_energy = tau_sphere_list * S / n_packets
        I_num = n_packets * p_esc_sim * packet_energy

        ax.semilogx(tau_sphere_list, I_num, '.--', label=fr'{albedo = }')
        ax.set_xlabel(r'$\tau_{sphere}$')
        ax.set_ylabel(r'$I_\nu(\tau_{sphere})$')
    ax.legend()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_packets = int(1e5)

    with plt.style.context('seaborn-talk'):
        # Escape probabilities
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        task1_esc_prob(n_packets, ax1)
        task3_esc_prob_non_homogeneous_sphere(n_packets, ax2)
        fig1.suptitle(f"Homogeneous photon distr, {n_packets:.0e} packets.")
        ax1.title.set_text(r"$\rho = 1$")
        ax2.title.set_text(r"$\rho = 1/(r+\epsilon)^2$")

        # Specific intensities
        fig2, (ax3, ax4) = plt.subplots(1, 2)
        task2_specific_intensity(n_packets, ax3)
        task3_non_homogeneous_sphere_specific_intensity(n_packets, ax4)
        fig2.suptitle(f"Homogeneous photon distr, {n_packets:.0e} packets.")
        ax3.title.set_text(r"$\rho = 1$")
        ax4.title.set_text(r"$\rho = 1/(r+\epsilon)^2$")
        plt.show()

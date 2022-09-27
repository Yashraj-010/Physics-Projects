

# BASIC LATTICE-BOLTZMANN SIMULATION'


import numpy as np
import matplotlib.pyplot as plt


def main():

    # First Defining latiice parameters for simulation

    N_x = 400        # total grids in x-direction
    N_y = 100        # total grids in y-direction
    N_t = 8000      # total time frames
    tau = 0.63       # relaxation (or collision time)

    N_v = 9     # number of velocity points in a single lattice. Since we are in 2D it is 9. (D2Q9)

    # now we'll define value of each of 9 points in two 1D array.
    #
    #             6---2---5
    #             3---0---1
    #             7---4---8
    #

    C_x = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1])                       # x values of points
    C_y = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1])                       # y values of points
    W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])        # weight of each point

    # Initial condition of velocity distribution

    F_ini = np.ones((N_y, N_x, N_v)) + 0.05 * np.random.randn(N_y, N_x, N_v)

    # We want the initial motion of the fluid towards right direction

    F_ini[:, :, 1] += 2.3

    # Obstacle Object - Here we have used cylinder

    X, Y = np.meshgrid(range(N_x), range(N_y))
    Obstacle = ((X - N_x/4)**2 / 4.0) + ((Y - N_y/2)**2 / 2.25 ) < 100


    # Calculations

    for t in range(N_t):
        print(t)

        F_ini[:, -1, [6, 3, 7]] = F_ini[:, -2, [6, 3 , 7]]
        F_ini[:,  0, [8, 1, 5]] = F_ini[:,  1, [8, 1 , 5]]

        # Drift

        for i, cx, cy in zip(range(N_v), C_x, C_y):
            F_ini[:,:,i] = np.roll(F_ini[:,:,i], cx, axis=1)
            F_ini[:,:,i] = np.roll(F_ini[:,:,i], cy, axis=0)

        # We'll consider that fluid reflects as the boundary

        bound = F_ini[Obstacle, :]
        bound = bound[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]

        # Calculating Fluid variables

        rho = np.sum(F_ini, 2)
        u_x = np.sum(C_x * F_ini, 2) / rho
        u_y = np.sum(C_y * F_ini, 2) / rho

        # Calculating Collisions

        F_eq = np.zeros(F_ini.shape)
        for i, cx, cy, w in zip(range(N_v), C_x, C_y, W):
             F_eq[:,:,i] = rho * w * ( 1 + 3*(cx*u_x+cy*u_y)  + 9*(cx*u_x+cy*u_y)**2/2 - 3*(u_x**2+u_y**2)/2 )

        F_ini += -(1.0/tau) * (F_ini - F_eq)

        F_ini[Obstacle,:] = bound  # Applying boundary condition after every calculation
        u_x[Obstacle] = 0
        u_y[Obstacle] = 0

        # Now plotting the calculation

        if (t % 2 == 0):
            plt.imshow(np.sqrt(u_x**2 + u_y**2))
            plt.pause(0.01)
            plt.cla()



if __name__== "__main__":
  main()

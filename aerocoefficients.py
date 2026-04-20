import numpy as np
import os

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, 'data', 'naca2412_polar.csv')

data = np.genfromtxt(data_path,
                     delimiter=',',
                     skip_header=11,
                     usecols=(0, 1, 2, 4))

def get_aero_coefficients(alpha_deg):

    CL = np.interp(alpha_deg, data[:,0], data[:,1])
    CD = np.interp(alpha_deg, data[:,0], data[:,2])
    Cm = np.interp(alpha_deg, data[:,0], data[:,3])

    return CL, CD, Cm

if __name__ == "__main__":
    alpha = np.arange(-10, 18, 1)  # test angles of attack in degrees
    for a in alpha:
        CL, CD, Cm = get_aero_coefficients(a)
        print(f"At alpha = {a:.1f} degrees:")
        print(f"CL = {CL:.4f}, CD = {CD:.4f}, Cm = {Cm:.4f}")
        print()
if __name__ == '__main__':
    import numpy as np


    def angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))


    A = (1, 0)
    B = (1, -0.1)
    print(angle_between(B, A))

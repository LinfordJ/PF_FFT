import taichi as ti

@ti.data_oriented
class MultiphaseEnergy:
    def __init__(self, W=1.0, U=3.0):
        self.W = W
        self.U = U

    @ti.func
    def compute_force(self, phi_p, sum_abs, sum_sq_abs):
        abs_p = ti.abs(phi_p)
        sign_p = ti.math.sign(phi_p)
        sum_abs_others = sum_abs - abs_p
        sum_sq_others = sum_sq_abs - abs_p * abs_p
        chemical_force = -self.W * sign_p * sum_abs_others
        penalty_sum = 0.5 * (sum_abs_others * sum_abs_others - sum_sq_others)
        penalty_force = -self.U * sign_p * penalty_sum
        return chemical_force + penalty_force

import taichi as ti

@ti.data_oriented
class InterpolationFunction:
    """
    Base class for interpolation functions in multiphase phase field modeling.
    Users can subclass this to define custom h(phi) and its derivative.
    """
    
    @ti.func
    def h(self, phi):
        """
        Interpolation function. Evaluates the fractional contribution.
        Ideally satisfies h(0) = 0 and h(1) = 1.
        """
        return 0.0
        
    @ti.func
    def dh(self, phi):
        """
        Derivative of the interpolation function with respect to phi.
        """
        return 0.0


@ti.data_oriented
class PolynomialInterpolation(InterpolationFunction):
    """
    Standard polynomial interpolation function: 
    h(phi) = phi^3 * (10 - 15 * phi + 6 * phi^2)
    
    Ensures smooth transition with zero first and second derivatives 
    at phi=0 and phi=1. Commonly used in N-phase models.
    """
    
    @ti.func
    def h(self, phi):
        return (phi**3) * (10.0 - 15.0 * phi + 6.0 * phi**2)
        
    @ti.func
    def dh(self, phi):
        return 30.0 * (phi**2) * ((1.0 - phi)**2)

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class SimulationConfig:
    dim: int = 2
    N: Tuple[int, ...] = (128, 128)
    L: Tuple[float, ...] = (128.0, 128.0)
    
    N_phases: int = 3
    dt: float = 0.01
    max_steps: int = 10000
    
    kappa: float = 1.0
    W: float = 1.0
    U: float = 3.0
    mobility: float = 1.0
    bulk_energies: List[float] = None  
    
    semi_implicit: bool = True
    equation_type: str = "Allen-Cahn"
    
    def __post_init__(self):
        assert self.dim in [1, 2, 3], "Dimension must be 1, 2, or 3"
        assert len(self.N) == self.dim, "Grid resolution must match dimension"
        assert len(self.L) == self.dim, "Domain size must match dimension"
        assert self.N_phases >= 2, "Must have at least 2 phases"
        assert self.equation_type in ["Allen-Cahn", "Cahn-Hilliard"], "equation_type must be Allen-Cahn or Cahn-Hilliard"
        
        if self.bulk_energies is None:
            self.bulk_energies = [0.0] * self.N_phases
        else:
            assert len(self.bulk_energies) == self.N_phases, "Length of bulk_energies must equal N_phases"

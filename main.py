import numpy as np
import random
from evolution import run_evolution

def main():
    np.random.seed(42)
    random.seed(42)
    
    run_evolution()

if __name__ == "__main__":
    main()
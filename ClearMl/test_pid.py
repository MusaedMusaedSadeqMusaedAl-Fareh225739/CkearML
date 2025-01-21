# test_pid.py
from sim_class import Simulation

if __name__ == "__main__":
    # 1) Create the simulation
    sim = Simulation(num_agents=1, render=True, rgb_array=False)
    
    # 2) Command the pipette to move to (X=150, Y=80, Z=15) mm
    #    with up to 1000 steps to get within ±0.1 mm (smaller tolerance)
    sim.run_pid([150.0, 80.0, 15.0], max_steps=1000, tolerance=0.1)
    
    # 3) Once done, close the simulation
    sim.close()

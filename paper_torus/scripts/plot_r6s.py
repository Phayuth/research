import os
import matplotlib.pyplot as plt
from simulator.sim_planar_6r import RobotArm6RSimulator
import numpy as np
from rnd_task_map import PaperTorusIFAC2025


def fig_r6s_snggoal():
    torus = True
    env = RobotArm6RSimulator(PaperTorusIFAC2025(), torusspace=torus)
    rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
    thetas = np.loadtxt(rsrc + "paper_r6s_snggoal_path.csv", delimiter=",")
    print(thetas.shape)
    env.plot_view(thetas)


def fig_r6s_cartgoal():
    pass


def fig_r6s_altgoal():
    pass


if __name__ == "__main__":
    while True:
        func = [
            fig_r6s_snggoal,
            fig_r6s_cartgoal,
            fig_r6s_altgoal,
        ]

        for i, f in enumerate(func, start=1):
            print(f"{i}: {f.__name__}")

        arg = input("Enter argument number (` to exit): ")

        if arg == "`":
            print("Exiting...")
            break
        elif arg.isdigit() and 1 <= int(arg) <= len(func):
            func[int(arg) - 1]()

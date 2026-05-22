"""
logger.py
Multi-Species Conflict Data Logger
"""
import csv
import os

class ConflictLogger:
    def __init__(self, filename="conflict_log.csv"):
        self.filename = filename
        self.headers = [
            "Simulation_Tick", "Species", "X_Coord", "Y_Coord", 
            "Hunger_Level", "Stress_Level", "Dist_to_Water", 
            "Dist_to_Crops", "Human_Density", "Q_Value_North",
            "Q_Value_South", "Q_Value_East", "Q_Value_West", "Chosen_Action"
        ]
        self._initialize_file()

    def _initialize_file(self):
        if os.path.exists(self.filename):
            os.remove(self.filename) # Start fresh every run
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)

    def log_conflict(self, tick, species, location, state_vector, q_values, chosen_action):
        row = [tick, species, location[0], location[1]] + state_vector + q_values + [chosen_action]
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
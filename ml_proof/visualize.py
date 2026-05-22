"""
visualize.py
Generates presentation-ready dashboard
"""
import pandas as pd
import matplotlib.pyplot as plt

def generate_dashboard():
    try:
        df = pd.read_csv("conflict_log.csv")
    except Exception as e:
        print("Could not load CSV. Run main.py first.")
        return

    if len(df) == 0:
        print("No conflicts logged yet! Let the model train longer.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {'Elephant': 'blue', 'Leopard': 'red', 'Sloth_Bear': 'green'}

    # Plot 1: Hotspots by Species
    for species in df['Species'].unique():
        subset = df[df['Species'] == species]
        ax1.scatter(subset['X_Coord'], subset['Y_Coord'], label=species, color=colors.get(species, 'gray'), alpha=0.6)
        
    ax1.set_title("Conflict Spatial Distribution")
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Decision Dilemma
    df['Max_Q'] = df[['Q_Value_North', 'Q_Value_South', 'Q_Value_East', 'Q_Value_West']].max(axis=1)
    
    for species in df['Species'].unique():
        subset = df[df['Species'] == species]
        ax2.scatter(subset['Hunger_Level'], subset['Max_Q'], label=species, color=colors.get(species, 'gray'), alpha=0.5)

    ax2.set_title("AI Decision Making: Hunger vs Action Value")
    ax2.set_xlabel("Agent Hunger Level")
    ax2.set_ylabel("Neural Network Q-Value")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Reinforcement Learning: Multi-Species Conflict Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig("multi_species_dashboard.png", dpi=300)
    print("Dashboard saved as multi_species_dashboard.png!")
    plt.show()

if __name__ == "__main__":
    generate_dashboard()
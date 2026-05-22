import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a professional academic style for the charts
sns.set_theme(style="darkgrid", context="paper")

def generate_charts():
    filename = "ecosystem_data.csv"
    
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found. Run the simulation for a bit first!")
        return

    print("📊 Loading ecosystem telemetry...")
    df = pd.read_csv(filename)
    
    # Create an output directory for the presentation images
    os.makedirs("presentation_charts", exist_ok=True)

    # ---------------------------------------------------------
    # CHART 1: Population Dynamics Over Time
    # Shows how populations grow or crash based on resources
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    # Count unique active agents per species per tick
    pop_df = df.groupby(['Tick', 'Species'])['Agent_ID'].nunique().unstack().fillna(0)
    
    colors = {'elephant': '#6b4c11', 'human': '#1e3a5f', 'leopard': '#8b2020', 'sloth_bear': '#3d5a3e'}
    for col in pop_df.columns:
        plt.plot(pop_df.index, pop_df[col], label=col.capitalize(), color=colors.get(col, 'black'), linewidth=2)
    
    plt.title('Ecosystem Population Dynamics', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Time (Ticks)', fontsize=12)
    plt.ylabel('Number of Active Agents', fontsize=12)
    plt.legend(title='Species')
    plt.tight_layout()
    plt.savefig('presentation_charts/1_population_dynamics.png', dpi=300)
    print("✅ Generated: Population Dynamics Chart")

    # ---------------------------------------------------------
    # CHART 2: Elephant Physiology (Weight vs. Energy)
    # Shows the physical toll of migration and resource scarcity
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    elephant_df = df[df['Species'] == 'elephant'].groupby('Tick').agg({'Weight_kg': 'mean', 'Energy': 'mean'})
    
    if not elephant_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot Weight on the left Y-axis
        color = 'tab:brown'
        ax1.set_xlabel('Simulation Time (Ticks)', fontsize=12)
        ax1.set_ylabel('Average Weight (kg)', color=color, fontsize=12)
        ax1.plot(elephant_df.index, elephant_df['Weight_kg'], color=color, linewidth=2, label='Weight (kg)')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot Energy on the right Y-axis
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Average Energy Level (%)', color=color, fontsize=12)
        ax2.plot(elephant_df.index, elephant_df['Energy'], color=color, linewidth=2, linestyle='--', label='Energy (%)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Elephant Herd Physiology: Weight vs Energy', fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.savefig('presentation_charts/2_elephant_physiology.png', dpi=300)
        print("✅ Generated: Elephant Physiology Chart")

    # ---------------------------------------------------------
    # CHART 3: Ecosystem Psychological & Behavioral States
    # Shows how often animals were stressed, starving, or normal
    # ---------------------------------------------------------
    plt.figure(figsize=(9, 5))
    # Filter out normal state to highlight the interesting behaviors
    stress_df = df[df['Behavioral_Condition'] != 'Normal']
    
    if not stress_df.empty:
        ax = sns.countplot(data=stress_df, y='Behavioral_Condition', hue='Species', palette=colors)
        plt.title('Ecosystem Stress Diagnostics', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Logged Occurrences', fontsize=12)
        plt.ylabel('Behavioral Condition', fontsize=12)
        plt.legend(title='Species')
        plt.tight_layout()
        plt.savefig('presentation_charts/3_behavioral_diagnostics.png', dpi=300)
        print("✅ Generated: Behavioral Diagnostics Chart")
    else:
        print("⚠️ No stress/starvation events logged yet. Let the simulation run longer!")

    print("\n🎉 All charts saved successfully in the 'presentation_charts' folder!")

if __name__ == "__main__":
    generate_charts()
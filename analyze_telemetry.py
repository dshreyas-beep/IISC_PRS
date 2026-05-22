import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Configuration & Styling
CSV_FILE = "ecosystem_data.csv"
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})

def run_analysis():
    if not os.path.exists(CSV_FILE):
        print(f"[Error] Could not find {CSV_FILE}. Run the simulation first!")
        return

    print("Loading ecosystem telemetry data...")
    df = pd.read_csv(CSV_FILE)

    if df.empty:
        print("[Warning] No data found in logs.")
        return

    print(f"Loaded {len(df)} total telemetry records. Applying megafauna filter...")

    # THE FIX: Filter out the background species (birds, small deer) to make graphs readable
    primary_wildlife = ['elephant', 'leopard', 'sloth_bear', 'tiger', 'lion']
    wildlife_df = df[df['Species'].isin(primary_wildlife)].copy()

    # =====================================================================
    # PLOT 1: What causes conflicts across different species?
    # =====================================================================
    plt.figure(figsize=(12, 6))
    
    # We use the main 'df' here because we MUST include 'human' for conflict logs
    conflict_df = df[df['Action_Mode'].isin(['DEFEND', 'HUNT'])]
    
    # Only plot species that actually had conflicts to keep the x-axis clean
    active_conflict_species = conflict_df['Species'].unique()
    clean_conflict_df = conflict_df[conflict_df['Species'].isin(active_conflict_species)]

    if not clean_conflict_df.empty:
        ax = sns.countplot(
            data=clean_conflict_df, 
            x='Species', 
            hue='Behavioral_Condition',
            palette='rocket'
        )
        
        plt.title('The Root Cause of Human-Wildlife Conflict by Species', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Species Involved in Conflict', fontsize=12)
        plt.ylabel('Total Number of Conflict Events', fontsize=12)
        plt.legend(title='Animal Condition During Conflict', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('ecosystem_conflict_causes.png', dpi=300)
        print("Saved: ecosystem_conflict_causes.png")

    # =====================================================================
    # PLOT 2: Migration Thresholds (Boxplots)
    # =====================================================================
    plt.figure(figsize=(14, 6))
    
    # Use our filtered wildlife_df so the x-axis doesn't turn into a black smudge
    movement_df = wildlife_df[wildlife_df['Action_Mode'].isin(['WANDER', 'MIGRATE'])]
    
    # Subplot 1: Energy
    plt.subplot(1, 2, 1)
    sns.boxplot(data=movement_df, x='Species', y='Energy', hue='Action_Mode', palette=['#95a5a6', '#e74c3c'])
    plt.title('Energy Levels: Wandering vs Migrating', fontsize=14, pad=10)
    plt.ylabel('Energy Level (0-100)', fontsize=12)
    plt.axhline(30, color='r', linestyle='--', alpha=0.3, label='Starvation Risk')
    plt.legend(loc='lower right')

    # Subplot 2: Hydration
    plt.subplot(1, 2, 2)
    sns.boxplot(data=movement_df, x='Species', y='Hydration', hue='Action_Mode', palette=['#95a5a6', '#3498db'])
    plt.title('Hydration Levels: Wandering vs Migrating', fontsize=14, pad=10)
    plt.ylabel('Hydration Level (0-100)', fontsize=12)
    plt.axhline(30, color='b', linestyle='--', alpha=0.3, label='Dehydration Risk')
    plt.legend(loc='lower right')

    plt.suptitle('Survival Mechanics: When Do Animals Decide to Migrate?', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ecosystem_migration_thresholds.png', dpi=300)
    print("Saved: ecosystem_migration_thresholds.png")

    # =====================================================================
    # PLOT 3: Ecosystem Stress Timeline
    # =====================================================================
    plt.figure(figsize=(14, 7))
    
    # Use the filtered wildlife_df to prevent the "spaghetti" line explosion
    timeline = wildlife_df.groupby(['Tick', 'Species'])['Hydration'].mean().reset_index()
    
    sns.lineplot(data=timeline, x='Tick', y='Hydration', hue='Species', linewidth=2.5)
    
    plt.title('Ecosystem Hydration Timeline: Competition for Water Sources', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Simulation Time (Ticks)', fontsize=12)
    plt.ylabel('Average Species Hydration (0-100)', fontsize=12)
    
    # Danger zone
    plt.fill_between(timeline['Tick'], 0, 30, color='red', alpha=0.05, label='Critical Dehydration Zone')
    
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Primary Species")
    plt.tight_layout()
    plt.savefig('ecosystem_water_competition.png', dpi=300)
    print("Saved: ecosystem_water_competition.png")

if __name__ == "__main__":
    run_analysis()
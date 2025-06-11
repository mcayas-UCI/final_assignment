import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import warnings

# Percentiles to analyze for reaction time distributions
PERCENTILES = [10, 30, 50, 70, 90]

# Map categorical variables to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Human-readable names for the four experimental conditions
CONDITION_NAMES = {
    0: 'Easy Simple',    # easy difficulty + simple stimulus
    1: 'Easy Complex',   # easy difficulty + complex stimulus
    2: 'Hard Simple',    # hard difficulty + simple stimulus
    3: 'Hard Complex'    # hard difficulty + complex stimulus
}

# Create output directory for saving results
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def read_data(file_path, prepare_for='sdt', display=False):
    """
    Load and preprocess experimental data for different types of analysis.
    
    Args:
        file_path: Path to CSV file containing raw experimental data
        prepare_for: Type of analysis ('sdt' for Signal Detection Theory, 'delta plots' for RT analysis)
        display: Whether to print data info (unused in current implementation)
    
    Returns:
        Processed DataFrame ready for the specified analysis type
    """
    # Load raw data and apply categorical mappings
    data = pd.read_csv(file_path)
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition identifiers
    data['pnum'] = data['participant_id']
    # Combine stimulus_type and difficulty into single condition variable (0-3)
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)

    if prepare_for == 'sdt':
        # ===== SIGNAL DETECTION THEORY DATA PREPARATION =====
        # Group trials by participant, condition, and signal presence
        # Count total trials and correct responses for each combination
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({'accuracy': ['count', 'sum']}).reset_index()
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        # Convert to SDT format: hits, misses, false alarms, correct rejections
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Separate signal-present and signal-absent trials
                signal_trials = c_data[c_data['signal'] == 0]  # signal present
                noise_trials = c_data[c_data['signal'] == 1]   # signal absent
                
                # Only include if participant has data for both signal and noise trials
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        # Signal present trials
                        'hits': signal_trials['correct'].iloc[0],  # correct "yes" responses
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],  # incorrect "no" responses
                        # Signal absent trials  
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],  # incorrect "yes" responses
                        'correct_rejections': noise_trials['correct'].iloc[0],  # correct "no" responses
                        # Trial counts
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        return pd.DataFrame(sdt_data)

    elif prepare_for == 'delta plots':
        # ===== DELTA PLOT DATA PREPARATION =====
        # Calculate RT percentiles for different trial types
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', *[f'p{p}' for p in PERCENTILES]])
        
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                if c_data.empty:
                    continue
                
                # Calculate percentiles for three different trial types
                for mode, subset in [('overall', c_data),                           # all trials
                                     ('accurate', c_data[c_data['accuracy'] == 1]), # correct trials only
                                     ('error', c_data[c_data['accuracy'] == 0])]:   # error trials only
                    if not subset.empty:
                        # Calculate RT percentiles for this subset
                        quantiles = {f'p{p}': [np.percentile(subset['rt'], p)] for p in PERCENTILES}
                        dp_data = pd.concat([dp_data, pd.DataFrame({
                            'pnum': [pnum],
                            'condition': [condition],
                            'mode': [mode],
                            **quantiles
                        })])
        return dp_data.reset_index(drop=True)

def apply_hierarchical_sdt_model(data):
    """
    Fit a hierarchical Bayesian Signal Detection Theory model.
    
    This model estimates d-prime (sensitivity) and criterion (bias) parameters
    for each participant and condition, while accounting for individual differences
    through hierarchical priors.
    
    Args:
        data: DataFrame with SDT-formatted data (hits, false alarms, etc.)
    
    Returns:
        PyMC model object ready for sampling
    """
    P = len(data['pnum'].unique())      # number of participants
    C = len(data['condition'].unique()) # number of conditions
    
    with pm.Model() as sdt_model:
        # ===== GROUP-LEVEL (HYPERPRIOR) PARAMETERS =====
        # Mean d-prime for each condition across all participants
        mean_d_prime = pm.Normal('mean_d_prime', mu=0.0, sigma=1.0, shape=C)
        # Standard deviation of d-prime across participants
        stdev_d_prime = pm.HalfNormal('stdev_d_prime', sigma=1.0)
        
        # Mean criterion for each condition across all participants
        mean_criterion = pm.Normal('mean_criterion', mu=0.0, sigma=1.0, shape=C)
        # Standard deviation of criterion across participants
        stdev_criterion = pm.HalfNormal('stdev_criterion', sigma=1.0)
        
        # ===== INDIVIDUAL-LEVEL PARAMETERS =====
        # d-prime for each participant in each condition
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=stdev_d_prime, shape=(P, C))
        # Criterion for each participant in each condition
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=stdev_criterion, shape=(P, C))
        
        # ===== LINK FUNCTIONS =====
        # Convert d-prime and criterion to hit rate and false alarm rate
        hit_rate = pm.math.invlogit(d_prime - criterion)        # P(yes | signal present)
        false_alarm_rate = pm.math.invlogit(-criterion)         # P(yes | signal absent)
        
        # ===== LIKELIHOOD =====
        # Model observed hits as binomial with hit rate
        pm.Binomial('hit_obs', 
                    n=data['nSignal'], 
                    p=hit_rate[data['pnum'] - 1, data['condition']], 
                    observed=data['hits'])
        
        # Model observed false alarms as binomial with false alarm rate
        pm.Binomial('false_alarm_obs', 
                    n=data['nNoise'], 
                    p=false_alarm_rate[data['pnum'] - 1, data['condition']], 
                    observed=data['false_alarms'])
    
    return sdt_model

def draw_delta_plots(data, pnum):
    """
    Create delta plots for a single participant showing RT differences between conditions.
    
    Delta plots show how RT differences between conditions vary across the RT distribution.
    This can reveal different patterns for fast vs slow responses.
    
    Args:
        data: DataFrame with delta plot data (RT percentiles by condition and mode)
        pnum: Participant number to plot
    """
    data = data[data['pnum'] == pnum]
    conditions = sorted(data['condition'].unique())
    n = len(conditions)
    
    # Create grid of subplots for all condition comparisons
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    marker_style = {'marker': 'o', 'markersize': 8, 'markerfacecolor': 'white', 
                   'markeredgewidth': 2, 'linewidth': 2}
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Only plot upper triangle (avoid redundant comparisons)
            if i > j:
                continue
                
            ax = axes[i, j]
            ax.set_ylim(-0.3, 0.5)
            ax.axhline(0, color='gray', linestyle='--')  # Reference line at zero difference
            
            # Skip diagonal (condition vs itself)
            if i == j:
                ax.axis('off')
                continue
            
            def get_deltas(mode):
                """Calculate RT differences between two conditions for a given trial type"""
                q1 = np.array([data[(data['condition'] == cond1) & (data['mode'] == mode)][f'p{p}'].values[0] for p in PERCENTILES])
                q2 = np.array([data[(data['condition'] == cond2) & (data['mode'] == mode)][f'p{p}'].values[0] for p in PERCENTILES])
                return q2 - q1
            
            # Plot overall RT differences (upper triangle)
            ax.plot(PERCENTILES, get_deltas('overall'), color='black', **marker_style)
            
            # Plot accuracy-split RT differences (lower triangle)
            axes[j, i].plot(PERCENTILES, get_deltas('error'), color='red', **marker_style)
            axes[j, i].plot(PERCENTILES, get_deltas('accurate'), color='green', **marker_style)
            axes[j, i].legend(['Error', 'Accurate'], loc='upper left')
            
            # Add titles showing which conditions are being compared
            axes[i, j].set_title(f'{CONDITION_NAMES[cond2]} - {CONDITION_NAMES[cond1]}')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'delta_plots_p{pnum}.png')
    plt.close()

def plot_delta_contrast(delta_data, cond_a, cond_b, label, mode='overall'):
    """
    Plot average RT differences between two conditions across all participants.
    
    This creates summary plots showing how condition effects vary across the RT distribution,
    averaged across participants with error bars.
    
    Args:
        delta_data: DataFrame with delta plot data
        cond_a, cond_b: Condition numbers to compare
        label: Label for the plot legend
        mode: Trial type ('overall', 'accurate', or 'error')
    """
    percentiles = [f'p{p}' for p in PERCENTILES]
    diffs = []
    
    # Calculate RT differences for each participant
    for p in delta_data['pnum'].unique():
        d1 = delta_data[(delta_data['pnum'] == p) & (delta_data['condition'] == cond_a) & (delta_data['mode'] == mode)]
        d2 = delta_data[(delta_data['pnum'] == p) & (delta_data['condition'] == cond_b) & (delta_data['mode'] == mode)]
        
        if not d1.empty and not d2.empty:
            q1 = d1[percentiles].values[0]
            q2 = d2[percentiles].values[0]
            diffs.append(q2 - q1)  # Condition B - Condition A
    
    # Calculate mean and standard error across participants
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs, axis=0)
    sem_diff = np.std(diffs, axis=0) / np.sqrt(len(diffs))

    # Plot with error bars
    plt.errorbar(PERCENTILES, mean_diff, yerr=sem_diff, label=label, marker='o', capsize=5)


def main():
    """
    Execute the complete analysis pipeline:
    1. Load and prepare data for SDT analysis
    2. Fit hierarchical Bayesian SDT model
    3. Generate summary statistics and plots
    4. Perform delta plot analysis
    5. Save all results and visualizations
    """
    
    # ===== SIGNAL DETECTION THEORY ANALYSIS =====
    print("Loading and preparing SDT data...")
    sdt_data = read_data("data.csv", prepare_for='sdt', display=True)
    if sdt_data.empty:
        raise ValueError("SDT data is empty. Check data format.")

    print("Fitting hierarchical SDT model...")
    sdt_model = apply_hierarchical_sdt_model(sdt_data)
    
    # Sample from the posterior distribution
    with sdt_model:
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # ===== SDT MODEL DIAGNOSTICS AND RESULTS =====
    print("Model convergence summary:")
    summary = az.summary(trace, var_names=["mean_d_prime", "mean_criterion"], round_to=2)
    print(summary)
    summary.to_csv(OUTPUT_DIR / "sdt_summary.csv")

    # Create forest plot of posterior distributions
    az.plot_forest(trace, var_names=["mean_d_prime", "mean_criterion"], combined=True)
    plt.title("Posterior Distributions for d′ and Criterion")
    plt.savefig(OUTPUT_DIR / "sdt_posteriors.png")
    plt.close()

    # ===== SDT CONDITION COMPARISONS =====
    # Extract posterior means for each condition
    mean_d = trace.posterior['mean_d_prime'].mean(dim=("chain", "draw")).values
    mean_c = trace.posterior['mean_criterion'].mean(dim=("chain", "draw")).values

    print("\n--- SDT Condition Means ---")
    for i in range(4):
        print(f"{CONDITION_NAMES[i]}: d′ = {mean_d[i]:.2f}, criterion = {mean_c[i]:.2f}")

    # Calculate and display main effects and interactions
    print("\n--- SDT Contrasts ---")
    print(f"Stimulus Type effect (Easy): Complex - Simple: d′ = {mean_d[1] - mean_d[0]:.2f}")
    print(f"Stimulus Type effect (Hard): Complex - Simple: d′ = {mean_d[3] - mean_d[2]:.2f}")
    print(f"Difficulty effect (Simple): Hard - Easy: d′ = {mean_d[2] - mean_d[0]:.2f}")
    print(f"Difficulty effect (Complex): Hard - Easy: d′ = {mean_d[3] - mean_d[1]:.2f}")

    # ===== DELTA PLOT ANALYSIS =====
    print("Generating delta plots...")
    delta_data = read_data("data.csv", prepare_for='delta plots', display=False)
    
    # Create individual delta plots for each participant
    for p in delta_data['pnum'].unique():
        draw_delta_plots(delta_data, p)

    # ===== SPECIFIC RT DIFFERENCE ANALYSIS =====
    print("Computing RT differences between Hard Complex and Easy Simple...")
    # Compare the most extreme conditions (hardest vs easiest)
    diff_list = []
    for p in delta_data['pnum'].unique():
        pdata = delta_data[delta_data['pnum'] == p]
        for mode in ['overall', 'accurate', 'error']:
            cond0 = pdata[(pdata['condition'] == 0) & (pdata['mode'] == mode)]  # Easy Simple
            cond3 = pdata[(pdata['condition'] == 3) & (pdata['mode'] == mode)]  # Hard Complex
            
            if not cond0.empty and not cond3.empty:
                rt0 = cond0[[f'p{q}' for q in PERCENTILES]].values[0].astype(float)
                rt3 = cond3[[f'p{q}' for q in PERCENTILES]].values[0].astype(float)
                diff = rt3 - rt0  # Hard Complex - Easy Simple
                
                diff_list.append({
                    'pnum': p,
                    'mode': mode,
                    **{f'diff_p{q}': diff[i] for i, q in enumerate(PERCENTILES)}
                })

    # Save RT differences to CSV
    diff_df = pd.DataFrame(diff_list)
    diff_df.to_csv(OUTPUT_DIR / 'rt_differences_HardComplex_vs_EasySimple.csv', index=False)
    print("RT differences saved to output/rt_differences_HardComplex_vs_EasySimple.csv")

    # ===== SUMMARY DELTA PLOT VISUALIZATION =====
    print("Plotting delta contrast summary...")
    plt.figure(figsize=(8, 6))
    
    # Plot main effects: how RT differences vary across the distribution
    plot_delta_contrast(delta_data, 0, 1, 'Stimulus Type (Easy)')    # Easy: Complex - Simple
    plot_delta_contrast(delta_data, 2, 3, 'Stimulus Type (Hard)')    # Hard: Complex - Simple
    plot_delta_contrast(delta_data, 0, 2, 'Difficulty (Simple)')     # Simple: Hard - Easy
    plot_delta_contrast(delta_data, 1, 3, 'Difficulty (Complex)')    # Complex: Hard - Easy
    
    # Add reference line and formatting
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Delta Plot: RT Difference (by Percentile)")
    plt.xlabel("Percentile")
    plt.ylabel("RT Difference (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delta_plot_contrasts.png")
    plt.close()

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

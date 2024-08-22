"""
Health Economic Analysis: Cost-Effectiveness Analysis (CEA) with Probabilistic Sensitivity Analysis (PSA)

Script compares two different interventions, calculating the total cost, health outcomes (in QALYs), and the incremental cost-effectiveness ratio (ICER).

1. Calculate the total cost of each intervention, including costs associated with treatment, outpatient visits, and tests.
2. Compute the cost-effectiveness in terms of cost per QALY for each intervention.
3. Discount future costs and QALYs to their present values using a standard discount rate, reflecting the time value of money and societal preferences for current consumption.
4. Compute the Incremental Cost-Effectiveness Ratio (ICER) to determine the cost per additional QALY gained when one intervention is used instead of the other.
5. Perform a PSA by running simulations with random sampling from specified distributions for costs and QALYs. This accounts for uncertainty in model parameters, providing a more robust analysis of cost-effectiveness.

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

######################################################################################################################################################
# Define key financial and health-related variables for interventions
######################################################################################################################################################
cost_intervention_A = 10000  # Base cost in GBP for more expensive intervention
cost_intervention_B = 7000  # Base cost in GBP for standard intervention

qaly_intervention_A = 6 # Quality-Adjusted Life Years for A
qaly_intervention_B = 4  # Quality-Adjusted Life Years for B

# Hypothetical costs in GBP for additional health care utilisation
cost_per_outpatient_visit = 0
cost_per_test = 0

# Quantity of additional health care utilisation for each intervention
number_of_visits_A = 0
number_of_tests_A = 0
number_of_visits_B = 0
number_of_tests_B = 0

# Parameters for discounting future costs and QALYs to their present values
discount_rate = 0.0  # Discount rate according to NICE guidelines is 3.5%
years = 0             # Time horizon for the analysis

# Willingness to pay threshold
wtp = 20000  # £20,000 per QALY

# Distributions for PSA, mean, standard deviation set as 20% of mean
n_simulations=1000

# SDs for costs and QALYs based on typical variability
cost_sd = 0.05  # 5% of mean for costs
qaly_sd = 0.01  # 1% of mean for QALYs

cost_intervention_A_dist = np.random.normal(cost_intervention_A, cost_intervention_A * cost_sd, n_simulations)
cost_intervention_B_dist = np.random.normal(cost_intervention_B, cost_intervention_B * cost_sd, n_simulations)
qaly_intervention_A_dist = np.random.normal(qaly_intervention_A, qaly_intervention_A * qaly_sd, n_simulations)
qaly_intervention_B_dist = np.random.normal(qaly_intervention_B, qaly_intervention_B * qaly_sd, n_simulations)

# Plot distributions
# Create a 1x2 panel plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.rcParams['font.family'] = 'Calibri'
# Plot costs
sns.histplot(cost_intervention_A_dist, kde=True, color="#ADD8E6", label="Intervention A", ax=axes[0],alpha=0.5)
sns.histplot(cost_intervention_B_dist, kde=True, color="#90EE90", label="Intervention B", ax=axes[0],alpha=0.5)
axes[0].set_title('Cost Distribution', fontsize=18, fontweight='bold')
axes[0].set_xlabel('Cost (£)', fontsize=18, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=18, fontweight='bold')
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.2)
# Plot QALYs
sns.histplot(qaly_intervention_A_dist, kde=True, color="#ADD8E6", label="Intervention A", ax=axes[1],alpha=0.5)
sns.histplot(qaly_intervention_B_dist, kde=True, color="#90EE90", label="Intervention B", ax=axes[1],alpha=0.5)
axes[1].set_title('QALY Distribution', fontsize=18, fontweight='bold')
axes[1].set_xlabel('QALYs', fontsize=18, fontweight='bold')
axes[1].set_ylabel('', fontsize=18, fontweight='bold')
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].grid(True, linestyle='--', alpha=0.2)
plt.tight_layout()
# Save the figure
save_folder = 'C:/Users/bc22/OneDrive - King\'s College London/KCL/Projects/HE_code/'
plt.savefig(save_folder + 'CE_PSA_kde.png', dpi=300, bbox_inches='tight')
plt.show()

######################################################################################################################################################
# Function definitions
######################################################################################################################################################
# Calculate the total cost of an intervention by summing base costs, and costs from outpatient visits and tests.
def total_cost(base_cost, visits, visit_cost, tests, test_cost):
    return base_cost + (visits * visit_cost) + (tests * test_cost)

# Discounting function
def discount_value(value, rate, years):
    return value / ((1 + rate) ** years)

# Calculate the cost per QALY
def calculate_cost_per_qaly(cost, qaly):
    if qaly == 0:
        return float('inf')  # Avoid division by zero; assume infinite cost effectiveness
    return cost / qaly

# Calculate ICER based on QALYs
# An ICER shows the extra costs divided by the extra benefit when comparing two treatments or interventions. 
# It shows how much more you have to spend to gain an additional unit of health benefit (like one extra healthy year, QALY). 
# This makes it easier to decide if the extra cost is worth the extra benefit.
def calculate_icer(cost1, cost2, qaly1, qaly2):
    """Calculates the Incremental Cost-Effectiveness Ratio (ICER) between two interventions."""
    delta_cost = abs(cost1 - cost2)
    delta_qaly = qaly1 - qaly2
    if delta_qaly == 0:
        return float('inf')  # Prevent division by zero
    return delta_cost / delta_qaly

######################################################################################################################################################
# Run PSA simulations
#####################################################################################################################################################
results = []
for i in range(n_simulations):
    # Draw random values from distributions
    cost_A = np.random.choice(cost_intervention_A_dist)
    cost_B = np.random.choice(cost_intervention_B_dist)
    qaly_A = np.random.choice(qaly_intervention_A_dist)
    qaly_B = np.random.choice(qaly_intervention_B_dist)
    
    # Calculate total costs
    total_cost_A = total_cost(cost_A, number_of_visits_A, cost_per_outpatient_visit, number_of_tests_A, cost_per_test)
    total_cost_B = total_cost(cost_B, number_of_visits_B, cost_per_outpatient_visit, number_of_tests_B, cost_per_test)
    
    # Apply discounting
    discounted_cost_A = discount_value(total_cost_A, discount_rate, years)
    discounted_cost_B = discount_value(total_cost_B, discount_rate, years)

    # Calculate cost per QALY
    cost_per_qaly_A = calculate_cost_per_qaly(discounted_cost_A, qaly_A)
    cost_per_qaly_B = calculate_cost_per_qaly(discounted_cost_B, qaly_B)

    # Calculate ICER
    icer = calculate_icer(discounted_cost_A, discounted_cost_B, qaly_A, qaly_B)
    
    # Results is list of output of simulations
    results.append((discounted_cost_A, discounted_cost_B, qaly_A, qaly_B, icer))

######################################################################################################################################################
# Convert results to a DataFrame
#####################################################################################################################################################
PSA_results = pd.DataFrame(results, columns=['Discounted Cost A', 'Discounted Cost B', 'QALY A', 'QALY B', 'ICER'])

######################################################################################################################################################
# Print summary statistics
#####################################################################################################################################################
print(PSA_results.describe())

######################################################################################################################################################
# Plot results with CEA plane
#####################################################################################################################################################
delta_cost = PSA_results['Discounted Cost A'] - PSA_results['Discounted Cost B']
delta_qaly = PSA_results['QALY A'] - PSA_results['QALY B']

plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Calibri'
plt.scatter(delta_qaly, delta_cost, color='#3498db', label='PSA results', s=20, edgecolors='#2980b9', zorder=5)
plt.axhline(0, color='black', linestyle='-')  # Horizontal zero line
plt.axvline(0, color='black', linestyle='-')  # Vertical zero line

# Plot WTP threshold
x_values = [min(-5, -max(delta_qaly) * 1.2), max(5, max(delta_qaly) * 1.2)]
y_values = [x * wtp for x in x_values]
plt.plot(x_values, y_values, color='#d62728', linestyle=':', label='WTP threshold')

# Formatting plot
plt.xlim(-max(delta_qaly) * 1.2, max(delta_qaly) * 1.2)
plt.ylim(-max(delta_cost) * 1.2, max(delta_cost) * 1.2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\Delta$ QALYs', fontsize=18, fontweight='bold')
plt.ylabel(r'$\Delta$ Cost (£)', fontsize=18, fontweight='bold')
plt.title('Cost-Effectiveness Plane', fontsize=18, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.2)
# Save the figure
save_folder = 'C:/Users/bc22/OneDrive - King\'s College London/KCL/Projects/HE_code/'
plt.savefig(save_folder + 'CE_plane_PSA.png', dpi=300, bbox_inches='tight')
plt.show()

######################################################################################################################################################
# Summarise the results in a DataFrame
#####################################################################################################################################################
# Here we calculate the mean ICER across all simulations as a summary
mean_icer = PSA_results['ICER'].mean()
mean_icer_stable = delta_cost.mean() / delta_qaly.mean()

df_summary = pd.DataFrame({
    'Mean Discounted Cost': [PSA_results['Discounted Cost A'].mean(), PSA_results['Discounted Cost B'].mean()],
    'Mean QALYs': [PSA_results['QALY A'].mean(), PSA_results['QALY B'].mean()],
    'Mean Incremental Cost': [delta_cost.mean(), None],
    'Mean Incremental QALY': [delta_qaly.mean(), None],
    'Mean ICER': [mean_icer_stable, None]
}, index=['Intervention A', 'Intervention B'])

print(df_summary)

######################################################################################################################################################
# Calculate Jackknife 95% CI
#####################################################################################################################################################
# Define a function to calculate the mean ICER excluding each observation
def jackknife_mean(df, column):
    n = len(df)
    jackknife_means = []
    for i in range(n):
        # Exclude the ith observation and calculate the mean
        jackknife_sample = df.drop(index=i)
        jackknife_mean = jackknife_sample[column].mean()
        jackknife_means.append(jackknife_mean)
    return np.array(jackknife_means)

# Calculate the jackknife mean ICERs
jackknife_means = jackknife_mean(PSA_results, 'ICER')

# Calculate the jackknife estimate of variance
n = len(PSA_results)
jackknife_variance = (n - 1) / n * np.sum((jackknife_means - mean_icer) ** 2)

# Calculate the standard error
jackknife_se = np.sqrt(jackknife_variance)

# Calculate the 95% CI
confidence_level = 0.95
z_score = 1.96  # For 95% confidence interval

lower_bound = mean_icer - z_score * jackknife_se
upper_bound = mean_icer + z_score * jackknife_se

print(f"95% Jackknife Confidence Interval for the Mean ICER: (£{lower_bound:.2f}, £{upper_bound:.2f})")

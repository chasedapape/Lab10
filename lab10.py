# Eric Reid
# 11/1/2026
# Python for Health Analytics

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

csv_file = "km_logrank_readmission.csv"

# Specifies the column names
time_column = "time_days"
event_column = "readmitted"
group_column = "treatment_group"

# Loads the data
df = pd.read_csv(csv_file)

# Initialize the fitter
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))

# Plot survival curves
for group_name, group_df in df.groupby(group_column):
    kmf.fit(
        durations=group_df[time_column],
        event_observed=group_df[event_column],
        label=f"Group: {group_name}",
    )
    kmf.plot_survival_function(ax=ax, show_censors=True)

# Log-rank test
standard = df[df[group_column] == "Standard care"]
telehealth = df[df[group_column] == "Telehealth follow-up"]

logrank_result = logrank_test(
    durations_A=standard[time_column],
    durations_B=telehealth[time_column],
    event_observed_A=standard[event_column],
    event_observed_B=telehealth[event_column],
)


# Print results
print("Log-rank test: Standard care vs. Telehealth follow-up")
print(f"Standard care patients: {len(standard)}")
print(f"Telehealth patients: {len(telehealth)}")
print(f"Test statistic: {logrank_result.test_statistic:.4f}")
print(f"P-value: {logrank_result.p_value:.4g}")

# Add log-rank p-value to plot
p = logrank_result.p_value
p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

ax.text(
    0.97, 0.97,
    f"Log-rank test\n{p_text}",
    transform=ax.transAxes,
    ha="right", va ="top",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Formatting the plot
plt.title("Kaplan-Meier Readmission-Free Survival", fontsize=14)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left")
plt.tight_layout()

# Save and display the plot
plt.savefig("kaplan_meier_logrank.png")
plt.show()

# Cox proportional hazards analysis

# Desired data file
cox_file = "cox_ph_heart_failure_survival.csv"

# Specifies the columns
cox_time_column = "time_months"
cox_event_column = "death"

numeric_covariates = ["age", "bmi", "systolic_bp", "ejection_fraction", "nyha_class"]

binary_covariates = ["sex", "diabetes", "smoker", "chronic_kidney_disease","beta_blocker"]

# Load the data
cox_df = pd.read_csv(cox_file)

# Convert sex to a numeric value
cox_df["sex"] = cox_df["sex"].astype(str).str.strip().str.lower().isin(["male"]).astype(int)

#prep the models data
model_df = cox_df[[cox_time_column, cox_event_column] + numeric_covariates + binary_covariates]

model_df = model_df.dropna()

# Fit the cox proportional hazards model
cph = CoxPHFitter()
cph.fit(model_df, duration_col=cox_time_column, event_col=cox_event_column)

# Print the results
cph.print_summary(decimals=3)
print(f"\nConcordance index: {cph.concordance_index_:.3f}")

#Check the proportional hazards assumption
print("\n--- Proportional hazards assumptions check ---")
cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)

# plot the hazard ratios
fig, ax = plt.subplots(figsize=(8, 5))
cph.plot(hazard_ratios=True, ax=ax)

ax.axvline(1, color="gray", linestyle="--")
ax.set_title("Cox PH Model: Hazard Ratios(95% CI)", fontsize=12)
ax.set_xlabel("Hazard Ratio", fontsize=12)

plt.tight_layout()

plt.savefig("cox_hazard_ratios.png")
plt.show()


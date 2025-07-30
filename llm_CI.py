import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from dowhy import CausalModel
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ===================== 1. LOAD & PREPROCESS DATA =====================
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

df = pd.read_csv("diabetes.csv")
print(df)
print(df.columns)


# Replace zero with NaN and drop missing
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, np.nan)
df.dropna(inplace=True)

# Define binary treatment variable
treatment = "Glucose_bin"
df[treatment] = (df["Glucose"] > df["Glucose"].median()).astype(int)

outcome = "Outcome"
confounders = ['Age', 'BMI', 'Pregnancies']

# Normalize confounders
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[confounders] = scaler.fit_transform(df[confounders])

# ===================== 2. DOWHY CAUSAL INFERENCE =====================

model = CausalModel(
    data=df_scaled,
    treatment=treatment,
    outcome=outcome,
    common_causes=confounders
)

identified_estimand = model.identify_effect()

estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    target_units="ate"
)

print("Causal Estimate (ATE): ", estimate.value)

# ===================== 3. PROPENSITY SCORE ANALYSIS =====================

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(df_scaled[confounders], df_scaled[treatment])
pscore = ps_model.predict_proba(df_scaled[confounders])[:, 1]
df_scaled['propensity_score'] = pscore

# Plot KDE
plt.figure(figsize=(8, 5))
sns.kdeplot(df_scaled.loc[df_scaled[treatment] == 1, 'propensity_score'], label='Treated')
sns.kdeplot(df_scaled.loc[df_scaled[treatment] == 0, 'propensity_score'], label='Control')
plt.title("Propensity Score Distribution by Treatment Group")
plt.xlabel("Propensity Score")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig("./figs/propensityScore_Distribution_by_treatmentGroup.png", dpi=300)

# Plot Overlap Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df_scaled, x='propensity_score', hue=treatment, element="step", stat="density", common_norm=False)
plt.title("Propensity Score Overlap by Treatment Group")
# plt.show()
plt.savefig("./figs/propensityScoreOverlap_by_TreatmentGroup.png", dpi=300)

# ===================== 4. REFUTATION TESTS =====================

print("\nRunning refutation tests...\n")
refute_placebo = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
refute_random_common_cause = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
refute_subset = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter")

# ===================== 5. CREATE SUMMARY TEXT FOR LLaMA =====================

def summarize_feature(series):
    return f"mean={series.mean():.2f}, std={series.std():.2f}, min={series.min():.2f}, max={series.max():.2f}"

summary_text = f"""
Dataset summary for causal inference:

Treatment variable: {treatment} (binary indicator if glucose > median)
  {summarize_feature(df_scaled[treatment])}

Outcome variable: {outcome} (binary diabetes diagnosis)
  Outcome distribution: {df_scaled[outcome].value_counts().to_dict()}

Confounders:
"""
for c in confounders:
    summary_text += f"  {c}: {summarize_feature(df_scaled[c])}\n"

summary_text += f"""

Estimated Average Treatment Effect (ATE) of {treatment} on {outcome}: {estimate.value:.4f}

Diagnostics:
- Propensity score distribution plotted.
- Common support assumption checked.
- Refutation tests performed:
  * Placebo treatment refuter: {refute_placebo.new_effect}
  * Random common cause refuter: {refute_random_common_cause.new_effect}
  * Data subset refuter: {refute_subset.new_effect}

Please provide a detailed interpretation of this causal effect estimate,
discuss assumptions, potential sources of bias, robustness, and how
propensity score model diagnostics and sensitivity analyses inform confidence
in this estimate.
"""

print("=== Summary Text prepared for LLaMA fine-tuning ===")
print(summary_text)

# ===================== 6. LLaMA MODEL FINE-TUNING =====================

model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your local path if needed
tokenizer = LlamaTokenizer.from_pretrained(model_name)
llama_model = LlamaForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Fix: Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token
llama_model.config.pad_token_id = tokenizer.pad_token_id

# Prepare training dataset
train_texts = [summary_text]

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts})
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./llama_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    evaluation_strategy="no",
    report_to=None,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=llama_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Starting fine-tuning LLaMA model...")
trainer.train()

trainer.save_model("./llama_finetuned")
tokenizer.save_pretrained("./llama_finetuned")

print("Fine-tuning complete and model saved.")

# ===================== 7. LOAD FINE-TUNED MODEL FOR GENERATION =====================

print("Loading fine-tuned model for generation...")
finetuned_tokenizer = LlamaTokenizer.from_pretrained("./llama_finetuned")
finetuned_model = LlamaForCausalLM.from_pretrained("./llama_finetuned").to('cuda' if torch.cuda.is_available() else 'cpu')

def llama_generate(prompt, max_new_tokens=350):
    inputs = finetuned_tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
    outputs = finetuned_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=finetuned_tokenizer.eos_token_id,
    )
    return finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate interpretation
print("\n=== Generating interpretation from fine-tuned LLaMA model ===\n")
output = llama_generate(summary_text)
print(output)

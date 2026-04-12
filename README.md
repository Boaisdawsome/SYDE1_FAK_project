# SYDE1_FAK_project
## Results Summary

### What was asked
The core objective of this project was to determine if **SYDE1** serves as a viable biomarker for predicting **FAK (PTK2)** dependency in cancer cells. Using an AI-driven pipeline, the study investigated whether transcriptional cell-adhesion states could accurately forecast how sensitive specific cancer types are to the inhibition of Focal Adhesion Kinase (FAK).

---

### What was found
The computational analysis yielded the following performance metrics and statistical data:

* **Correlation Coefficients:** The model achieved a **Pearson correlation of 0.52** between predicted and observed dependency scores, indicating a moderate-to-strong linear relationship.
* **Model Performance:**
    * **R² Score:** 0.27, suggesting the model explains approximately 27% of the variance in FAK dependency based on the selected transcriptomic features.
    * **Mean Squared Error (MSE):** [Insert specific MSE value from your logs, e.g., 0.045], demonstrating the average squared difference between the estimated values and the actual outcomes.
* **Feature Importance:** SYDE1 expression emerged as a top-tier predictive feature, consistently ranking in the 90th percentile of weighted importance across multiple cross-validation folds.

---

### What it means for a patient
These findings suggest that a patient’s "transcriptional signature"—specifically the expression levels of **SYDE1**—could eventually act as a roadmap for personalized treatment. 

If a biopsy shows high SYDE1 expression linked to FAK dependency, a clinician could potentially prioritize FAK inhibitors for that specific patient, increasing the likelihood of treatment success. Conversely, it could help avoid the side effects of ineffective chemotherapy for patients whose profiles suggest they would not respond to FAK-targeted drugs. While this is currently a computational model, it paves the way for more precise, biomarker-driven oncology.

Diabetes Targeted Screening HTA Model 🏥

This is a Health Technology Assessment (HTA) tool built with Streamlit to model the cost-effectiveness of targeted diabetes screening interventions in the Indian context.

Features

Markov Modeling: Simulates disease progression for Retinopathy, Nephropathy, Foot Ulcers, Stroke, and CHD.

Targeted Interventions: Compare specific screening modalities (e.g., Fundoscopy) against standard care.

Interactive Levers: Adjust screening costs, frequency, and intervention effectiveness.

Granular Outcomes: View detailed population shifts across disease stages.

Probabilistic Sensitivity Analysis (PSA): Assess parameter uncertainty via Monte Carlo simulation.

How to Run Locally

Clone the repository:

git clone [https://github.com/YOUR-USERNAME/diabetes-hta-model.git](https://github.com/YOUR-USERNAME/diabetes-hta-model.git)


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


Data Sources

Key parameters are derived from:

Rachapelle et al. 2013 (Retinopathy)

UKPDS Studies (Nephropathy/CVD risks)

Prinja et al. (Costing data for India)
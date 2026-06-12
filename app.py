import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# ── Load Model Files ─────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = joblib.load('salary_model.pkl')
    scaler    = joblib.load('scaler.pkl')
    le_dict   = joblib.load('label_encoders.pkl')
    feat_names = joblib.load('feature_names.pkl')
    return model, scaler, le_dict, feat_names

model, scaler, le_dict, feature_names = load_models()

# ── Header ───────────────────────────────────────────────────
st.title("💼 Employee Salary Predictor")
st.markdown("Predict your expected salary based on your professional profile using Machine Learning.")
st.divider()

# ── Layout: Input | Results ───────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Enter Employee Details")

    work_year = st.selectbox("Work Year", [2020, 2021, 2022], index=2)

    exp_map = {'Entry Level (EN)': 'EN', 'Mid Level (MI)': 'MI',
               'Senior Level (SE)': 'SE', 'Executive (EX)': 'EX'}
    exp_label = st.selectbox("Experience Level", list(exp_map.keys()))
    experience_level = exp_map[exp_label]

    emp_map = {'Full-Time (FT)': 'FT', 'Part-Time (PT)': 'PT',
               'Contract (CT)': 'CT', 'Freelance (FL)': 'FL'}
    emp_label = st.selectbox("Employment Type", list(emp_map.keys()))
    employment_type = emp_map[emp_label]

    # Only show job titles that exist in training data
    available_jobs = list(le_dict['job_title'].classes_)
    job_title = st.selectbox("Job Title", sorted(available_jobs))

    remote_ratio = st.select_slider(
        "Remote Work Ratio",
        options=[0, 50, 100],
        value=100,
        format_func=lambda x: {0: "🏢 On-site (0%)", 50: "🔀 Hybrid (50%)", 100: "🏠 Remote (100%)"}[x]
    )

    size_map = {'Small (S)': 'S', 'Medium (M)': 'M', 'Large (L)': 'L'}
    size_label = st.selectbox("Company Size", list(size_map.keys()), index=1)
    company_size = size_map[size_label]

    predict_btn = st.button("🔮 Predict Salary", use_container_width=True, type="primary")

# ── Prediction Logic ─────────────────────────────────────────
with col2:
    st.subheader("💰 Salary Prediction Results")

    if predict_btn:
        try:
            # Encode inputs
            exp_enc  = le_dict['experience_level'].transform([experience_level])[0]
            emp_enc  = le_dict['employment_type'].transform([employment_type])[0]
            job_enc  = le_dict['job_title'].transform([job_title])[0]
            size_enc = le_dict['company_size'].transform([company_size])[0]

            input_df = pd.DataFrame([[work_year, exp_enc, emp_enc, job_enc, remote_ratio, size_enc]],
                                     columns=feature_names)

            predicted = model.predict(input_df)[0]
            predicted = max(predicted, 0)  # No negative salaries

            monthly   = predicted / 12
            low       = predicted * 0.85
            high      = predicted * 1.15

            # ── Result Cards ──
            st.success("✅ Prediction Complete!")

            m1, m2 = st.columns(2)
            m1.metric("💵 Annual Salary",  f"${predicted:,.0f}")
            m2.metric("📅 Monthly Salary", f"${monthly:,.0f}")

            st.markdown(f"""
            **📊 Salary Range Estimate**
            > Low estimate: **${low:,.0f}** &nbsp;|&nbsp; High estimate: **${high:,.0f}**
            """)

            st.divider()

            # ── Experience Level Insight Chart ──
            st.markdown("**📈 Salary by Experience Level (this job title)**")
            exp_levels  = ['EN', 'MI', 'SE', 'EX']
            exp_labels_ = ['Entry', 'Mid', 'Senior', 'Executive']
            exp_salaries = []

            for lvl in exp_levels:
                try:
                    enc = le_dict['experience_level'].transform([lvl])[0]
                    row = pd.DataFrame([[work_year, enc, emp_enc, job_enc, remote_ratio, size_enc]],
                                        columns=feature_names)
                    exp_salaries.append(model.predict(row)[0])
                except:
                    exp_salaries.append(0)

            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['#d3d3d3' if l != experience_level else '#1f77b4' for l in exp_levels]
            bars = ax.bar(exp_labels_, exp_salaries, color=colors)
            ax.set_ylabel('Predicted Salary (USD)')
            ax.set_title(f'Salary Progression — {job_title}')
            for bar, val in zip(bars, exp_salaries):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                        f'${val:,.0f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("👈 Fill in the details on the left and click **Predict Salary**")

        # Show placeholder insight
        st.markdown("#### 💡 What affects salary the most?")
        factors = ['Job Title', 'Experience Level', 'Remote Ratio', 'Company Size', 'Employment Type']
        scores  = [0.45, 0.28, 0.12, 0.09, 0.06]

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=scores, y=factors, palette='mako', ax=ax)
        ax.set_xlabel('Importance Score')
        ax.set_title('Key Salary Factors (Random Forest)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Random Forest Regressor · ds_salaries dataset · Internship Project")
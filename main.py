import streamlit as st
import pandas as pd
from bias_modules.llm_calls import ModelHandler, Constants
from bias_modules.stat import class_balance_checker, chi_square_test, text_length_classifier_deviation

def dataset_uploader():
    if st.session_state["formstate"] == 1:
        uploaded_file = st.file_uploader("Upload your training dataset (.csv)", type="csv")

        if uploaded_file:
            st.session_state["dataset"] = pd.read_csv(uploaded_file)
            st.session_state["formstate"] = 2
            st.rerun()
    
    if st.session_state["formstate"] == 2:
        columns_list = st.session_state["dataset"].columns.tolist()

        text_column = st.selectbox("Select the column containing the text", options=columns_list)
        label_column = st.selectbox("Select the label column", options=columns_list)

        next_button = st.button("Next")
        if next_button:
            st.session_state["text_column"] = text_column
            st.session_state["label_column"] = label_column
            st.session_state["formstate"] = 3
            st.rerun()
    
    if st.session_state["formstate"] == 3:
        st.write("Dataset details:")

        if "dataset_purpose" not in st.session_state:
            llm = ModelHandler()
            prompt = Constants.get_classifier_determination_system_prompt(st.session_state["label_column"], st.session_state["text_column"], st.session_state["dataset"][st.session_state["label_column"]].unique().tolist())
            dataset_purpose = llm.generate_response(prompt)
            st.session_state["dataset_purpose"] = dataset_purpose
        else:
            dataset_purpose = st.session_state["dataset_purpose"]
        dataset_purpose = st.text_area("What is the purpose of this dataset?", value=dataset_purpose)

        next_button = st.button("Analyze")

        if next_button:
            st.session_state["dataset_purpose"] = dataset_purpose
            st.session_state["formstate"] = 4
            st.rerun()

    if st.session_state["formstate"] == 4:
            with st.spinner("Analyzing dataset... Please be patient"):
                class_dist, balanced = class_balance_checker(st.session_state["dataset"], st.session_state["label_column"])
                chi_df = chi_square_test(st.session_state["dataset_purpose"], st.session_state["dataset"], st.session_state["text_column"], st.session_state["label_column"])
                length_df, lbalanced = text_length_classifier_deviation(st.session_state["dataset"], st.session_state["text_column"], st.session_state["label_column"])
            
            if balanced:
                st.write(":white_check_mark: The dataset is balanced")
            else:
                st.write(":x: The dataset is imbalanced")
                st.bar_chart(class_dist, x_label=st.session_state["label_column"], y_label="Count of rows in dataset")

            if chi_df["response"].str.contains("INVALID").any():
                st.write(":x: The dataset contains textual bias [Chi Square Test]" + f" | Potential biased words: {', '.join(chi_df[chi_df['response'].str.contains('INVALID', na=False)]['word'].tolist())}")
                st.write(chi_df)
            else:
                st.write(":white_check_mark: The dataset does not contain textual bias [Chi Square Test]")

            if lbalanced:
                st.write(":white_check_mark: The dataset is balanced in terms of text length")
            else:
                st.write(":x: The dataset is imbalanced in terms of text length")
                st.write(length_df)

def main():
    st.title("DetBias")
    st.write("A bias detection tool (mitigation coming soon)")
    
    if "formstate" not in st.session_state:
        st.session_state["formstate"] = 1
    dataset_uploader()

if __name__ == "__main__":
    main()

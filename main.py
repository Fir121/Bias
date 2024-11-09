import streamlit as st
import pandas as pd
from bias_modules.llm_calls import ModelHandler, Constants

def dataset_uploader():
    uploaded_file = st.file_uploader("Upload your training dataset (.csv)", type="csv")
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        columns_list = dataset.columns.tolist()

        text_column = st.selectbox("Select the column containing the text", options=columns_list)
        label_column = st.selectbox("Select the label column", options=columns_list)

        if text_column is not None and label_column is not None:
            st.write("Dataset details:")
    
            llm = ModelHandler()
            prompt = Constants.get_classifier_determination_system_prompt(label_column, text_column, dataset[label_column].unique().tolist())
            dataset_purpose = llm.generate_response(prompt)
            dataset_purpose = st.text_area("What is the purpose of this dataset?", value=dataset_purpose)

            next_button_2 = st.button("Next")


def main():
    st.title("Bias Detector")
    st.write("Upload your dataset to begin")
    
    dataset_uploader()

if __name__ == "__main__":
    main()

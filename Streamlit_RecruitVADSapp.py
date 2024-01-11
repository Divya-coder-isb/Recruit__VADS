#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import pickle
import urllib.request  # Import urllib.request to fetch data from URLs

# Load model from GitHub raw link
model_url = "https://github.com/Divya-coder-isb/Recruit__VADS/raw/main/Recruit_VADS_model.pkl"
model = pickle.load(urllib.request.urlopen(model_url))

# Load vectorizer from GitHub raw link
vectorizer_url = "https://github.com/Divya-coder-isb/Recruit__VADS/raw/main/Tfidf_Vectorizer.pkl"
vectorizer = pickle.load(urllib.request.urlopen(vectorizer_url))

# Load CSV data from GitHub raw link
resume_data_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit__VADS/main/Modifiedresumedata_data.csv"
resume_data = pd.read_csv(resume_data_url)

def main():
    st.title("Recruit VADS App")

    job_title = st.text_input("Job Title")
    skills = st.text_input("Skills")
    experience = st.text_input("Experience")
    certification = st.text_input("Certification")

    if st.button("Apply"):
        relevancy_score = get_relevancy_score(job_title, skills, certification, experience)
        st.table(relevancy_score)

def get_relevancy_score(job_title, skills, certification, experience):
    input_features = [job_title, skills, certification, experience]
    input_vector = vectorizer.transform(input_features).toarray()

    similarity = model.predict(input_vector)

    sorted_indices = similarity.argsort(axis=0)[::-1]
    sorted_similarity = similarity[sorted_indices]

    output = pd.DataFrame()
    output['Candidate Name'] = resume_data['Candidate Name'][sorted_indices].squeeze()
    output['Email ID'] = resume_data['Email ID'][sorted_indices].squeeze()
    output['Relevancy Score'] = (sorted_similarity * 100).round(2).squeeze()
    output['Relevancy Score'] = output['Relevancy Score'].astype(str) + '%'

    return output

if __name__ == "__main__":
    main()


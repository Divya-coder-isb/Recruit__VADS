#!/usr/bin/env python
# coding: utf-8

# In[1]:


# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the model and vectorizer
model = pickle.load(open('Recruit_VADS_model.pkl', 'rb'))
vectorizer = pickle.load(open('Tfidf_Vectorizer.pkl', 'rb'))

# Load resume data
resume_data_path = "Modifiedresumedata_data.csv"
resume_data = pd.read_csv(resume_data_path)

# Define the Streamlit app
def main():
    st.title("Recruit VADS App")

    # Input fields
    job_title = st.text_input("Job Title")
    skills = st.text_input("Skills")
    experience = st.text_input("Experience")
    certification = st.text_input("Certification")

    # Apply button
    if st.button("Apply"):
        # Get relevancy score using the model
        relevancy_score = get_relevancy_score(job_title, skills, certification, experience)

        # Display the results in a table
        st.table(relevancy_score)

# Define a function that takes input from the UI and returns the relevancy score
def get_relevancy_score(job_title, skills, certification, experience):
    input_features = [job_title, skills, certification, experience]
    input_vector = vectorizer.transform(input_features).toarray()

    # Compute the cosine similarity with the model
    similarity = model.predict(input_vector)

    # Sort the candidates by descending order of similarity
    sorted_indices = similarity.argsort(axis=0)[::-1]
    sorted_similarity = similarity[sorted_indices]

    # Format the output as a dataframe with candidate name, email, and relevancy score
    output = pd.DataFrame()
    output['Candidate Name'] = resume_data['Candidate Name'][sorted_indices].squeeze()
    output['Email ID'] = resume_data['Email ID'][sorted_indices].squeeze()
    output['Relevancy Score'] = (sorted_similarity * 100).round(2).squeeze()
    output['Relevancy Score'] = output['Relevancy Score'].astype(str) + '%'

    return output

# Run the Streamlit app
if __name__ == "__main__":
    main()


# In[ ]:





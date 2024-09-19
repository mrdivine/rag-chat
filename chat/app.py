# chat/app.py
import streamlit as st
import requests


def main():
    st.title("RAG Chat Interface")

    user_input = st.text_input("Enter your query:")
    if st.button("Submit"):
        response = requests.get("http://rag-service:8000/generate_response", params={"query": user_input})
        if response.status_code == 200:
            result = response.json()
            st.write("RAG Response:", result["response"])
        else:
            st.write("Error in RAG service")


if __name__ == "__main__":
    main()

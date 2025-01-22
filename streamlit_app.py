import streamlit as st
import os
from groq import Groq
from PyPDF2 import PdfReader

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to generate response based on the query
def generate_response(text, query):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    # Construct the prompt with context from PDF
    prompt = f"""Based on the following text from a PDF document, please answer the question.

Text: {text}

Question: {query}

Please provide a clear and concise answer based only on the information provided in the text."""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",  # Updated to use the correct model name
        temperature=0.1,  # Added for more focused responses
        max_tokens=1024,  # Added to control response length
    )
    
    return chat_completion.choices[0].message.content

# Streamlit UI setup
st.title("PDF Question Answering System")

# File uploader for PDF files
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf("temp.pdf")
    
    # Show a preview of the extracted text
    with st.expander("Preview PDF Content"):
        st.text(pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text)
    
    # Input field for user's question
    query_text = st.text_input("Enter your question about the PDF content:")
    
    if st.button("Submit"):
        if query_text:
            with st.spinner("Generating response..."):
                try:
                    response = generate_response(pdf_text, query_text)
                    st.write("**Response:**", response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Add a button to clear the uploaded file
    if st.button("Clear"):
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        st.experimental_rerun()

# Clean up temporary file after processing
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
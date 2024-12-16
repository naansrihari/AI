import openai
from transformers import pipeline
from PyPDF2 import PdfReader
import pandas as pd
from docx import Document
import os

# Set your OpenAI API key (make sure to replace this with your own key)
openai.api_key = os.getenv("OPENAI_API_KEY", "Open API key replace panna venum")  # Better to store it as an env variable


def extract_text_from_file(file_path):
    """
    Extract text content from supported file types (.txt, .pdf, .docx, .csv).
    """
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            return " ".join(page.extract_text() for page in reader.pages)
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            return " ".join(paragraph.text for paragraph in doc.paragraphs)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            return df.to_string()  # Convert the dataframe to a string
        else:
            raise ValueError("Unsupported file type!")
    except Exception as e:
        return f"Error extracting content from file: {e}"


def ask_question_openai(content, question):
    """
    Ask a question using OpenAI API (ChatGPT).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can also use "gpt-4" for better performance (higher cost)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"The following is the file content: {content}"},
                {"role": "user", "content": question}
            ]
        )
        return response['choices'][0]['message']['content']
    except openai.error.RateLimitError:
        return "You have exceeded your OpenAI API quota. Please upgrade your plan or try later."
    except openai.error.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {e}"


def ask_question_local(content, question):
    """
    Ask a question using a local Hugging Face model (fallback when API is exceeded).
    """
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        result = qa_pipeline(question=question, context=content)
        return result['answer']
    except Exception as e:
        return f"An error occurred with the local model: {e}"


def ask_question(content, question):
    """
    Try OpenAI API first; fallback to local processing if quota is exceeded.
    """
    print("Using OpenAI API for the response...")
    openai_response = ask_question_openai(content, question)
    if "exceeded your OpenAI API quota" in openai_response:
        print("Falling back to local Hugging Face model...")
        return ask_question_local(content, question)
    return openai_response


def main():
    """
    Main function to run the chatbot.
    """
    print("Welcome to the AI File-Based Chatbot!")
    file_path = input("Please upload a file (path): ")

    # Extract content from the provided file
    file_content = extract_text_from_file(file_path)
    if "Error" in file_content:
        print(file_content)
        return

    print("File successfully loaded. You can now ask questions based on the uploaded file.")
    while True:
        question = input("\nYour question (type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Ask the question and print the answer
        answer = ask_question(file_content, question)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import streamlit as st
import time
import logging

# Directly set API keys (paste your keys here)
groq_api_key = "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"
google_api_key = "AIzaSyCJQrqSmM40fsNywHNBIlEoVobOYX35ZMM"

# Set up the API keys in environment variables if needed for specific libraries
import os
os.environ['GOQ_API_KEY'] = groq_api_key
os.environ['GOOGLE_API_KEY'] = google_api_key

# Setup logging for metrics
logging.basicConfig(level=logging.INFO)

# Ground truth data: A dictionary of questions and reference answers
GROUND_TRUTH = {
    "What is AI?": "AI stands for Artificial Intelligence, which enables machines to simulate human intelligence.",
    "What is the capital of France?": "The capital of France is Paris.",
    "What is the largest planet in our solar system?": "The largest planet in our solar system is Jupiter.",
    "Explain machine learning in simple terms.": "Machine learning is a subset of AI where machines learn patterns from data to make decisions or predictions.",
}

# Function to compute ROUGE score
def compute_rouge(reference, generated):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# Function to compute BLEU score
def compute_bleu(reference, generated):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    reference = reference.split()  # Split reference into words
    generated = generated.split()  # Split generated into words
    smoothing_function = SmoothingFunction().method4  # Smoothing to avoid zero BLEU score
    bleu_score = sentence_bleu([reference], generated, smoothing_function=smoothing_function)
    return bleu_score

# Streamlit UI setup
st.set_page_config(page_title="QA BOT")

# Input field for user question
input_text = st.text_input("Input: ", key="input")

# Submit button for generating response
submit = st.button("Ask Question")

def get_response(question):
    start_time = time.time()  # Start time for response generation
    chatllm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=150,
        timeout=None,
        max_retries=2,
    )
    response = chatllm([HumanMessage(content=question)])
    end_time = time.time()  # End time after getting response
    
    # Calculate response time
    response_time = end_time - start_time
    logging.info(f"Response time: {response_time:.2f} seconds")  # Log response time
    
    # Get generated content
    generated_content = response.content if response and response.content else "No content received."

    # Fetch reference answer from ground truth
    reference_answer = GROUND_TRUTH.get(question, "No reference answer available.")

    # Calculate ROUGE and BLEU scores if a reference answer exists
    if reference_answer != "No reference answer available.":
        rouge_scores = compute_rouge(reference_answer, generated_content)
        bleu_score = compute_bleu(reference_answer, generated_content)
    else:
        rouge_scores = "N/A"
        bleu_score = "N/A"

    logging.info(f"ROUGE Scores: {rouge_scores}")  # Log ROUGE scores
    logging.info(f"BLEU Score: {bleu_score}")  # Log BLEU score

    return generated_content, response_time, rouge_scores, bleu_score, reference_answer

if submit:
    if input_text:
        response_content, response_time, rouge_scores, bleu_score, reference_answer = get_response(input_text)  # Get the response and metrics
        st.subheader("The response is:")
        st.write(response_content)  # Display the response content
        st.write(f"Reference Answer: {reference_answer}")  # Display reference answer
        st.write(f"Response Time: {response_time:.2f} seconds")  # Display response time metric
        
        # Check if metrics are available and display accordingly
        if rouge_scores != "N/A" and bleu_score != "N/A":
            st.write(f"ROUGE Scores: {rouge_scores}")  # Display ROUGE scores
            st.write(f"BLEU Score: {bleu_score:.2f}")  # Display BLEU score
        else:
            st.write("ROUGE Scores: N/A")  # Handle missing ROUGE scores
            st.write("BLEU Score: N/A")  # Handle missing BLEU score
    else:
        st.write("Please enter a question.")

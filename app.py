from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from utils.rag_utils import *

genai.configure(api_key="api_key")
model = genai.GenerativeModel("gemini-1.5-flash-002")

app = Flask(__name__)
index, chunks = load_resources("Total-Skincare-Guide.pdf")

def generate_response(query, context):
    prompt = f"""
You are a skincare assistant. Use the below context to answer the user's question with empathy and useful advice.
Context: {context}
User's question: {query}
Answer:
Also suggest skincare products.
"""
    response = model.generate_content(prompt)
    clean = response.text.strip().replace("*", "").replace("\n", "<br>")
    return clean
    return response.text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["query"]
    relevant_chunks = search_index(index, user_input, chunks)
    context = "\n".join(relevant_chunks)
    answer = generate_response(user_input, context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

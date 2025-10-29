# 🤖 Gen AI RAG Application for Q&A

A **Retrieval-Augmented Generation (RAG)** powered **Gen AI application** built using **LangChain**, **Hugging Face**, and **Groq LLM**, with an interactive **Streamlit** chat interface.

This app allows you to **ask questions about your PDF documents**, retrieves relevant context using **FAISS**, and generates intelligent, context-aware answers from your **LLM**.

---

## 🚀 Features

✅ Retrieval-Augmented Generation (RAG) Pipeline  
✅ Vector Store using FAISS  
✅ PDF Document Loading and Chunking  
✅ HuggingFace Sentence Embeddings  
✅ Chat Interface with Streamlit  
✅ Environment Variables with `.env`  
✅ Package Management using `uv` and `pyproject.toml`

---

## 🧱 Project Structure

```
Gen-Ai-Both-with-Rag/
│
├── src/
│   └── resources/
│       └── Attention.pdf      # PDF file(s) used for retrieval
│
├── .env                        # API keys and environment variables
├── pyproject.toml              # Dependency management with uv
├── app.py                      # Main Streamlit app
└── README.md                   # Project documentation
```

---

## ⚙️ Prerequisites

Make sure you have the following installed:

- **Python ≥ 3.10**
- **uv** (package manager)
- **Git**

---

## 🧩 Installation Steps

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2️⃣ Create Environment Using uv

```bash
uv venv
uv pip install -r requirements.txt  # OR directly install from pyproject.toml
```

If your dependencies are defined only in `pyproject.toml`, simply run:

```bash
uv sync
```

This installs all required libraries with the correct versions.

### 3️⃣ Setup Environment Variables

Create a `.env` file in your project root:

```bash
touch .env
```

Add your API keys:

```env
HF_TOKEN=your_huggingface_token
GROQ_KEY=your_groq_api_key
```

### 4️⃣ Run the Application Locally

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`) in your browser.

---

## 🧠 How It Works

### 1. Document Loading
The app loads your PDF (e.g., `resources/Attention.pdf`) using `PyPDFDirectoryLoader`.

### 2. Text Splitting
The text is split into smaller chunks using `RecursiveCharacterTextSplitter`.

### 3. Embedding Creation
Each chunk is converted into an embedding vector using `HuggingFaceEmbeddings`.

### 4. FAISS Vector Store
Embeddings are stored in FAISS for efficient similarity search.

### 5. Retrieval
When you ask a question, the most relevant chunks are retrieved based on similarity.

### 6. LLM Response
Retrieved chunks are passed to Groq's LLaMA model via LangChain's `create_stuff_documents_chain()` and `create_retrieval_chain()` to generate an accurate, context-based answer.

---

## 🧪 Example Interaction

**User:**
> What is the main concept of Attention mechanism?

**Assistant:**
> The attention mechanism allows neural networks to focus on specific parts of the input sequence when generating output, improving performance in NLP tasks.

---

## 📦 Key Dependencies

- **LangChain** - Framework for LLM applications
- **Streamlit** - Interactive web interface
- **FAISS** - Vector similarity search
- **HuggingFace Transformers** - Embeddings and models
- **Groq** - LLM inference
- **PyPDF** - PDF document loading
- **python-dotenv** - Environment variable management

---

## 🛠️ Troubleshooting

### Issue: `ModuleNotFoundError`
**Solution:** Make sure all dependencies are installed via `uv sync` or `uv pip install -r requirements.txt`

### Issue: API Key Error
**Solution:** Verify that your `.env` file contains valid `HF_TOKEN` and `GROQ_KEY` values

### Issue: FAISS Index Not Found
**Solution:** Ensure PDFs are placed in the correct `src/resources/` directory and the app has read permissions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 👨‍💻 Author

Created with ❤️ by **[Sai Aravind Koduru]**

- GitHub: [Sai Aravind Koduru](https://github.com/saiAravind-1104)
- LinkedIn: [Sai Aravind Koduru](https://www.linkedin.com/in/sai-aravind-koduru-a704a5222)

---

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for fast LLM inference
- **HuggingFace** for embeddings and models
- **Streamlit** for the amazing UI framework
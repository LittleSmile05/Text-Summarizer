# 📝 Advanced Text Summarizer

**A multi-technique text summarization tool combining traditional NLP and modern transformer approaches**

---

## 🔧 Technologies Used

### 🧠 Core NLP
- **`NLTK`** - Text preprocessing, tokenization & stopword removal  
- **`scikit-learn`** - TF-IDF vectorization & cosine similarity  
- **`NetworkX`** - Graph algorithms for TextRank implementation  

### 🤖 Deep Learning
- **`Transformers`** (Hugging Face) - BART abstractive summarization  
- **`PyTorch`** - Underlying DL framework  

### 🌐 Web Interface
- **`Streamlit`** - Interactive web app deployment  
- **`Matplotlib`** - Method comparison visualizations  

---

## ✨ Key Features

### 📑 Summarization Methods
| Method | Type | Description |
|--------|------|-------------|
| **Frequency-based** | Extractive | Scores sentences by word frequency |
| **TF-IDF** | Extractive | Uses term importance metrics |
| **TextRank** | Extractive | Graph-based ranking algorithm |
| **BART** | Abstractive | Generates new summary text |

### 📊 Evaluation Metrics
- ✅ Compression ratio (% text reduced)  
- 🔠 Vocabulary overlap (% terms preserved)  
- 📏 Length comparison (original vs summary)  

### 🔍 Comparison Tools
- 📈 Side-by-side method evaluation  
- 🎨 Interactive visualizations  

---

## 🎯 Purpose & Applications

### 🎓 Educational Value
- Demonstrates NLP technique progression  
- Practical transformer model implementation  

### 💼 Professional Use Cases
- News article summarization  
- Research paper abstraction  
- Legal/document review condensation  

### 🔬 Research Potential
- Framework for algorithm experimentation  
- Easy model extension/swap capability  

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run main.py

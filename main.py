def create_streamlit_app():
    """Streamlit veb tətbiqini yaradır """
    st.set_page_config(page_title="Advanced Text Summarizer", page_icon="📝", layout="wide")
    
    st.title("📝 Advanced Text Summarizer")
    st.markdown("""
    Bu tətbiq müxtəlif mətn xülasə texnikalarını göstərir:
    - **Tezlik əsaslı**: Yüksək tezlikli vacib sözləri olan cümlələri seçir
    - **TF-IDF**: Term Frequency-Inverse Document Frequency istifadə edərək vacib cümlələri müəyyənləşdirir
    - **TextRank**: Qraf əsaslı reytinq alqoritmi tətbiq edir (PageRank kimidir)
    - **Abstrakt**: BART transformer modeli ilə yeni xülasə yaradır
    
    Bu mənim ilk böyük layihəmdir! Universitetdə son kursam :)
    """)
    
    # Tablar yaradırıq - bunları özüm öyrəndim
    tab1, tab2, tab3 = st.tabs(["Mətn Xülasə", "Üsulları Müqayisə Et", "Haqqında"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area("Xülasə etmək üçün mətn daxil edin:", height=300)
            
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                summarization_method = st.selectbox(
                    "Xülasə üsulunu seçin:",
                    ["Tezlik əsaslı", "TF-IDF", "TextRank", "Abstrakt (BART)"]
                )
            
            with col1b:
                if summarization_method != "Abstrakt (BART)":
                    num_sentences = st.slider("Xülasədəki cümlə sayı:", 1, 10, 3)
                else:
                    max_length = st.slider("Maksimum xülasə uzunluğu (token):", 50, 250, 150)
                    min_length = st.slider("Minimum xülasə uzunluğu (token):", 10, 100, 50)
            
            with col1c:
                if st.button("Xülasə Et", type="primary"):
                    if text_input:
                        with st.spinner('Xülasə yaradılır... biraz gözləyin...'):
                            summarizer = Textimport nltk
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import matplotlib.pyplot as plt
import streamlit as st
import time

# Lazım olan kitabxanaları yükləyirik
# Mənə tələbə dostlarım bunu örgətdilər :D

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')  # Ay bu cümlələri ayırmaq üçündür
    nltk.download('stopwords')  # Stop sözlər, məsələn "və", "bu", "o"... 
    nltk.download('wordnet')  # Sözlər üçün lüğət kimi

class TextSummarizer:
    """Müxtəlif üsullarla mətn xülasə edən sinif."""
    
    def __init__(self):
        # Vay, ingilis dilində olan stop sözlər
        # Bunları çıxardacağıq çünki xülasədə vacib deyillər
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Bu BART modeli çox güclüdür! Müəllimim dedi ki, CNN-də istifadə olunub
        # Amma yükləməsi biraz vaxt apara bilər, internetim yavaşdır...
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Bu asan yoldur!
    
    def preprocess_text(self, text):
        """Mətni təmizləyir və hazırlayır."""
        # Hər şeyi kiçik hərflə yazırıq - nə fərqi var ki? :)
        text = text.lower()
        
        # Xüsusi simvollar və rəqəmləri silirik
        # Bir dəfə imtahanda bunu soruşdular, ona görə əzbərləmişəm
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Artıq boşluqları silirik, çünki çox boşluq nəyə lazımdır?
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_sentences(self, text):
        """Mətndən cümlələri ayırır."""
        # Bu asan funksiya, sadəcə NLTK-nın sent_tokenize istifadə edirik
        # Qrupumuzda hamı belə edir, kitabdakı kimi :D
        sentences = sent_tokenize(text)
        return sentences
    
    def extract_word_frequency(self, text):
        """Stop sözləri silir və sözlərin tezliyini hesablayır."""
        # Sözləri ayırırıq - məncə burda mənim kodumu müəllim bəyənəcək
        words = word_tokenize(text)
        
        # Stop sözləri silirik və lemmatize edirik (əsas formaya salırıq)
        # Məsələn, "running" -> "run" olur. Çox ağıllıdır!
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        # Sözlərin tezliyini hesablayırıq - bunu özüm yazmışam :)
        # Həmkarım dedi ki, Counter() istifadə etmək olar amma mən belə başa düşürəm
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        return word_freq
    
    def score_sentences(self, sentences, word_freq):
        """Cümlələri sözlərin tezliyinə görə qiymətləndirir."""
        # Hər cümlənin balını hesablayırıq
        # İlk dəfə bunu görəndə çox çətin idi, amma anladım ki əslində sadədir
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            # Cümlədəki hər sözə baxırıq
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:  # Əgər bu söz vacibdirsə
                    if i in sentence_scores:
                        sentence_scores[i] += word_freq[word]  # Balı artırırıq
                    else:
                        sentence_scores[i] = word_freq[word]  # İlk bal
        return sentence_scores
    
    def frequency_based_summary(self, text, num_sentences=3):
        """Söz tezliyi əsasında xülasə yaradır."""
        # İlk öncə mətni hazırlayırıq
        processed_text = self.preprocess_text(text)
        # Sonra cümlələri ayırırıq (orijinal mətndən, çünki daha yaxşı olur)
        sentences = self.extract_sentences(text)
        # Sözlərin tezliyini hesablayırıq
        word_freq = self.extract_word_frequency(processed_text)
        # Cümlələri qiymətləndiririk
        sentence_scores = self.score_sentences(sentences, word_freq)
        
        # Ən yüksək ballı n cümləni seçirik
        # sorted() funksiyasını dərsdə öyrəndik, çox faydalıdır!
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Mətndəki sıraya görə düzürük
        
        # Xülasəni yaradırıq - cümlələri sadəcə birləşdiririk
        summary = ' '.join([sentences[i] for i, _ in top_sentences])
        return summary
    
    def tfidf_based_summary(self, text, num_sentences=3):
        """TF-IDF istifadə edərək xülasə yaradır. Bu daha elmidir :)"""
        # Cümlələri ayırırıq
        sentences = self.extract_sentences(text)
        
        # TF-IDF vektorizatoru yaradırıq
        # Bunu dərslikdən kopyaladım, düzü tam başa düşmürəm, amma işləyir!
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Cümlələrin balını TF-IDF dəyərlərinə görə hesablayırıq
        # Burada bir az çətinlik çəkdim amma internetdən baxıb anladım
        sentence_scores = [sum(tfidf_matrix[i].toarray()[0]) for i in range(len(sentences))]
        
        # Ən yüksək ballı n cümləni seçirik
        # argsort() - indeksləri sıralayır, çox maraqlı funksiyadır!
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Mətndəki sıraya görə düzürük
        
        # Xülasəni yaradırıq
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def textrank_summary(self, text, num_sentences=3):
        """TextRank alqoritmi ilə xülasə yaradır - bu Google PageRank kimidir!"""
        # Cümlələri ayırırıq
        sentences = self.extract_sentences(text)
        
        # Oxşarlıq matrisini yaradırıq
        # Allah bilir bu necə işləyir amma müəllim dedi yaxşı üsuldur :D
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # TextRank alqoritmini NetworkX ilə tətbiq edirik
        # Qraflardakı tapşırığı edəndə bunu öyrəndim, çox mentalitet şeydir
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Ən yüksək ballı n cümləni seçirik
        ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        
        # Xülasəni yaradırıq
        summary = ' '.join([sentences[i] for _, i in top_sentences])
        return summary
    
    def abstractive_summary(self, text, max_length=150, min_length=50):
        """BART modeli ilə abstrakt xülasə yaradır - bu tamamilə yeni cümlələr yazır!"""
        # Mətni modelin anlaya biləcəyi şəkildə kəsirik
        # Tokens are weird stuff - rəqəm kimi görünür amma söz deməkdir?
        max_tokens = 1024
        inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=max_tokens, truncation=True)
        
        # Xülasəni generasiya edirik
        # Bu başağrısı parametrləri proyektdən kopyaladım, təsadüfən silmə :D
        summary_ids = self.model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=2.0,  # Nə üçün 2.0? İnternetdə belə yazılıb...
            num_beams=4,  # Məncə bu "nur" kimi bir şeydir
            early_stopping=True
        )
        
        # Xülasəni decode edirik (yəni oxunaqlı mətnə çeviririk)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def pipeline_summary(self, text, max_length=150, min_length=50):
        """Pipeline API ilə xülasə, daha asandır."""
        # Mətni kəsirik lazım gələrsə
        # 100000 çox böyük rəqəmdir, heç kim belə uzun mətn yazmaz!
        if len(text) > 100000:
            text = text[:100000]
            
        # Pipeline-da summarize edirik - bir sətirdə! Möhtəşəm!
        result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    
    def evaluate_summary(self, original_text, summary):
        """Xülasənin keyfiyyətini qiymətləndirir - əslində mənim fikrimcə subyektivdir!"""
        # Sıxılma nisbətini hesablayırıq - yəni nə qədər kiçiltdik
        original_words = len(word_tokenize(original_text))
        summary_words = len(word_tokenize(summary))
        compression = (1 - summary_words / original_words) * 100  # Faizə çeviririk
        
        # Lüğət örtüyünü hesablayırıq - bu mənim əlavəmdir
        # Yəni xülasədə orijinal mətndən nə qədər söz var?
        original_vocab = set(word_tokenize(self.preprocess_text(original_text)))
        summary_vocab = set(word_tokenize(self.preprocess_text(summary)))
        overlap = len(original_vocab.intersection(summary_vocab)) / len(original_vocab) * 100
        
        # Bütün məlumatları qaytarırıq
        # Dictionary yaratmaqda dictionary comprehension daha yaxşı olardı amma belə başa düşürəm
        return {
            "compression_ratio": compression,  # Nə qədər sıxılıb
            "vocabulary_overlap": overlap,     # Nə qədər söz saxlanılıb
            "original_length": original_words, # Orijinal uzunluq
            "summary_length": summary_words    # Xülasə uzunluğu
        }
    
    def visualize_summary_comparison(self, text, methods=['frequency', 'tfidf', 'textrank', 'abstractive']):
        """Müxtəlif xülasə üsullarını vizual müqayisə edir - bunu təqdimatda təqdim edə bilərəm!"""
        summaries = {}
        evaluation = {}
        
        # Müxtəlif üsullarla xülasələr yaradırıq
        # Loop nədir? Döngü, təkrar demək istəyirəm! :)
        for method in methods:
            if method == 'frequency':
                summaries[method] = self.frequency_based_summary(text)
            elif method == 'tfidf':
                summaries[method] = self.tfidf_based_summary(text)
            elif method == 'textrank':
                summaries[method] = self.textrank_summary(text)
            elif method == 'abstractive':
                summaries[method] = self.abstractive_summary(text)
            
            # Hər xülasəni qiymətləndiririk
            evaluation[method] = self.evaluate_summary(text, summaries[method])
        
        # Vizuallaşdırmaları yaradırıq
        # Bu qrafikləri də müəllimim öyrətdi, rəngarəng olur!
        metrics = ['compression_ratio', 'vocabulary_overlap', 'summary_length']
        method_names = list(evaluation.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [evaluation[method][metric] for method in method_names]
            axes[i].bar(method_names, values)  # Bu bar qrafikdir, sevimli qrafikimdir
            axes[i].set_title(f'{metric.replace("_", " ").title()}')  # Başlığı gözəlləşdiririk
            axes[i].set_ylabel('Value')  # Y oxunun adı
            if metric == 'compression_ratio' or metric == 'vocabulary_overlap':
                axes[i].set_ylabel('Faiz (%)')  # Faizdirsə belə yazırıq
            
        plt.tight_layout()  # Qrafikləri düzgün yerləşdirir
        return fig, summaries, evaluation


def create_streamlit_app():
    """Create a Streamlit web application for the text summarizer."""
    st.set_page_config(page_title="Advanced Text Summarizer", page_icon="📝", layout="wide")
    
    st.title("📝 Advanced Text Summarizer")
    st.markdown("""
    This application demonstrates multiple text summarization techniques:
    - **Frequency-based**: Extracts sentences with high-frequency important words
    - **TF-IDF**: Uses Term Frequency-Inverse Document Frequency to identify important sentences
    - **TextRank**: Applies a graph-based ranking algorithm (similar to PageRank)
    - **Abstractive**: Generates a new summary using BART transformer model
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Text Summarization", "Compare Methods", "About"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area("Enter the text to summarize:", height=300)
            
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                summarization_method = st.selectbox(
                    "Select summarization method:",
                    ["Frequency-based", "TF-IDF", "TextRank", "Abstractive (BART)"]
                )
            
            with col1b:
                if summarization_method != "Abstractive (BART)":
                    num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
                else:
                    max_length = st.slider("Maximum summary length (tokens):", 50, 250, 150)
                    min_length = st.slider("Minimum summary length (tokens):", 10, 100, 50)
            
            with col1c:
                if st.button("Generate Summary", type="primary"):
                    if text_input:
                        with st.spinner('Generating summary...'):
                            summarizer = TextSummarizer()
                            
                            # Apply selected summarization method
                            if summarization_method == "Frequency-based":
                                summary = summarizer.frequency_based_summary(text_input, num_sentences)
                            elif summarization_method == "TF-IDF":
                                summary = summarizer.tfidf_based_summary(text_input, num_sentences)
                            elif summarization_method == "TextRank":
                                summary = summarizer.textrank_summary(text_input, num_sentences)
                            else:  # Abstractive
                                summary = summarizer.abstractive_summary(text_input, max_length, min_length)
                                
                            # Evaluate summary
                            eval_metrics = summarizer.evaluate_summary(text_input, summary)
                            
                            # Store results in session state
                            st.session_state.summary = summary
                            st.session_state.eval_metrics = eval_metrics
                            st.session_state.original_text = text_input
                    else:
                        st.error("Please enter some text to summarize.")
        
        with col2:
            st.subheader("Summary Output")
            
            if 'summary' in st.session_state:
                st.markdown("### Generated Summary")
                st.write(st.session_state.summary)
                
                st.markdown("### Summary Statistics")
                metrics = st.session_state.eval_metrics
                st.markdown(f"**Original Length:** {metrics['original_length']} words")
                st.markdown(f"**Summary Length:** {metrics['summary_length']} words")
                st.markdown(f"**Compression Ratio:** {metrics['compression_ratio']:.2f}%")
                st.markdown(f"**Vocabulary Overlap:** {metrics['vocabulary_overlap']:.2f}%")
                
                # Option to download summary
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.summary,
                    file_name="text_summary.txt",
                    mime="text/plain"
                )
            else:
                st.info("Your summary will appear here.")
    
    with tab2:
        st.subheader("Compare Summarization Methods")
        
        compare_text = st.text_area("Enter text to compare different summarization methods:", height=200)
        
        if st.button("Compare Methods"):
            if compare_text:
                with st.spinner('Comparing methods...'):
                    summarizer = TextSummarizer()
                    
                    # Get comparison results
                    fig, summaries, evaluations = summarizer.visualize_summary_comparison(compare_text)
                    
                    # Display chart
                    st.pyplot(fig)
                    
                    # Display summaries
                    st.subheader("Summaries by Method")
                    for method, summary in summaries.items():
                        with st.expander(f"{method.title()} Summary"):
                            st.write(summary)
                            metrics = evaluations[method]
                            st.markdown(f"**Compression:** {metrics['compression_ratio']:.2f}% | **Overlap:** {metrics['vocabulary_overlap']:.2f}% | **Length:** {metrics['summary_length']} words")
            else:
                st.error("Please enter some text to compare methods.")
    
    with tab3:
        st.subheader("About This Project")
        st.markdown("""
        ### Advanced Text Summarization Project
        
        This project implements multiple text summarization techniques using a combination of traditional NLP approaches and modern transformer-based models.
        
        #### Features:
        - Multiple summarization algorithms (frequency-based, TF-IDF, TextRank, and abstractive BART)
        - Summary evaluation metrics
        - Visual comparison of different methods
        - User-friendly web interface
        
        #### Technologies Used:
        - Python
        - NLTK for natural language processing
        - scikit-learn for TF-IDF and cosine similarity
        - NetworkX for graph-based algorithms
        - Hugging Face Transformers for BART model
        - Streamlit for the web interface
        - Matplotlib for visualization
        
        #### Potential Applications:
        - Summarizing news articles
        - Creating abstracts for research papers
        - Condensing long documents or reports
        - Generating summaries for content curation
        
        #### Future Improvements:
        - Multi-language support
        - More comprehensive evaluation metrics (ROUGE, BLEU)
        - Fine-tuning the transformer model for specific domains
        - Additional summarization algorithms
        """)


if __name__ == "__main__":
    create_streamlit_app()

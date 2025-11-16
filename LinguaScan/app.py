import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from detection_backend import (
    detect_single_langdetect, detect_single_ngram, detect_batch,
    parse_txt_file, parse_csv_file, get_evaluation_dataset,
    LANGUAGE_CODES, get_ngram_model
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Language Detection App",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Language Detection Application")
st.markdown("""
Advanced NLP-powered language detection with batch processing, file uploads, analytics, and algorithm comparison.
""")

SAMPLE_TEXTS = {
    "English": "Hello! This is a sample text in English. Language detection is an important part of natural language processing.",
    "Spanish": "¡Hola! Este es un texto de ejemplo en español. La detección de idioma es muy útil en aplicaciones globales.",
    "French": "Bonjour! Ceci est un exemple de texte en français. La détection de langue aide à traiter des données multilingues.",
    "German": "Hallo! Dies ist ein Beispieltext auf Deutsch. Spracherkennung ist wichtig für internationale Anwendungen.",
    "Italian": "Ciao! Questo è un testo di esempio in italiano. Il rilevamento della lingua è fondamentale per l'elaborazione del linguaggio.",
    "Portuguese": "Olá! Este é um texto de exemplo em português. A detecção de idioma é essencial para aplicações multilíngues.",
    "Dutch": "Hallo! Dit is een voorbeeldtekst in het Nederlands. Taaldetectie is belangrijk voor NLP-toepassingen.",
    "Russian": "Привет! Это пример текста на русском языке. Определение языка важно для обработки многоязычных данных.",
    "Chinese": "你好！这是一个中文示例文本。语言检测在自然语言处理中非常重要。",
    "Japanese": "こんにちは！これは日本語のサンプルテキストです。言語検出は多言語データの処理に役立ちます。",
    "Arabic": "مرحبا! هذا نص تجريبي باللغة العربية. اكتشاف اللغة مهم لمعالجة البيانات متعددة اللغات.",
    "Hindi": "नमस्ते! यह हिंदी में एक नमूना पाठ है। भाषा का पता लगाना प्राकृतिक भाषा प्रसंस्करण का महत्वपूर्ण हिस्सा है।",
    "Korean": "안녕하세요! 이것은 한국어 샘플 텍스트입니다. 언어 감지는 자연어 처리에서 중요한 부분입니다.",
    "Turkish": "Merhaba! Bu, Türkçe örnek bir metindir. Dil tespiti, doğal dil işleme için önemli bir uygulamadır.",
    "Swedish": "Hej! Detta är en exempeltext på svenska. Språkdetektering är viktigt för flerspråkiga applikationer.",
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Text Detection",
    "📦 Batch Processing",
    "📁 File Upload",
    "📊 Analytics Dashboard",
    "⚖️ Algorithm Comparison"
])

with tab1:
    st.subheader("Single Text Language Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter text to detect language:",
            height=200,
            placeholder="Type or paste your text here...",
            key="single_text"
        )
    
    with col2:
        st.markdown("**Try a sample:**")
        selected_sample = st.selectbox("Choose a sample text:", list(SAMPLE_TEXTS.keys()))
        
        if selected_sample != "Select...":
            if st.button("Load Sample", key="load_sample"):
                st.session_state.single_text_input = SAMPLE_TEXTS[selected_sample]
                st.rerun()
        
        user_input = st.text_area(
            "Enter text to detect language:",
            height=200,
            placeholder="Type or paste your text here...",
            key="single_text_input"
        )        
    
    if user_input and len(user_input.strip()) > 0:
        if user_input == "langdetect":
            result = detect_single_langdetect(user_input)
        else:
            result = detect_single_ngram(user_input)
        
        if result['success']:
            st.success(f"✅ Language Detected: **{result['language_name']}** (Confidence: {result['confidence']*100:.2f}%)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Language", result['language_name'])
            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
            with col3:
                st.metric("Characters", len(user_input))
            with col4:
                st.metric("Detection Time", f"{result['elapsed_time']*1000:.2f}ms")
            
            if len(result['all_probabilities']) > 1:
                st.markdown("---")
                st.subheader("📊 Probability Distribution")
                
                prob_df = pd.DataFrame([
                    {'Language': LANGUAGE_CODES.get(lang, lang.upper()), 'Code': lang, 'Probability': prob}
                    for lang, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = sns.color_palette("viridis", len(prob_df))
                    bars = ax.barh(prob_df['Language'][:10], prob_df['Probability'][:10], color=colors)
                    ax.set_xlabel('Probability', fontsize=11)
                    ax.set_ylabel('Language', fontsize=11)
                    ax.set_xlim(0, 1)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    for bar, prob in zip(bars, prob_df['Probability'][:10]):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{prob*100:.1f}%', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    display_df = prob_df.copy()
                    display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(display_df[['Language', 'Code', 'Probability']], height=400, use_container_width=True)
        else:
            st.error(f"⚠️ Detection failed: {result['error']}")
    else:
        st.info("👆 Enter text above to detect its language")

with tab2:
    st.subheader("Batch Text Processing")
    st.markdown("Analyze multiple texts at once. Enter one text per line.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=300,
            placeholder="Line 1: First text...\nLine 2: Second text...\nLine 3: Third text...",
            key="batch_input"
        )
    
    with col2:
        batch_algo = st.radio(
            "Algorithm:",
            ["langdetect", "n-gram"],
            key="batch_algo"
        )
        
        if st.button("Load Sample Batch", key="load_batch_sample"):
            sample_batch = "\n".join([
                "Hello, this is English text.",
                "Bonjour, ceci est un texte français.",
                "Hola, este es un texto en español.",
                "Guten Tag, das ist deutscher Text.",
                "Ciao, questo è un testo italiano."
            ])
            st.session_state.batch_input = sample_batch
            st.rerun()
    
    if batch_input and len(batch_input.strip()) > 0:
        texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
        
        if len(texts) > 0:
            with st.spinner(f"Processing {len(texts)} texts..."):
                batch_df = detect_batch(texts, algorithm=batch_algo)
            
            st.success(f"✅ Processed {len(texts)} texts!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Texts", len(batch_df))
            with col2:
                successful = batch_df['success'].sum()
                st.metric("Successful", successful)
            with col3:
                avg_confidence = batch_df[batch_df['success']]['confidence'].mean() * 100
                st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
            
            st.markdown("---")
            st.subheader("Results Table")
            
            display_df = batch_df[['index', 'text', 'language_name', 'confidence', 'char_count', 'word_count']].copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
            display_df.columns = ['#', 'Text', 'Language', 'Confidence', 'Chars', 'Words']
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            st.session_state.batch_results = batch_df
        else:
            st.warning("Please enter at least one text to process")
    else:
        st.info("👆 Enter multiple texts above (one per line) to process them in batch")

with tab3:
    st.subheader("File Upload Detection")
    st.markdown("Upload a TXT or CSV file to detect languages for all entries.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file (TXT or CSV):",
            type=['txt', 'csv'],
            key="file_upload"
        )
    
    with col2:
        file_algo = st.radio(
            "Algorithm:",
            ["langdetect", "n-gram"],
            key="file_algo"
        )
        
        csv_column = None
        if uploaded_file and uploaded_file.name.endswith('.csv'):
            csv_column = st.text_input(
                "CSV column name (leave empty for first column):",
                key="csv_column"
            )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.txt'):
                texts = parse_txt_file(uploaded_file)
            else:
                texts = parse_csv_file(uploaded_file, csv_column if csv_column else None)
            
            st.info(f"📄 Loaded {len(texts)} texts from {uploaded_file.name}")
            
            if st.button("Analyze File", key="analyze_file"):
                with st.spinner(f"Processing {len(texts)} texts from file..."):
                    file_df = detect_batch(texts, algorithm=file_algo)
                
                st.success(f"✅ Analyzed {len(texts)} texts!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Texts", len(file_df))
                with col2:
                    successful = file_df['success'].sum()
                    st.metric("Successful", successful)
                with col3:
                    avg_confidence = file_df[file_df['success']]['confidence'].mean() * 100
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                
                st.markdown("---")
                st.subheader("Detection Results")
                
                display_df = file_df[['index', 'text', 'language_name', 'confidence', 'char_count', 'word_count']].copy()
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                display_df.columns = ['#', 'Text', 'Language', 'Confidence', 'Chars', 'Words']
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                st.session_state.file_results = file_df
                
                csv_data = file_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv_data,
                    file_name="language_detection_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"⚠️ Error processing file: {str(e)}")
    else:
        st.info("👆 Upload a TXT or CSV file to begin analysis")

with tab4:
    st.subheader("📊 Analytics Dashboard")
    st.markdown("View language distribution and statistics from your batch/file analysis.")
    
    if 'batch_results' in st.session_state or 'file_results' in st.session_state:
        data_source = st.radio(
            "Select data source:",
            ["Batch Processing", "File Upload"],
            key="analytics_source"
        )
        
        if data_source == "Batch Processing" and 'batch_results' in st.session_state:
            analytics_df = st.session_state.batch_results
        elif data_source == "File Upload" and 'file_results' in st.session_state:
            analytics_df = st.session_state.file_results
        else:
            st.warning(f"No data available for {data_source}. Please process some texts first.")
            analytics_df = None
        
        if analytics_df is not None and len(analytics_df) > 0:
            successful_df = analytics_df[analytics_df['success']].copy()
            
            if len(successful_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Analyzed", len(analytics_df))
                with col2:
                    st.metric("Success Rate", f"{len(successful_df)/len(analytics_df)*100:.1f}%")
                
                st.markdown("---")
                st.subheader("Language Distribution")
                
                lang_counts = successful_df['language_name'].value_counts()
                lang_stats = successful_df.groupby('language_name').agg({
                    'confidence': ['mean', 'min', 'max'],
                    'char_count': 'mean',
                    'word_count': 'mean'
                }).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Language Counts")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = sns.color_palette("husl", len(lang_counts))
                    bars = ax.barh(lang_counts.index, lang_counts.values, color=colors)
                    ax.set_xlabel('Count', fontsize=11)
                    ax.set_ylabel('Language', fontsize=11)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    for bar, count in zip(bars, lang_counts.values):
                        width = bar.get_width()
                        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                               f'{count}', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("##### Language Percentage")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors_pie = sns.color_palette("Set2", len(lang_counts))
                    wedges, texts, autotexts = ax.pie(
                        lang_counts.values,
                        labels=lang_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors_pie
                    )
                    
                    for text in texts:
                        text.set_fontsize(9)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(9)
                        autotext.set_weight('bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                st.markdown("---")
                st.subheader("Confidence Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Confidence Distribution by Language")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    for lang in successful_df['language_name'].unique():
                        lang_data = successful_df[successful_df['language_name'] == lang]['confidence']
                        ax.hist(lang_data, alpha=0.6, label=lang, bins=10)
                    
                    ax.set_xlabel('Confidence', fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)
                    ax.legend()
                    ax.grid(alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("##### Statistics by Language")
                    stats_display = pd.DataFrame({
                        'Language': lang_counts.index,
                        'Count': lang_counts.values,
                        'Avg Confidence': [f"{successful_df[successful_df['language_name']==lang]['confidence'].mean()*100:.1f}%" for lang in lang_counts.index],
                        'Avg Length': [f"{successful_df[successful_df['language_name']==lang]['char_count'].mean():.0f}" for lang in lang_counts.index]
                    })
                    st.dataframe(stats_display, use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("Text Length Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(successful_df['char_count'], successful_df['confidence'], alpha=0.6, c=sns.color_palette("viridis", len(successful_df)))
                    ax.set_xlabel('Character Count', fontsize=11)
                    ax.set_ylabel('Confidence', fontsize=11)
                    ax.set_title('Confidence vs Text Length')
                    ax.grid(alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(successful_df['char_count'], bins=20, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Character Count', fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)
                    ax.set_title('Text Length Distribution')
                    ax.grid(alpha=0.3, linestyle='--', axis='y')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.warning("No successful detections to analyze")
    else:
        st.info("👆 Process some texts in the Batch Processing or File Upload tabs first to see analytics")

with tab5:
    st.subheader("⚖️ Algorithm Comparison & Model Performance")
    st.markdown("Compare langdetect vs character n-gram classifier performance.")
    
    comparison_mode = st.radio(
        "Comparison Mode:",
        ["Single Text Comparison", "Model Evaluation (Benchmark Dataset)"],
        key="comparison_mode"
    )
    
    if comparison_mode == "Single Text Comparison":
        st.markdown("### Compare Both Algorithms on Your Text")
        
        compare_text = st.text_area(
            "Enter text to compare:",
            height=150,
            placeholder="Enter text to see how both algorithms perform...",
            key="compare_text"
        )
        
        if compare_text and len(compare_text.strip()) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 langdetect Algorithm")
                result_ld = detect_single_langdetect(compare_text)
                
                if result_ld['success']:
                    st.success(f"**Language:** {result_ld['language_name']}")
                    st.metric("Confidence", f"{result_ld['confidence']*100:.2f}%")
                    st.metric("Detection Time", f"{result_ld['elapsed_time']*1000:.2f}ms")
                else:
                    st.error(f"Failed: {result_ld['error']}")
            
            with col2:
                st.markdown("#### 🟢 N-gram Algorithm")
                result_ng = detect_single_ngram(compare_text)
                
                if result_ng['success']:
                    st.success(f"**Language:** {result_ng['language_name']}")
                    st.metric("Confidence", f"{result_ng['confidence']*100:.2f}%")
                    st.metric("Detection Time", f"{result_ng['elapsed_time']*1000:.2f}ms")
                else:
                    st.error(f"Failed: {result_ng['error']}")
            
            if result_ld['success'] and result_ng['success']:
                st.markdown("---")
                st.subheader("Comparison Summary")
                
                comparison_df = pd.DataFrame({
                    'Algorithm': ['langdetect', 'N-gram'],
                    'Detected Language': [result_ld['language_name'], result_ng['language_name']],
                    'Confidence': [f"{result_ld['confidence']*100:.2f}%", f"{result_ng['confidence']*100:.2f}%"],
                    'Detection Time (ms)': [f"{result_ld['elapsed_time']*1000:.2f}", f"{result_ng['elapsed_time']*1000:.2f}"],
                    'Agreement': ['✅' if result_ld['language'] == result_ng['language'] else '❌'] * 2
                })
                
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Enter text above to compare both algorithms")
    
    else:
        st.markdown("### Model Performance Evaluation")
        st.markdown("Evaluate both models on a benchmark dataset with known languages.")
        
        if st.button("Run Evaluation", key="run_eval"):
            eval_data = get_evaluation_dataset()
            texts, true_labels = zip(*eval_data)
            
            with st.spinner("Evaluating both models on benchmark dataset..."):
                langdetect_predictions = []
                langdetect_confidences = []
                langdetect_times = []
                
                for text in texts:
                    result = detect_single_langdetect(text)
                    langdetect_predictions.append(result['language'] if result['success'] else 'unknown')
                    langdetect_confidences.append(result['confidence'])
                    langdetect_times.append(result['elapsed_time'])
                
                ngram_predictions = []
                ngram_confidences = []
                ngram_times = []
                
                for text in texts:
                    result = detect_single_ngram(text)
                    ngram_predictions.append(result['language'] if result['success'] else 'unknown')
                    ngram_confidences.append(result['confidence'])
                    ngram_times.append(result['elapsed_time'])
            
            st.success("✅ Evaluation Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 langdetect Performance")
                ld_accuracy = accuracy_score(true_labels, langdetect_predictions)
                st.metric("Accuracy", f"{ld_accuracy*100:.2f}%")
                st.metric("Avg Confidence", f"{np.mean(langdetect_confidences)*100:.2f}%")
                st.metric("Avg Time", f"{np.mean(langdetect_times)*1000:.2f}ms")
            
            with col2:
                st.markdown("#### 🟢 N-gram Performance")
                ng_accuracy = accuracy_score(true_labels, ngram_predictions)
                st.metric("Accuracy", f"{ng_accuracy*100:.2f}%")
                st.metric("Avg Confidence", f"{np.mean(ngram_confidences)*100:.2f}%")
                st.metric("Avg Time", f"{np.mean(ngram_times)*1000:.2f}ms")
            
            st.markdown("---")
            st.subheader("Performance Comparison")
            
            perf_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Avg Confidence', 'Avg Detection Time (ms)', 'Total Predictions'],
                'langdetect': [
                    f"{ld_accuracy*100:.2f}%",
                    f"{np.mean(langdetect_confidences)*100:.2f}%",
                    f"{np.mean(langdetect_times)*1000:.2f}",
                    len(langdetect_predictions)
                ],
                'N-gram': [
                    f"{ng_accuracy*100:.2f}%",
                    f"{np.mean(ngram_confidences)*100:.2f}%",
                    f"{np.mean(ngram_times)*1000:.2f}",
                    len(ngram_predictions)
                ]
            })
            
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("Detailed Results")
            
            results_df = pd.DataFrame({
                'Text': [t[:80] + '...' if len(t) > 80 else t for t in texts],
                'True Language': [LANGUAGE_CODES.get(l, l) for l in true_labels],
                'langdetect': [LANGUAGE_CODES.get(l, l) for l in langdetect_predictions],
                'N-gram': [LANGUAGE_CODES.get(l, l) for l in ngram_predictions],
                'LD Confidence': [f"{c*100:.1f}%" for c in langdetect_confidences],
                'NG Confidence': [f"{c*100:.1f}%" for c in ngram_confidences],
            })
            
            st.dataframe(results_df, use_container_width=True, height=400)
        else:
            st.info("👆 Click 'Run Evaluation' to test both models on the benchmark dataset")

st.markdown("---")
st.markdown(""" """)

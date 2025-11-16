import streamlit as st
import pandas as pd
from langdetect import detect, detect_langs, LangDetectException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import time
import io

LANGUAGE_CODES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'nl': 'Dutch', 'ru': 'Russian', 'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)', 'ja': 'Japanese', 'ar': 'Arabic', 'hi': 'Hindi',
    'ko': 'Korean', 'tr': 'Turkish', 'sv': 'Swedish', 'pl': 'Polish', 'da': 'Danish',
    'fi': 'Finnish', 'no': 'Norwegian', 'cs': 'Czech', 'ro': 'Romanian', 'el': 'Greek',
    'he': 'Hebrew', 'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'uk': 'Ukrainian',
    'ca': 'Catalan', 'sk': 'Slovak', 'hr': 'Croatian', 'bg': 'Bulgarian', 'lt': 'Lithuanian',
    'sl': 'Slovenian', 'et': 'Estonian', 'lv': 'Latvian', 'af': 'Afrikaans', 'sq': 'Albanian',
    'hu': 'Hungarian', 'tl': 'Tagalog', 'sw': 'Swahili', 'cy': 'Welsh', 'ur': 'Urdu',
    'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi', 'fa': 'Persian',
    'ml': 'Malayalam', 'kn': 'Kannada', 'gu': 'Gujarati', 'pa': 'Punjabi'
}

@st.cache_resource
def get_ngram_model():
    training_data = [
        ("Hello, how are you doing today?", "en"),
        ("Machine learning is fascinating.", "en"),
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("Artificial intelligence will transform the world.", "en"),
        ("Natural language processing enables computers to understand human language.", "en"),
        ("¿Cómo estás? Espero que tengas un buen día.", "es"),
        ("El aprendizaje automático es el futuro de la tecnología.", "es"),
        ("La inteligencia artificial está cambiando el mundo.", "es"),
        ("Me gusta mucho la música y el arte.", "es"),
        ("España es un país hermoso con mucha historia.", "es"),
        ("Bonjour, comment allez-vous aujourd'hui?", "fr"),
        ("L'intelligence artificielle est très intéressante.", "fr"),
        ("J'aime beaucoup la langue française.", "fr"),
        ("Paris est la capitale de la France.", "fr"),
        ("Le machine learning transforme notre monde.", "fr"),
        ("Guten Tag, wie geht es Ihnen?", "de"),
        ("Künstliche Intelligenz ist sehr wichtig.", "de"),
        ("Ich lerne gerne neue Sprachen.", "de"),
        ("Deutschland hat eine reiche Geschichte.", "de"),
        ("Das maschinelle Lernen entwickelt sich schnell.", "de"),
        ("Ciao, come stai oggi?", "it"),
        ("L'intelligenza artificiale è molto utile.", "it"),
        ("Mi piace molto la cucina italiana.", "it"),
        ("Roma è una città bellissima.", "it"),
        ("Il machine learning è il futuro.", "it"),
        ("Olá, como você está hoje?", "pt"),
        ("A inteligência artificial é fascinante.", "pt"),
        ("Eu gosto muito de aprender idiomas.", "pt"),
        ("O Brasil é um país grande e diverso.", "pt"),
        ("Portugal tem uma história rica.", "pt"),
        ("Привет, как дела?", "ru"),
        ("Искусственный интеллект очень интересен.", "ru"),
        ("Я люблю изучать языки.", "ru"),
        ("Россия большая страна.", "ru"),
        ("Машинное обучение развивается быстро.", "ru"),
        ("你好，你今天怎么样？", "zh-cn"),
        ("机器学习非常有趣。", "zh-cn"),
        ("我喜欢学习新语言。", "zh-cn"),
        ("中国有悠久的历史。", "zh-cn"),
        ("人工智能改变世界。", "zh-cn"),
        ("こんにちは、元気ですか？", "ja"),
        ("機械学習はとても面白いです。", "ja"),
        ("日本語を学ぶのが好きです。", "ja"),
        ("東京は大きな都市です。", "ja"),
        ("人工知能は重要です。", "ja"),
        ("مرحبا، كيف حالك اليوم؟", "ar"),
        ("الذكاء الاصطناعي مثير للاهتمام.", "ar"),
        ("أحب تعلم اللغات الجديدة.", "ar"),
        ("العربية لغة جميلة.", "ar"),
        ("التعلم الآلي يتطور بسرعة.", "ar"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("कृत्रिम बुद्धिमत्ता बहुत रोचक है।", "hi"),
        ("मुझे नई भाषाएँ सीखना पसंद है।", "hi"),
        ("भारत एक विविध देश है।", "hi"),
        ("मशीन लर्निंग तेजी से विकसित हो रही है।", "hi"),
    ]
    
    texts, labels = zip(*training_data)
    
    model = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=1000)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    model.fit(texts, labels)
    return model

def detect_single_langdetect(text):
    try:
        start_time = time.time()
        detected_lang = detect(text)
        lang_probs = detect_langs(text)
        elapsed_time = time.time() - start_time
        
        probabilities = {prob.lang: prob.prob for prob in lang_probs}
        main_prob = probabilities.get(detected_lang, 0.0)
        
        return {
            'success': True,
            'language': detected_lang,
            'language_name': LANGUAGE_CODES.get(detected_lang, detected_lang.upper()),
            'confidence': main_prob,
            'all_probabilities': probabilities,
            'elapsed_time': elapsed_time,
            'error': None
        }
    except LangDetectException as e:
        return {
            'success': False,
            'language': None,
            'language_name': 'Unknown',
            'confidence': 0.0,
            'all_probabilities': {},
            'elapsed_time': 0.0,
            'error': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'language': None,
            'language_name': 'Unknown',
            'confidence': 0.0,
            'all_probabilities': {},
            'elapsed_time': 0.0,
            'error': str(e)
        }

def detect_single_ngram(text):
    try:
        model = get_ngram_model()
        start_time = time.time()
        
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_
        
        elapsed_time = time.time() - start_time
        
        prob_dict = {lang: float(prob) for lang, prob in zip(classes, probabilities)}
        main_prob = prob_dict.get(prediction, 0.0)
        
        return {
            'success': True,
            'language': prediction,
            'language_name': LANGUAGE_CODES.get(prediction, prediction.upper()),
            'confidence': main_prob,
            'all_probabilities': prob_dict,
            'elapsed_time': elapsed_time,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'language': None,
            'language_name': 'Unknown',
            'confidence': 0.0,
            'all_probabilities': {},
            'elapsed_time': 0.0,
            'error': str(e)
        }

@st.cache_data
def detect_batch(texts, algorithm='langdetect'):
    results = []
    
    for idx, text in enumerate(texts):
        if not text or len(str(text).strip()) == 0:
            results.append({
                'index': idx,
                'text': text,
                'char_count': 0,
                'word_count': 0,
                'language': None,
                'language_name': 'Empty',
                'confidence': 0.0,
                'success': False,
                'error': 'Empty text'
            })
            continue
        
        text_str = str(text).strip()
        
        if algorithm == 'langdetect':
            detection = detect_single_langdetect(text_str)
        else:
            detection = detect_single_ngram(text_str)
        
        words = text_str.split()
        
        results.append({
            'index': idx,
            'text': text_str[:100] + ('...' if len(text_str) > 100 else ''),
            'full_text': text_str,
            'char_count': len(text_str),
            'word_count': len(words),
            'language': detection['language'],
            'language_name': detection['language_name'],
            'confidence': detection['confidence'],
            'success': detection['success'],
            'error': detection['error']
        })
    
    return pd.DataFrame(results)

def parse_txt_file(uploaded_file):
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('latin-1')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            return lines
        except Exception as e:
            raise Exception(f"Unable to decode file: {str(e)}")

def parse_csv_file(uploaded_file, text_column=None):
    try:
        df = pd.read_csv(uploaded_file)
        
        if text_column and text_column in df.columns:
            texts = df[text_column].dropna().astype(str).tolist()
        else:
            texts = df.iloc[:, 0].dropna().astype(str).tolist()
        
        return texts
    except Exception as e:
        raise Exception(f"Unable to parse CSV: {str(e)}")

def get_evaluation_dataset():
    eval_data = [
        ("The weather is beautiful today, perfect for a walk in the park.", "en"),
        ("Data science combines statistics, programming, and domain knowledge.", "en"),
        ("Climate change is one of the biggest challenges facing humanity.", "en"),
        ("El cambio climático es uno de los mayores desafíos de la humanidad.", "es"),
        ("La ciencia de datos combina estadística, programación y conocimiento del dominio.", "es"),
        ("Barcelona es una ciudad hermosa en la costa mediterránea.", "es"),
        ("Le changement climatique est l'un des plus grands défis de l'humanité.", "fr"),
        ("La science des données combine statistiques, programmation et connaissances du domaine.", "fr"),
        ("La Tour Eiffel est le monument le plus visité au monde.", "fr"),
        ("Der Klimawandel ist eine der größten Herausforderungen der Menschheit.", "de"),
        ("Datenwissenschaft kombiniert Statistik, Programmierung und Fachwissen.", "de"),
        ("Berlin ist die Hauptstadt von Deutschland.", "de"),
        ("Il cambiamento climatico è una delle sfide più grandi dell'umanità.", "it"),
        ("La scienza dei dati combina statistica, programmazione e conoscenza del dominio.", "it"),
        ("Venezia è una città unica costruita sull'acqua.", "it"),
        ("A mudança climática é um dos maiores desafios da humanidade.", "pt"),
        ("A ciência de dados combina estatística, programação e conhecimento de domínio.", "pt"),
        ("Lisboa é a capital de Portugal.", "pt"),
        ("Изменение климата является одной из самых больших проблем человечества.", "ru"),
        ("Наука о данных сочетает статистику, программирование и знание предметной области.", "ru"),
        ("Москва - столица России.", "ru"),
        ("气候变化是人类面临的最大挑战之一。", "zh-cn"),
        ("数据科学结合了统计学、编程和领域知识。", "zh-cn"),
        ("北京是中国的首都。", "zh-cn"),
        ("気候変動は人類が直面する最大の課題の一つです。", "ja"),
        ("データサイエンスは統計、プログラミング、ドメイン知識を組み合わせています。", "ja"),
        ("東京は日本の首都です。", "ja"),
        ("تغير المناخ هو أحد أكبر التحديات التي تواجه البشرية.", "ar"),
        ("علم البيانات يجمع بين الإحصاء والبرمجة ومعرفة المجال.", "ar"),
        ("القاهرة هي عاصمة مصر.", "ar"),
    ]
    
    return eval_data

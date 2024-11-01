import pandas as pd, re, nlp_utils, py_vncorenlp, os, joblib
from sklearn.feature_extraction.text import TfidfVec
from sklearn.decomposition import TruncatedSVD

# Xóa HTML code
def remove_html(txt: str) -> str:
    return re.sub(r'<[^>]*>', '', txt)

# Chuẩn hóa kiểu gõ dấu cũ và mới, đưa văn bản về viết thường
def standardize_vietnamese_typing(txt: str) -> str:
    return nlp_utils.chuan_hoa_dau_cau_tieng_viet(txt)

# Tách từ tiếng Việt
wseg_dir = os.getcwd()
py_vncorenlp.download_model(save_dir=wseg_dir)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=wseg_dir)
def word_segment_to_str(txt: str) -> str:
    txt_list = rdrsegmenter.word_segment(txt)
    txt = ''.join(txt_list)
    return txt

# Loại bỏ số, dấu space thừa, dấu câu và các ký tự đặc biệt
def remove_num_punctuation(txt: str) -> str:
    txt = re.sub(r'[^\w\s]', '', txt)
    txt = re.sub(r'\d+', '', txt)
    txt = re.sub(r' +', ' ', txt)
    return txt

# Loại bỏ stopwords
with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# Gọi hàm này để tiền xử lý cho 1 dữ liệu predict
def preprocess_for_predict(txt: str):
    txt = remove_html(txt)
    txt = standardize_vietnamese_typing(txt)
    txt = word_segment_to_str(txt)
    txt = remove_num_punctuation(txt)
    txt = remove_stopwords(txt)
    vectorizer = joblib.load('tfidf_vectorizer.joblibe')
    txt_vector = vectorizer.fit_transform([txt])
    svd = TruncatedSVD(n_components=300)
    txt_vector_reduced = svd.fit_transform(txt_vector)
    return txt_vector_reduced
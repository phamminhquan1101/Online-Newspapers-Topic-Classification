import pandas as pd, re, nlp_utils, py_vncorenlp, os, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Xem qua dữ liệu
df = pd.read_csv('./Data/VnExpressArticles.csv')
print('Dữ liệu trước khi tiền xử lý:')
print(df)
# Kiểm tra xem dữ liệu có null không
print(df.info())
# Kiểm tra unique
print('Số nội dung độc nhất:', df['content'].nunique())
print('Số chủ đề độc nhất:', df['topic'].nunique())
# --> Dữ liệu không có bất thường, có thể bắt đầu tiền xử lý ngôn ngữ tiếng Việt

# Xóa HTML code
def remove_html(txt: str) -> str:
    return re.sub(r'<[^>]*>', '', txt)
df['content'] = df['content'].apply(remove_html)
print('Dữ liệu sau khi xóa HTML code:')
print(df['content'])

# Chuẩn hóa kiểu gõ dấu cũ và mới, đưa văn bản về viết thường
def standardize_vietnamese_typing(txt: str) -> str:
    return nlp_utils.chuan_hoa_dau_cau_tieng_viet(txt)
df['content'] = df['content'].apply(standardize_vietnamese_typing)
print('Dữ liệu sau khi chuẩn hóa kiểu gõ dấu:')
print(df['content'])

# Tách từ tiếng Việt
wseg_dir = os.getcwd()
py_vncorenlp.download_model(save_dir=wseg_dir)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=wseg_dir)
def word_segment_to_str(txt: str) -> str:
    txt_list = rdrsegmenter.word_segment(txt)
    txt = ''.join(txt_list)
    return txt
df['content'] = df['content'].apply(word_segment_to_str)
print('Dữ liệu sau khi tách từ tiếng Việt:')
print(df['content'])

# Loại bỏ số, dấu space thừa, dấu câu và các ký tự đặc biệt
def remove_num_punctuation(txt: str) -> str:
    txt = re.sub(r'[^\w\s]', '', txt)
    txt = re.sub(r'\d+', '', txt)
    txt = re.sub(r' +', ' ', txt)
    return txt

df['content'] = df['content'].apply(remove_num_punctuation)
print('Dữ liệu sau khi bỏ dấu câu và các ký tự đặc biệt:')
print(df['content'])

# Loại bỏ stopwords
with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

df['content'] = df['content'].apply(remove_stopwords)
print('Dữ liệu sau khi loại bỏ stopwords:')
print(df['content'])

# Lưu dữ liệu tiền xử lý
df.to_csv('./Data/VnExpressArticlesPreprocessing.csv')
print('Lưu dữ liệu sau khi tiền xử lý thành công!')

# vector hóa dữ liệu
vectorizer = TfidfVectorizer(min_df=5)
matrix = vectorizer.fit_transform(df['content'])

print('Các đặc trưng thu được:')
print(vectorizer.get_feature_names_out())
print('Kích thước của dữ liệu sau vector hóa:')
print(matrix.shape)
print('Dữ liệu được vector hóa:')
print(matrix.toarray())

# Trích chọn đặc trưng
svd = TruncatedSVD(n_components=300)
matrix_reduced = svd.fit_transform(matrix)
print('Kích thước của dữ liệu sau khi trích chọn đặc trưng:')
print(matrix_reduced.shape)
print('Dữ liệu sau khi trích chọn đặc trưng:')
print(matrix_reduced)

# Lưu lại dữ liệu và vectorizer để sử dụng sau này
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print('Lưu TF-IDF Vectorizer thành công!')
df_tfidf = pd.DataFrame(matrix_reduced)
df_tfidf['topic'] = df['topic']
joblib.dump(df_tfidf, './Data/df_tfidf.joblib')
print('Lưu dữ liệu sau khi vector hóa thành công!')

# # Load dữ liệu lên dataframe
# df_load = joblib.load('./Data/df_tfidf.joblib')
# print(df_load)
# print('Load dữ liệu thành công!')
from flask import (
    flash, render_template, request, Flask
)
import requests, bs4
from keras.models import load_model
import numpy as np, joblib
import PreprocessingForPredict as pre

app = Flask(__name__)
app.secret_key = "super secret key"

@app.route("/", methods=('GET', 'POST'))
def index():
    data = ""
    predict = ""
    if request.method == 'POST':
        if request.form['button'] == 'Lấy dữ liệu':
            try:
                article_res = requests.get(request.form['url'])
                article_res.raise_for_status()
            except:
                flash('Không thể gửi yêu cầu tới URL. Kiểm tra lại URL của bạn (yêu cầu URL đầy đủ tới trang) hoặc kiểm tra lại kết nối mạng.')
                return render_template('index.html', data=data, predict=predict)

            try:
                article_soup = bs4.BeautifulSoup(article_res.text, 'lxml')
                title_elems = article_soup.select('h1.title-detail')
                content_elems = article_soup.select('p.Normal')

                data = title_elems[0].text.strip() + '.'
                for ip in range(len(content_elems)-1):
                    data += (' \n' + content_elems[ip].text.strip())
                
                check_author_str = content_elems[-1].text.strip()
                check_author_strs = check_author_str.split()
                is_author_str = True
                for s in check_author_strs:
                    if s[0].islower():
                        is_author_str = False
                        break
                if is_author_str == False: data += (' \n' + content_elems[-1].text)
            except:
                flash('Bài báo trong URL của bạn không hợp lệ. Bài báo của bạn có thể có cấu trúc đặc biệt (poster, bài quảng cáo, video, ...). Bạn cũng có thể kiểm tra xem bài báo của bạn đã đúng là đến từ VnExpress hay chưa.')
                return render_template('index.html', data="", predict=predict)

        if request.form['button'] == 'Dự đoán':
            if request.form['content'].strip() == "":
                flash('Bạn cần điền url và lấy dữ liệu thành công từ trang web trước khi thực hiện dự đoán.')
                return render_template('index.html', data=data, predict=predict)
            else:
                label_encoder = joblib.load('label_encoder.joblib')
                model = load_model('model_output.h5')
                text = pre.preprocess_for_predict(request.form['content'].strip())
                predict_index = np.argmax(model.predict(text))
                predict = label_encoder.inverse_transform([predict_index])
                return render_template('index.html', data=request.form['content'].strip(), predict=predict[0])
                

    return render_template('index.html', data=data, predict=predict)

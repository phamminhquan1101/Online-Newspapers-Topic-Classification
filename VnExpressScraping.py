'''
Script này sẽ rút trích nội dung các bài báo trong trang web: vnexpress.net
Các bài báo được rút trích sẽ tương ứng với 6 chủ đề: Kinh doanh, Khoa học, Giải trí, Pháp luật, Sức khỏe, Du lịch
Mỗi chủ đề, sẽ có khoảng 600 bài báo được trích xuất (Thay api từ scraperapi của riêng bạn vào API_KEY)
Nội dung trích xuất của mỗi bài báo bao gồm: Nội dung (content), Chủ đề (topic)
Toàn bộ nội dung sau khi trích xuất sẽ được lưu theo đường dẫn: ./Data/VnExpressArticles.csv
'''
import threading, pandas as pd, requests, bs4, re

URL = 'https://vnexpress.net'
NO_ARTICLES_PER_TOPIC = 1000
API_KEY = '4356dfe15bc06407bc35df088b1a4224'

# Chuyển tên chủ đề thành đuôi url kết nối tới chủ đề đó trên trang web
def topic_to_href(topic: str) -> str:
    rs = topic.lower()

    rs = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', rs)
    rs = re.sub('[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'A', rs)
    rs = re.sub('[éèẻẽẹêếềểễệ]', 'e', rs)
    rs = re.sub('[ÉÈẺẼẸÊẾỀỂỄỆ]', 'E', rs)
    rs = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', rs)
    rs = re.sub('[ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]', 'O', rs)
    rs = re.sub('[íìỉĩị]', 'i', rs)
    rs = re.sub('[ÍÌỈĨỊ]', 'I', rs)
    rs = re.sub('[úùủũụưứừửữự]', 'u', rs)
    rs = re.sub('[ÚÙỦŨỤƯỨỪỬỮỰ]', 'U', rs)
    rs = re.sub('[ýỳỷỹỵ]', 'y', rs)
    rs = re.sub('[ÝỲỶỸỴ]', 'Y', rs)
    rs = re.sub('đ', 'd', rs)
    rs = re.sub('Đ', 'D', rs)

    rsList = rs.split()
    rs = '-'.join(rsList)

    return '/' + rs

# Tải các bài báo theo 1 chủ đề và lưu lại thành 1 file csv cùng tên với chủ đề trong thư mục Data
def download_articles_to_dataframe(article_num: int, href: str, topic: str) -> None:
    payload1 = { 'api_key': API_KEY, 'url': (URL + href) }
    res = requests.get('https://api.scraperapi.com/', params=payload1)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, 'lxml')

    results = []
    i = 1
    page = 1
    count = 0
    while count != article_num:
        elems = soup.select(f'a[data-medium="Item-{i}"]')

        if len(elems) > 0:
            article_href = elems[0].get('href')
            i += 1

            if 'video.vnexpress.net' in article_href: continue
            payload2 = { 'api_key': API_KEY, 'url': article_href }
            article_res = requests.get('https://api.scraperapi.com/', params=payload2)
            try:
                article_res.raise_for_status()
                article_soup = bs4.BeautifulSoup(article_res.text, 'lxml')
                title_elems = article_soup.select('h1.title-detail')
                content_elems = article_soup.select('p.Normal')

                result = title_elems[0].text.strip() + '.'
                for ip in range(len(content_elems)-1):
                    result += (' ' + content_elems[ip].text.strip())
                
                check_author_str = content_elems[-1].text.strip()
                check_author_strs = check_author_str.split()
                is_author_str = True
                for s in check_author_strs:
                    if s[0].islower():
                        is_author_str = False
                        break
                if is_author_str == False: result += (' ' + content_elems[-1].text)

                results.append(result)
                count += 1
            except:
                continue
        elif len(soup.select(f'a[data-medium="Item-{i+1}"]')) > 0:
            i += 1
            continue
        else:
            try:
                page += 1
                payload3 = { 'api_key': API_KEY, 'url': (URL + href + '-p' + str(page)) }
                res = requests.get('https://api.scraperapi.com/', params=payload3)
                res.raise_for_status()
                soup = bs4.BeautifulSoup(res.text, 'lxml')
            except:
                break

    print(count, f'articles in "{topic}" topic have just been downloaded')
    data = {
        'content' : [],
        'topic' : []
    }

    data['content'].extend(results)
    data['topic'] = [topic for _ in range(count)]

    df = pd.DataFrame(data)
    df.to_csv(f'./Data/{topic}.csv', encoding='utf-8', index=False)

topics = ['Kinh doanh', 'Khoa học', 'Giải trí', 'Pháp luật', 'Sức khỏe', 'Du lịch']
href_topics = [topic_to_href(t) for t in topics]

# Tải không dùng thread
# for ih, h in enumerate(href_topics):
#     download_articles_to_dataframe(NO_ARTICLES_PER_TOPIC, h, topics[ih])

# Chia thành 6 threads tải cùng lúc
downloadThreads = []
for ih, h in enumerate(href_topics):
    downloadThread = threading.Thread(target=download_articles_to_dataframe, args=(NO_ARTICLES_PER_TOPIC, h, topics[ih]))
    downloadThreads.append(downloadThread)
    downloadThread.start()

for downloadThread in downloadThreads:
    downloadThread.join()

# Gộp dữ liệu thành 1 file csv
myDatas = []
for t in topics:
    df = pd.read_csv(f'./Data/{t}.csv')
    myDatas.append(df)
VnExpressArticles_df = pd.concat(myDatas, axis=0)
VnExpressArticles_df.to_csv('./Data/VnExpressArticles.csv', encoding='utf-8', index=False)
print(VnExpressArticles_df['content'].nunique(), 'unique articles.')
print('Done.')
from flask import Flask, request, render_template
from bs4 import BeautifulSoup
import requests
from transformers import pipeline

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)

# Özetleme modelini yüklüyoruz
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# URL'den sayfa içeriğini almak için fonksiyon
def get_page_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"URL'ye erişim hatası: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"İstek hatası: {e}")
        return None

# Sayfa içeriğini özetlemek için ana işlev
def summarize_page_content(content):
    soup = BeautifulSoup(content, 'html.parser')
    article_content = soup.find('article') or soup.find('main')

    if article_content:
        paragraphs = article_content.find_all('p')
    else:
        paragraphs = soup.find_all('p')

    # Paragrafları birleştirip tek bir string yapıyoruz
    text_content = ' '.join([para.get_text() for para in paragraphs])

    # Eğer içerik çok kısa veya boşsa, kullanıcıya bir hata mesajı döndürüyoruz
    if len(text_content.strip()) == 0:
        return "Sayfa içeriği yeterli değil veya bozuk."

    # Modelin token sınırına uymak için metnin uzunluğunu sınırlıyoruz
    max_input_length = 1024  # Modelin token limiti
    if len(text_content) > max_input_length:
        text_content = text_content[:max_input_length]

    try:
        # Hugging Face pipeline'ı ile özetliyoruz
        summary = summarizer(text_content, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Özetleme sırasında bir hata oluştu: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        content = get_page_content(url)

        if content:
            summary = summarize_page_content(content)
            return render_template('index.html', summary=summary)
        else:
            return render_template('index.html', error="URL'den içerik alınamadı veya geçersiz URL!")

    return render_template("index.html", summary=None)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template  # Flask modüllerini içeri alıyoruz: Web framework'ü oluşturacağız
from bs4 import BeautifulSoup  # BeautifulSoup, HTML içeriğini parse etmek için kullanılır
import requests  # URL'ye HTTP istekleri göndermek için requests modülü
from transformers import pipeline  # Hugging Face'ün özetleme için sağladığı 'pipeline' fonksiyonu

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)  # Flask uygulaması başlatılıyor, '__name__' parametresi uygulamanın adını belirtir

# Özetleme modelini yüklüyoruz
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # Hugging Face'ten özetleme için önceden eğitilmiş model yükleniyor

# URL'den sayfa içeriğini almak için fonksiyon
def get_page_content(url):
    """
    Bu fonksiyon, verilen URL'den HTML içeriğini alır ve döndürür.
    """
    try:
        response = requests.get(url)  # URL'ye HTTP GET isteği gönderiyoruz
        if response.status_code == 200:  # Eğer yanıt başarılı (200 OK) ise
            return response.text  # HTML içeriği döndürüyoruz
        else:
            print(f"URL'ye erişim hatası: {response.status_code}")  # Eğer HTTP durumu 200 değilse hata mesajı yazdırıyoruz
            return None  # Geçersiz bir içerik varsa None döndürüyoruz
    except requests.exceptions.RequestException as e:  # Eğer herhangi bir istek hatası meydana gelirse
        print(f"İstek hatası: {e}")  # Hata mesajını yazdırıyoruz
        return None  # Hata durumunda None döndürüyoruz

# Sayfa içeriğini özetlemek için ana işlev
def summarize_page_content(content):
    """
    Bu fonksiyon, verilen HTML içeriğinden metin çıkartır ve bu metni özetler.
    """
    soup = BeautifulSoup(content, 'html.parser')  # BeautifulSoup ile HTML içeriğini parse ediyoruz
    article_content = soup.find('article') or soup.find('main')  # Sayfada 'article' veya 'main' tag'ini buluyoruz

    if article_content:  # Eğer 'article' veya 'main' tag'i bulunduysa
        paragraphs = article_content.find_all('p')  # Bu element içindeki tüm <p> etiketlerini alıyoruz
    else:
        paragraphs = soup.find_all('p')  # Eğer 'article' veya 'main' yoksa, sayfada bulunan tüm <p> etiketlerini alıyoruz

    # Tüm paragrafların metin içeriklerini birleştiriyoruz
    text_content = ' '.join([para.get_text() for para in paragraphs])  # Paragrafları birleştirip tek bir metin haline getiriyoruz

    # Eğer içerik çok kısa veya boşsa, kullanıcıya bir hata mesajı döndürüyoruz
    if len(text_content.strip()) == 0:
        return "Sayfa içeriği yeterli değil veya bozuk."  # İçerik boşsa ya da sadece boşluk varsa hata mesajı

    # Modelin token sınırına uymak için metnin uzunluğunu sınırlıyoruz
    max_input_length = 1024  # Hugging Face modelinin token limiti
    if len(text_content) > max_input_length:  # Eğer metin çok uzunsa
        text_content = text_content[:max_input_length]  # Metnin sadece ilk 1024 karakterini alıyoruz

    try:
        # Hugging Face pipeline'ı ile metni özetliyoruz
        summary = summarizer(text_content, max_length=150, min_length=30, do_sample=False)  # 'summarizer' modelini kullanarak metni özetliyoruz
        return summary[0]['summary_text']  # Modelin döndürdüğü özeti alıp döndürüyoruz
    except Exception as e:  # Eğer özetleme sırasında bir hata oluşursa
        return f"Özetleme sırasında bir hata oluştu: {str(e)}"  # Hata mesajını döndürüyoruz

@app.route("/", methods=["GET", "POST"])  # Ana sayfa rotası tanımlanıyor
def index():
    """
    Ana sayfayı yöneten fonksiyon. URL'yi alır ve özetleme sonucunu gösterir.
    """
    if request.method == "POST":  # Eğer kullanıcı POST isteği gönderirse
        url = request.form.get("url")  # Formdan URL'yi alıyoruz
        content = get_page_content(url)  # URL'den içeriği çekiyoruz

        if content:  # Eğer içerik başarılı şekilde alındıysa
            summary = summarize_page_content(content)  # İçeriği özetliyoruz
            return render_template('index.html', summary=summary)  # Özet sonucu ile HTML sayfasını döndürüyoruz
        else:  # Eğer içerik alınamazsa
            return render_template('index.html', error="URL'den içerik alınamadı veya geçersiz URL!")  # Hata mesajını gösteriyoruz

    return render_template("index.html", summary=None)  # İlk başta özet olmadan sayfayı gösteriyoruz

# Flask uygulamasını çalıştırıyoruz
if __name__ == "__main__":  # Eğer bu dosya ana dosya olarak çalışıyorsa
    app.run(debug=True)  # Flask uygulamasını başlatıyoruz ve debug modunda çalıştırıyoruz

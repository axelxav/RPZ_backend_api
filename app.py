from flask import Flask, request, jsonify
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
from requests.exceptions import Timeout
from urllib.parse import urlparse, urljoin
import tldextract
from catboost import CatBoostClassifier
import csv

app = Flask(__name__)

# Inisialisasi model dan vectorizer
model_svc = load(open('modelSVC_15jan.pkl', 'rb'))
vectorizer = load(open('vectorizer_15jan.pkl','rb'))
model_cb = load(open('modelCB_15jan.pkl', 'rb'))

pornography_keywords = [
    'adult film industry', 'porn star', 'adult actress', 'adult actor', 'pornography production', 'erotic photography', 'adult content creator',
    'explicit videos', 'adult entertainment industry', 'adult film awards', 'erotic models', 'adult film director', 'adult film studio',
    'pornographic art', 'adult film scenes', 'erotic films', 'adult film database', 'adult film streaming', 'adult film reviews',
    'adult film scripts', 'adult film festivals', 'adult film actors', 'erotic film industry', 'pornography awards', 'adult film website',
    'adult film conventions', 'erotic film director', 'adult film genres', 'adult film streaming platform', 'adult film production company',
    'pornography news', 'adult film release', 'erotic film festivals', 'adult film recommendations', 'adult film history', 'pornographic literature',
    'adult film technology', 'pornographic culture', 'erotic film reviews', 'adult film exhibition', 'adult film festivals', 'adult film rating',
    'pornographic analysis', 'adult film forum', 'erotic film database', 'adult film festivals worldwide', 'adult film industry trends',
    'erotic film criticism', 'adult film studies', 'adult film archive', 'erotic film festivals', 'adult film production history', 'adult film scenes',
    'porno', 'pornografi', 'seks', 'dewasa', 'XXX', 'bokep', 'mesum', 'film porno', 'video dewasa', 
    'situs porno', 'film dewasa', 'sex tape', 'hubungan intim', 'gambar porno', 'nudes', 'bikini',
    'playboy', 'playgirl', 'adult entertainment', 'erotic', 'nude art', 'playmate', 'sexy',
    'hot model', 'erotic stories', 'adult chat', 'adult dating', 'adult content', 'intimate content',
    'sexting', 'erotic literature', 'explicit content', 'adult film', 'adult website',
    'artis porno', 'bintang porno', 'film dewasa', 'aktris dewasa', 'aktor dewasa', 'produksi porno', 'fotografi erotis', 'pencipta konten dewasa',
    'video eksplisit', 'industri hiburan dewasa', 'penghargaan film dewasa', 'model erotis', 'sutradara film dewasa', 'studio film dewasa',
    'seni porno', 'adegan film dewasa', 'film erotis', 'basis data film dewasa', 'streaming film dewasa', 'ulasan film dewasa',
    'naskah film dewasa', 'festival film dewasa', 'aktor film dewasa', 'industri film erotis', 'penghargaan porno', 'situs web film dewasa',
    'konvensi film dewasa', 'sutradara film erotis', 'genre film dewasa', 'platform streaming film dewasa', 'perusahaan produksi film dewasa',
    'berita industri film dewasa', 'rilis film dewasa', 'festival film erotis', 'rekomendasi film dewasa', 'sejarah film dewasa', 'literatur porno',
    'teknologi film dewasa', 'budaya pornografi', 'ulasan film erotis', 'pameran film dewasa', 'festival film dewasa', 'peringkat film dewasa',
    'analisis pornografi', 'forum film dewasa', 'basis data film erotis', 'festival film dewasa di seluruh dunia', 'tren industri film dewasa',
    'kritik film erotis', 'studi film dewasa', 'arsip film dewasa', 'festival film erotis', 'sejarah produksi film dewasa', 'adegan film dewasa',
    'dragon ball z grapefruit', 'sunny outdoor message', 'cliff rubbing',
    'has nipples', 'cougars and kittens', 'hypno lesbian mind control', 'ahme sweater',
    'dirty talking aeroplane', 'sybian', 'diving underwater', 'china sex tape', 'sensual perfect',
    'backfro', 'manila exposed', 'iran hentai', 'japanese mom', 'japanese wife', 'japanese uncensored',
    'hentai', 'hentai uncensored', 'hentai sub indo', 'hentai sub indo', 'hentai sub indo', 
    'hentai sub indo', 'or some shit', 'ass parade', 'homemade cum', 'sexy patty cake', 'how to stay hard', 
    'girls making themselves cum with using their hands', 'get caught', 'masturbate', 'masturbating', 
    'milf', 'intense incese', 'incese', 'doctor gives her', 'bootytalk', 'jerking', 'jerk off', 
    'cum race', 'sweet piece of ash', 'pussy', 'cock', 'cum', 'anal', 'sextape', 'monstercock', 
    'monster cock', 'swimming in clothes', 'porn', 'porno', 'lesbian', 'asian', 'step mom', 'japanese', 
    'hentai', 'teen', 'massage', 'korean', 'ebony', 'anal', 'threesome', 'big ass', 'fortnite', 
    'chinese', 'big tits', 'cartoons', 'creampie', 'overwatch', 'gangbang', 'anime', 'trans', 
    'stormy daniels', 'mia khalifa', 'lana rhoades', 'brandi love', 'nicole aniston', 
    'august ames', 'lisa ann', 'alexis texas', 'abella danger', 'jordi el nino polla', 
    'kim kardhasian sex tape', 'asa akira', 'madison ivy', 'big black cock', 'bbc',
    'dillion harper', 'kim kardashian', 'sunny leone', 'adriana chechik', 'kimny granger',
    'dani daniels', 'big tits', 'big dick', 'transgender', 'gangbang', 'bondage', 
    'blowjob', 'aoi sora', 'ai uehara', 'lexi belle', 'amanda love', 'julia japanese',
    'lesbian scissoring', 'big ass', 'bbw', 'ebony milf', '3d hentai', 'anime hentai uncensored',
    'pornstar', 'bokep', 'bokep viral', 'bokepviral', 'bokep indo', 'bokepindo', 'live omek',
    'liveomek', 'bokep smp', 'bokepsmp','nyepong', 'colmek', 'jilbabviral','jilbab viral', 'bokep chindo',
    'bokepchindo', 'bokep binor', 'bokepbinor', 'semok', 'bokep tante', 'bokeptante', 'prank ojol',
    'prankojol', 'ngewe', 'ngecrot', 'crot', 'memek', 'skandal bokep', 'skandalbokep','sampe klimaks',
    'sampeklimaks', 'ampe klimaks', 'ampeklimaks', 'layanin nafsu','layaninnafsu', 'memek tembem',
    'memektembem', 'abg smp', 'abgsmp', 'abgttmulus', 'abg tt mulus', 'tt mulus', 'ttmulus', 
    'nenen', 'suka nenen', 'sukanenen', 'bokep abg', 'bokepabg', 'ewe', 'funcrot', 'toket gede', 'toketgede',
    'toket', 'jilboobs', 'jilboob', 'memek gundul', 'memekgundul', 'ngewe sama boss', 'ngewesamaboss', 
    'entod', 'ngentod', 'ngentot', 'toket mungil', 'toketmungil', 'toket kecil', 'toketkecil', 'toket kecil',
    'sange', 'engas', 'toge', 'colek memek', 'colekmemek', 'memek basah', 'memekbasah', 'binal', 'body binal',
    'bodybinal', 'avtub','bokepbocil', 'bokep bocil','bokep terbaru', 'bokepterbaru','bokep indonesia', 
    'bokepindonesia','bokephijab', 'bokep hijab','bokeplive','bokep live','bokepindia','bokep india',
    'bokepindoh', 'bokep indoh','bokepcindo','bokep cindo','bokep avtub','bokepavtub','bokepcolmek',
    'bokep colmek','bokep 2024','bokepcrot','bokep crot','bokep malay','bokepmalay','bbffme','bokepsin',
    'bokep sin', 'bbywulan','billa seleb tiktok','bebas indo','bokep janda','ciya tiktok','cantik','doggy',
    'doggy style','funcrot','facecrot','hijab husna','kebokepanku','meki sempit','porn india','pornindia',
    'remes toket', 'remestoket','sebokep','toketgede', 'toket gede','vibokep','wot','zee bokep','zeebokep',
    'ngewein binor', 'binor', 'hijab montok', 'hijabmontok', 'hijab toge', 'hijabtoge', 'hijab sange',
    'asupan bokep', 'asupanbokep', 'asupan bokep indo', 'asupanbokepindo', 'asupan bokep indonesia',
    'asupanbokepindonesia', 'asupan bokep viral', 'asupanbokepviral', 'asupan bokep terbaru', 'asupanbokepterkini',
    'tete mulus', 'tetemulus', 'tete gede', 'tetegede', 'tete kecil', 'tetemulus', 'tete montok', 'tetemontok',
    'doodcrot', 'doodstream', 'dood', 'doodcrot.live', 'doodcrotlive', 'tobrut', 'toket brutal', 'livecrot',
    'ngentotin', 'entodin', 'entotin', 'ngentotin', 'ngentodin', 'ngewein', 'montok', 'genjotin', 'genjot',
    'remes tt', 'hypersex', 'hyper sex', 'prindavan', 'xnxx', 'xxx', 'xvideos', 'xhamster', 'xhamsterlive',
    'ayam kampus','asupan bokep','asupanbokep','bokep nt','bokepnyepong', 'bokep nyepong','bokep ini', 
    'bokepini','bokep viral terbaru', 'bokepviralterbaru','bokep rare', 'bokeprare','cum','hot mom',
    'hotmom', 'milana milka', 'milanamilka','pornhub','stw', 'crot dimulut', 'crot di mulut',
    'crotdimulut', 'crot di muka', 'crot dimuka', 'sepongan', 'sepong', 'sepong kontol', 'sepongan kontol',
    'sepongankontol', 'crot di mulut', 'skandal bokep', 'skandalbokep', 'skandal bokep indo', 'skandalbokepindo',
    'skandal bokep indonesia', 'skandalbokepindonesia', 'skandal bokep terbaru', 'skandalbokepterkini',
    'skandal bokep viral', 'skandalbokepviral', 'skandal bokep 2021', 'skandalbokep2021', 'skandal bokep 2022',
    'skandalbokep2022', 'skandal bokep 2023', 'skandalbokep2023', 'skandal bokep 2024', 'skandalbokep2024',
    'body mulus', 'bodymulus', 'ketahuan colmek', 'pejuh', 'pejuhin', 'pejuhin memek', 'pejuh memek',
    'ngewe di kos', 'ngentod di kos', 'desahan', 'desahan nikmat', 'crot di memek', 'crot dimemek',
    'colemkin', 'colemkin memek', 'colemkinmemek', 'colemekin', 'colemekin memek', 'colemekinmemek',
    'jilbobs', 'ngews', 'ngew', 'climax', 'squirt', 'squirting', 'squirted', 'vaginal cum shot',
    'half orgasm', 'fuck', 'fucked', 'emma futaba', 'mion sakuragi', 'sex', 'model telanjang',
    'javsubindo', 'javgg', 'onlyfans', 'amateurs', 'black aktor', 'black actor',
    'vidcallsex', 'vcs', 'tocil', 'meruchan', 'live hot', 'gesek memek', 'dildo', 'doodstream',
    'jembud', 'chudai', 'https://poophd.com', 'kendra james', 'reagan foxx', 'hazel moore', 'chloe surreal',
    'mommysgirl', 'jerking off', 'poophd', 'mengocok daging hangat', 'cerita dewasa', 'novel dewasa'
]

gambling_keywords = [
    'kasino', 'casino', 'taruhan', 'judi', 'poker', 'slot', 'blackjack', 'roulette', 'bandarq', 'dominoqq',
    'sportsbook', 'togel', 'bola tangkas', 'sabung', 'agen judi', 'online betting', 
    'situs judi online', 'agen judi online', 'bandar judi', 'taruhan bola online', 
    'domino online', 'poker online', 'togel online', 'slot online', 'live casino online',
    'agen bola terpercaya', 'judi sabung ayam', 'judi online terbaik', 'bandarq online', 
    'agen togel terpercaya', 'agen slot terbaik', 'agen bola online terpercaya', 'agen bola online terbaik', 
    'permainan kasino online', 'mesin slot', 'mesin judi', 'permainan kartu online', 'betting online', 'agen bola online',
    'permainan judi online', 'bonus judi online', 'uang asli', 'penjudi online', 'daftar judi online',
    'cara main judi online', 'jackpot', 'uang taruhan', 'uang kemenangan', 'deposit judi online', 'withdraw judi online',
    'sistem permainan judi online', 'peluang menang', 'taruhan sepak bola online', 'situs judi terpercaya', 'agen slot online',
    'agen togel online', 'turnamen poker online', 'permainan live casino', 'game judi online terbaik', 'sistem keamanan judi online',
    'panduan bermain judi online', 'metode pembayaran judi online', 'pemenang judi online', 'taktik judi online', 'agen sabung ayam online',
    'agen bola tangkas online', 'teknologi judi online', 'uang taruhan bola', 'permainan sabung ayam online', 'sistem fair play judi online',
    'keuntungan bermain judi online', 'agen live casino online', 'panduan taruhan online', 'agen poker terpercaya', 'agen domino online',
    'bandarq terbaik', 'agen judi blackjack', 'panduan bermain roulette online', 'bandarq fair play', 'situs togel terpercaya',
    'panduan bermain bola tangkas online', 'pemenang togel online', 'agen judi bola tangkas', 'agen judi online terbaik di Indonesia',
    'metode deposit judi online', 'metode withdraw judi online', 'agen judi online resmi', 'panduan bermain blackjack online',
    'cara menang judi online', 'situs judi terbaik di Indonesia',
    'turnamen slot online', 'agen judi slot online terpercaya', 'kemenangan besar judi online', 'situs judi online paling populer',
    'taruhan olahraga online', 'agen judi online terbesar', 'cara memilih agen judi online', 'daftar situs judi online terpercaya',
    'agen judi online terbaik 2022', 'daftar bandar judi online terpercaya', 'panduan bermain poker online', 'agen judi domino terpercaya',
    'keuntungan bermain judi bola online', 'sistem keamanan terbaik judi online', 'agen judi online resmi dan terpercaya',
    'agen judi online terpercaya di Indonesia', 'agen judi online terpercaya 2022', 'agen judi online terpercaya 2021', 'agen judi online terpercaya 2020', 
    'betting site', 'online gambling', 'gambling site', 'betting platform', 'betting odds', 'casino games', 'poker tournament', 'online poker room', 
    'online poker room', 'slot machine games', 'sports betting', 'betting tips', 'online casino bonus', 'poker strategy', 'online roulette', 
    'live blackjack', 'poker chips', 'online betting guide', 'online sportsbook', 'online slot games', 'online poker guide', 'roulette system',
    'blackjack strategy', 'online gambling community', 'casino promotions', 'betting strategies', 'online casino reviews', 'poker rules',
    'online gambling news', 'casino bonuses', 'online casino promotions', 'online poker tips', 'online betting reviews', 'online poker reviews',
    'online casino games', 'gambling news', 'online casino strategy', 'online betting news', 'poker odds', 'online gambling guide',
    'casino jackpots', 'slot machine tips', 'poker hands', 'online casino tournaments', 'online betting strategies', 'online poker strategies',
    'poker tournaments online', 'online gambling tips', 'online betting tips', 'online casino tips', 'casino strategy', 'poker tips',
    'online casino odds', 'online poker odds', 'online poker tournaments', 'online slot tips', 'sports betting tips', 'casino betting',
    'online betting odds', 'poker betting', 'online gambling odds', 'online gambling strategies', 'online slot strategy', 'casino betting strategies',
    'casino', 'online casino', 'betting', 'gambling', 'poker', 'slot', 'blackjack', 'roulette', 'baccarat', 'bingo',
    'sportsbook', 'lottery', 'baccarat', 'cockfighting', 'gambling site', 'online betting', 'online gambling site',
    'online poker', 'online slot', 'live casino', 'trusted online casino', 'gambling platform', 'online sportsbook',
    'online roulette', 'virtual sports betting', 'online baccarat', 'online bingo', 'online craps', 'online lottery',
    'online poker room', 'online blackjack', 'online gambling community', 'online casino games', 'online betting odds',
    'online gambling tips', 'online casino bonuses', 'online poker strategy', 'online betting guide', 'online poker tips',
    'online sports betting tips', 'online gambling news', 'online casino promotions', 'online poker tournaments',
    'online casino reviews', 'online gambling strategies', 'online slot tips', 'online betting strategies', 'online roulette system',
    'online blackjack strategy', 'online casino bonuses', 'online casino jackpots', 'online slot machine tips', 'online poker odds',
    'online sports betting odds', 'online poker hands', 'online gambling odds', 'online gambling promotions', 'online casino strategy',
    'online poker reviews', 'online gambling news', 'online casino bonuses', 'online casino promotions', 'online poker tips',
    'online betting reviews', 'online poker reviews', 'online casino games', 'online gambling community', 'online casino promotions',
    'online poker strategy', 'online betting guide', 'online sports betting tips', 'online gambling news', 'online casino reviews',
    'online gambling strategies', 'online poker tournaments', 'online betting strategies', 'online roulette system', 'online blackjack strategy',
    'online casino jackpots', 'online slot machine tips', 'online poker odds', 'online sports betting odds', 'online poker hands',
    'online gambling odds', 'online gambling promotions', 'online casino strategy', 'online poker reviews', 'online gambling news', 
    'gacor', 'toto', 'slot', 'judi', 'poker', 'togel', 'judi bola', 'casino', 'kasino', 'tangkas', 
    'tangkasnet', 'olympus', 'zeus', 'maxwin', 'maxbet', 'raja bandit', 'qq', 'hacksitus', 'gacors', 'index of', 'qq slot', 'dominoqq',
    'qqpulsa', 'wd', 'qiuqiu', 'qiu qiu', 'togel', 'deposit', 'qqmybet', 'qqmamibet', 'qqemas', 'qqalfa', 'qq801', 'qqmobil', 'gampang menang',
    'depo via pulsa ewallet', 'agen togel online aman Dan terpercaya', 'situs togel online terpercaya', 'situs judi slot online terpercaya',
    'web togel', 'judi togel online', 'slot online', 'casino online', 'pasti profit', 'jamin betah',
    'pasti profit jamin betah', 'bandar togel', 'bandar togel aman dan terpercaya', 'himalaya4d', 'himalaya 4d',
    'judi', 'poker', 'kasino', 'togel', 'casino', 'judi bola', 'slot', 'roulette', 'blackjack', 'taruhan', 
    'toto', 'lotre', 'lottery', 'casino', 'betting', 'bet', 'gambling', 'pacuan kuda', 'tembak ikan', 
    'casino online', 'bandar slot', 'bandar judi', 'agen slot', 'agen judi', 'agen casino', 
    'agen bola', 'agen togel', 'agen poker', 'gacor', 'gacor77', 'gacor88', 'gacor99', 'gacor123', 
    'gacor678', 'gacor6789', 'idnpoker', 'merupakan salah satu situs agen judi online terpercaya',
    'hanya perlu 1 id', 'minimal deposti hanya', 'permainan judi online berkualitas', 'bonus togel',
    'pasaran togel', 'live draw', 'pools', 'gambling', 'lottery pools', 'singapore pools', 'horse racing',
    'winning ticket prize', 'ticket prize', 'pola slot', 'info bocoran pola', 'rtp live slot',
    'rtp live', 'pola slot', 'pragmatic play','pgsoft', 'habanero', 'spadegaming', 'joker123', 'playtech',
    'joker', 'cq9', 'spade gaming', 'ion slot', 'slot88', 'ae tiger', 'betsoft', 'astrotech', 'jamgacor',
    'jamgacorrange', 'rajavip', 'raja vip', 'rajavipslot', 'raja vip slot', 'slot777', 'funky games',
    'jiligaming', 'jili gaming', 'afb gaming', 'afbgaming', 'rich88', 'kingmaker', 'netent', '568win', '568 win',
    'sexygaming', 'ion casino', 'allbet', 'maxbet', 'money roll', 'new member slot', 'lucky spin anti zonk',
    'casino filipino','pagcor casinos','casino filipino website', 'casino filipino manila',
    'manila casinos', 'idnslot', 'slot-mania', 'pragmaticplay', 'habanero', 'pgsoft', 'evolution-nlc',
    'microgaming', 'pragmaticplay98', 'spadegaming_slot', 'evolution-redtiger', 'evolution-netent',
    'fastspin', 'rtp slot', 'rtp slot terbaik', 'rtp slot terbesar', 'rtp slot terpercaya', 'rtp slot terbaru',
    'pasang togel', 'nolimit city', 'sbo slot', 'live22', 'all bet', 'sbobet', 'sbo bet', 'games gacor',
    'bandar betting terbaik', 'pg soft', 'ibcbet', 'sgd777', 'bola tangkas', 'sbc168', 'sbobet casino',
    'situs bandar bola', 'situs judi bola', 'situs judi slot', 'situs judi online', 'situs judi online terpercaya',
    'joker gaming', 'agen bola terbesar'
]

word_list = load(open('modelCB_columns_15jan.pkl', 'rb'))

# Fungsi untuk mengambil konten HTML dari URL
def fetch_html_content(url):
    try:
        response = requests.get(url, verify=False, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.get_text() if soup else None 
        return content.lower()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

# Fungsi untuk memproses URL
def process_url(url, word_list, timeout_duration=2):
    try:
        # Mencoba mendapatkan konten HTML dari URL
        response = requests.get(url, timeout=timeout_duration)
        response.raise_for_status()
        html = response.text

        # Memproses HTML untuk menemukan kata-kata
        soup = BeautifulSoup(html, 'lxml')

        # Menggabungkan teks dari halaman utama dengan teks dari hyperlink terpilih
        page_text = soup.get_text(separator=' ', strip=True).lower()
        hyperlinks = set([a.get('href') for a in soup.find_all('a', href=True)])
        hyperlink_texts = []

        # Memproses hanya 3 hyperlink unik pertama
        for link in list(hyperlinks)[:3]:
            try:
                if not link.startswith('http'):
                    link = urljoin(url, link)

                link_response = requests.get(link, timeout=timeout_duration)
                link_response.raise_for_status()
                link_html = link_response.text
                link_soup = BeautifulSoup(link_html, 'lxml')
                hyperlink_texts.append(link_soup.get_text(separator=' ', strip=True).lower())
            except:
                # Gagal mengakses hyperlink, lanjutkan ke link berikutnya
                continue

        # Gabungkan semua teks yang didapat
        combined_text = page_text + ' '.join(hyperlink_texts)

        # Menghitung keberadaan setiap kata kunci
        found_word_counts = {word: 1 if word in combined_text else 0 for word in word_list}
        return found_word_counts
    except (ConnectionError, Timeout, requests.exceptions.HTTPError, requests.exceptions.RequestException):
        # Jika terjadi kesalahan, kembalikan None
        print(f"Error fetching or processing URL: {url}")
        return None

# Fungsi untuk mendapatkan tag berdasarkan label prediksi
def get_tags(label, content):
    if label == 0:
        return "safe"
    elif label == 1:
        gambling_count = sum(keyword in content for keyword in gambling_keywords)
        pornography_count = sum(keyword in content for keyword in pornography_keywords)
        
        if gambling_count > pornography_count:
            return "gambling"
        elif pornography_count > gambling_count:
            return "pornography"
        else:
            return "malicious"
    else:
        return "unknown"

# Fungsi untuk menentukan jenis berdasarkan konten
def assign_type(row, pornography_keywords, gambling_keywords):
    porn_count = sum(row[keyword] for keyword in pornography_keywords)
    judi_count = sum(row[keyword] for keyword in gambling_keywords)
    # Menentukan tipe berdasarkan jumlah keyword terbanyak
    if porn_count > judi_count:
        return 'pornography'
    elif judi_count > porn_count:
        return 'gambling'
    return 'safe'

# Route untuk endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    
    new_content_svc = fetch_html_content(url)
    new_content_cb = process_url(url, word_list)

    if new_content_svc is not None:
        new_content_svc = new_content_svc.replace('\n', '').replace('\t', '')
        new_features = vectorizer.transform([new_content_svc])
        predicted_label_svc = model_svc.predict(new_features)[0]
        svc_proba = np.max(model_svc.predict_proba(new_features)[0])
        predicted_tag = get_tags(predicted_label_svc, new_content_svc)
        result = {
            'model': 'SVC',
            'label': predicted_label_svc,
            'tag': predicted_tag,
            'probability': svc_proba
        }
    else:
        result = {
            'error': f"Error fetching URL content: {url}"
        }

    if new_content_cb is not None:
        new_features = [new_content_cb.get(word, 0) for word in word_list]
        predicted_label_cb = model_cb.predict([new_features])[0]
        cb_proba = np.max(model_cb.predict_proba([new_features])[0])
        predicted_type = assign_type(new_content_cb, pornography_keywords, gambling_keywords)
        if svc_proba > cb_proba:
            result = {
                'model': 'SVC',
                'label': predicted_label_svc,
                'tag': predicted_tag,
                'probability': svc_proba
            }
        elif cb_proba > svc_proba:
            result = {
                'model': 'CB',
                'label': predicted_label_cb,
                'type': predicted_type,
                'probability': cb_proba
            }
    else:
        result = {
            'error': f"Error fetching URL content: {url}"
        }

    return result

if __name__ == '__main__':
    app.run(debug=True)
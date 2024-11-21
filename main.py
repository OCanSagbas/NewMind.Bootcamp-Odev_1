import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Interval Continuous Data --> mean, median, standart deviation -- anlamlı sıfır noktası yok
#Ratio Continuous Data --> arithmetic, geometric mean -- anlamlı sıfır noktası var
#Continuous Data --> histogram, line plots, box plots, scatter plots
#Continuous Data --> scaling, normalization, transformation
#Categorical Data --> bar, pie chart
#Categorical Data --> encoding


pd.set_option("display.max_columns", 15)
musteri_verisi = pd.read_csv('musteri_verisi_5000_utf8.csv')
satis_verisi = pd.read_csv('satis_verisi_5000.csv')

#Verilerin ilk 5 satırını inceledim.
musteri_verisi.head()
satis_verisi.head()

#Verilerin türlerine baktım.
musteri_verisi.info()
#satış verisinde toplam_satış, fiyat verileri float olması gerekirken, object
#yanlış verileri barındırıyor içinde
satis_verisi.info()

#Verilerin mean,std,min,max değerlerinde anormallik incelemesi.
musteri_verisi.describe()
satis_verisi.describe()

#Müşteri verilerinde outlier veya yanlış veri görünmüyor.
plt.figure(figsize=(10,10))
sns.boxplot(musteri_verisi["harcama_miktari"])
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(satis_verisi["adet"])
plt.show()

#Null değer yok.
null_count1 = musteri_verisi.isnull().sum()
null_count2 = satis_verisi.isnull().sum()

#String şeklinde olan toplam satış verisini float değerine taşıdık, e4 gibi gösterimden kurtulmak için formatı değiştirdik.
satis_verisi["toplam_satis"] = pd.to_numeric(satis_verisi["toplam_satis"], errors="coerce", downcast ="float")
pd.options.display.float_format = "{:.2f}".format
#fiyatxadet=toplam_satis ama fiyat değeri de tarih //4 veri var
satis_verisi.loc[satis_verisi["toplam_satis"].isnull()]["toplam_satis"]

#adet*fiyat=toplam_satis //4 veri dışındakilerden fiyat değeri çıkıyor. Burada direkt fiyat hesaplanır.
satis_verisi["fiyat"] = pd.to_numeric(satis_verisi["fiyat"], errors="coerce").astype("float64")
#Null yaptığımız fiyat verileri adet*fiyat=toplam_satis'dan hesaplandı.
satis_verisi.loc[satis_verisi["fiyat"].isnull(), "fiyat"] = (satis_verisi["toplam_satis"] / satis_verisi["adet"])[satis_verisi["fiyat"].isnull()]
satis_verisi.loc[satis_verisi["fiyat"].isnull()]["fiyat"]
satis_verisi.iloc[794,5]
satis_verisi.iloc[159,5]

#Burada yapılan işlemlerde satiş ve toplam fiyat verilerinde maksimum değerlerde aşırı yükseklik var.
satis_verisi.describe()
satis_verisi.nunique()

#4 null değerimiz ve outlier verilerimiz var. Null değerleri çok yüksek verileri dışarda bırakarak mean değere ekledim.
satis_verisi.loc[satis_verisi["fiyat"].isnull(), "fiyat"] = satis_verisi.loc[satis_verisi["fiyat"]<50000,"fiyat"].mean()
satis_verisi.loc[satis_verisi["toplam_satis"].isnull(), "toplam_satis"] = satis_verisi.loc[satis_verisi["toplam_satis"]<100000,"toplam_satis"].mean()

#Outlier'ı bulma ve çıkarma
def find_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

variable="toplam_satis"
upper_boundary, lower_boundary = find_boundaries(satis_verisi, variable, 1.5)

outliers_index = (satis_verisi[variable]>upper_boundary) | (satis_verisi[variable]<lower_boundary)
listofoutliers = list(satis_verisi[outliers_index].index)
satis_verisi.loc[listofoutliers,"toplam_satis"] = satis_verisi.loc[listofoutliers,"fiyat"]*satis_verisi.loc[listofoutliers,"adet"]

satis_verisi.describe()

variable2='fiyat'
upper_boundary, lower_boundary = find_boundaries(satis_verisi, variable2, 1.5)

outliers_index = (satis_verisi[variable2]>upper_boundary) | (satis_verisi[variable2]<lower_boundary)
listofoutliers2 = list(satis_verisi[outliers_index].index)
satis_verisi=satis_verisi.drop(index=listofoutliers2)

satis_verisi.describe()

#One to Many ilişkisi
merged_data = pd.merge(musteri_verisi, satis_verisi, on="musteri_id")
merged_data.head()

merged_data.info()

# 'tarih' sütununu datetime formatına çevirme
merged_data["tarih"] = pd.to_datetime(merged_data["tarih"])

# 'tarih' sütununu indeks olarak ayarlama
merged_data.set_index("tarih", inplace=True)
merged_data = merged_data.sort_index()

haftalik_toplam_satis = merged_data["toplam_satis"].resample("W").sum()
aylik_toplam_satis = merged_data["toplam_satis"].resample("ME").sum()

haftalik_urun_satis = merged_data.groupby("ürün_adi")["toplam_satis"].resample("W").sum()
aylik_urun_satis = merged_data.groupby("ürün_adi")["toplam_satis"].resample("ME").sum()

haftalik_toplam_satis.plot(title="Haftalık Toplam Satış")
aylik_toplam_satis.plot(title="Aylık Toplam Satış")
plt.show()

#Resample'daki MS ve ME kullanımları tarihlerin ilk ve son gününü veriyor satışın değil. Satışın ilk ve son
# gününü bulmak için kodu bu şekle getirdim.
ilk_satis_gunleri = merged_data.groupby(merged_data.index.to_period("M")).apply(lambda x: x.index.min())
son_satis_gunleri = merged_data.groupby(merged_data.index.to_period("M")).apply(lambda x: x.index.max())

# Her hafta satılan toplam ürün adeti
haftalik_urun_sayisi = merged_data["adet"].resample("W").sum()

aylik_toplam_satis.plot(
    figsize=(8, 5),
    marker='o',
    title="Aylık Toplam Satış",
    ylabel="Toplam Satış (₺)",
    xlabel="Tarih",
    grid=True
)
plt.show()

haftalik_urun_sayisi.plot(
    figsize=(8, 5),
    marker='o',
    color="orange",
    title="Aylık Ürün Satış Miktarı",
    ylabel="Toplam Ürün Adedi",
    xlabel="Tarih",
    grid=True
)
plt.show()

# Kategorilere göre toplam satışların değerleri
kategori_toplam_satis = merged_data.groupby("kategori")["toplam_satis"].sum()

# her bir kategorinin tüm satışlara oranı
kategori_oranlari = (kategori_toplam_satis / kategori_toplam_satis.sum()) * 100

bins = [18, 25, 35, 50, 100]
labels = ["18-25", "26-35", "36-50", "50+"]
merged_data["yas_grubu"] = pd.cut(merged_data["yas"], bins=bins, labels=labels, right=False)
yas_grubu_toplam_satis = merged_data.groupby("yas_grubu")["toplam_satis"].sum()

harcama_analizi = merged_data.groupby("cinsiyet")["harcama_miktari"].agg(["sum", "mean"])
kategori_harcamalari = merged_data.groupby(["cinsiyet", "kategori"])["toplam_satis"].sum()
en_cok_harcama_kategori = kategori_harcamalari.groupby("cinsiyet").idxmax()

# Şehir bazında toplam harcama miktarı
sehir_bazinda_harcamalar = merged_data.groupby("sehir")["harcama_miktari"].sum()

# Şehirleri toplam harcama miktarının sıralaması
sehir_bazinda_harcamalar_sirali = sehir_bazinda_harcamalar.sort_values(ascending=False)

# Tarihleri aylık döneme çevirdim.
merged_data['tarih_ay'] = merged_data.index.to_period('M')
# Her bir ürün için aylık toplam satışları hesapladım.
aylik_urun_satislari = merged_data.groupby(['tarih_ay', 'ürün_kodu'])['toplam_satis'].sum().reset_index()

# Değişim oranını hesaplamak için veriyi sıralayın
aylik_urun_satislari.sort_values(by=['ürün_kodu', 'tarih_ay'], inplace=True)
# Bir önceki ayın toplam satışını hesaplayın
aylik_urun_satislari['onceki_ay_satis'] = aylik_urun_satislari.groupby('ürün_kodu')['toplam_satis'].shift(1)
# Değişim oranı hesaplamas kısmı
aylik_urun_satislari['satis_degisim_orani'] = ((aylik_urun_satislari['toplam_satis'] - aylik_urun_satislari['onceki_ay_satis']) / aylik_urun_satislari['onceki_ay_satis']) * 100

# Tarihi aylık bazda gruplamak için dönüştürme
merged_data['tarih_ay'] = merged_data.index.to_period('M')
# Her kategori için aylık toplam satışları hesapla
aylik_satislar = merged_data.groupby(['tarih_ay', 'kategori'])['toplam_satis'].sum()
# Aylık değişim oranını hesapla (pct_change ile yüzdelik değişim)
degisim_oranlari = aylik_satislar.groupby(level=1).pct_change() * 100

# Ürün bazında toplam satışları hesapla
urun_satislari = merged_data.groupby("ürün_adi")["toplam_satis"].sum()

# Satışlara göre azalan sırada sıralama
urun_satislari = urun_satislari.sort_values(ascending=False)

# Kümülatif satış oranını hesapla
kümülatif_oran = urun_satislari.cumsum() / urun_satislari.sum()

# %80'in altındaki ürünleri belirle
pareto_urunleri = urun_satislari[kümülatif_oran <= 0.8]

# Grafik çizimi
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
pareto_urunleri.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title("Pareto Analizi: Satışların %80'ini Oluşturan Ürünler")
plt.ylabel("Toplam Satış")
plt.xlabel("Ürün Adı")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
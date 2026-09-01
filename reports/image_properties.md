# Goruntu Ozellikleri Ozeti

Ham APTOS-2019 fundus goruntuleri, n=3662.

## Okunabilirlik ve renk modu

| kontrol | sonuc |
|---|---|
| Okunabilir | 3662 / 3662 |
| Renk modu | RGB: 3662 |
| Kanal sayisi | 3: 3662 |

Renk modu dosya bicimine degil icerige gore belirlendi: uc kanali
birebir esit olan bir goruntu RGB kaydedilmis olsa da gri tonlamadir.

## Cozunurluk

| olcu | en kucuk | medyan | en buyuk |
|---|---|---|---|
| Genislik | 474 | 2144 | 4288 |
| Yukseklik | 358 | 1536 | 2848 |
| Megapiksel | 0.17 | 3.15 | 12.21 |
| En-boy orani | 1.000 | 1.333 | 1.506 |

Toplam **17 farkli cozunurluk** var. En sik gorulenler:

| cozunurluk | adet | oran |
|---|---|---|
| 1050x1050 | 974 | %26.6 |
| 2416x1736 | 638 | %17.4 |
| 2588x1958 | 533 | %14.6 |
| 3216x2136 | 410 | %11.2 |
| 2048x1536 | 351 | %9.6 |
| 819x614 | 287 | %7.8 |
| 3388x2588 | 141 | %3.9 |
| 1504x1000 | 92 | %2.5 |

974 goruntu (%26.6) zaten kare; geri kalani resize oncesi kareye tamamlanir. Cozunurlugun bu kadar degisken olmasi sabit boyuta getirmeyi zorunlu kiliyor.

## Piksel istatistikleri

| olcu | en dusuk | ortalama | en yuksek |
|---|---|---|---|
| Parlaklik | 15.0 | 66.8 | 129.6 |
| Kontrast (std) | 9.6 | 38.8 | 75.8 |
| Siyah piksel orani | 0.000 | 0.232 | 0.528 |

Parlaklik hicbir goruntude 130'un uzerine cikmiyor; fundus fotograflari
dogasi geregi koyu. Sabit bir "asiri parlak" esigi (orn. 240) bu veri
setinde hicbir seyi elemez, yuzdelik tabanli esik daha anlamli.

## Auto-crop kazanci

Retina disindaki siyah cerceve kirpildiginda alanin ortalama **%10.4**'i atiliyor (en az %0.0, en cok %39.9).

---

Grafikler: `reports/figures/06_image_properties.png`. Goruntu basina ayrintili veri: BigQuery `aptos_image_stats`.

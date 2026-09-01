# Kisayol Ozellikleri ve Etiket Gurultusu

Modelin skorunu dogru sebeple alip almadigini ve ulasabilecegi tavani sorgulayan iki analiz.

## 1. Meta-veri kisayolu

Yalnizca dosya ozelliklerinden (cozunurluk, en-boy orani, parlaklik, kontrast, dosya boyutu) egitilen bir RandomForest. **Retinaya hic bakmiyor.** 5 katli capraz dogrulama.

| olcut | meta-veri tabani | hep-0 tahmini |
|---|---|---|
| Accuracy | 0.7081 | 0.4929 |
| QWK | **0.6519** | 0.0000 |

En ayirt edici meta-veri ozellikleri:

| ozellik | onem |
|---|---|
| `file_kb` | 0.235 |
| `contrast_std` | 0.190 |
| `brightness` | 0.186 |
| `aspect_ratio` | 0.129 |
| `width` | 0.100 |

Kaynagi: en sik cozunurluk olan 1050x1050 goruntulerin **%92.5**'i `No DR`, diger cozunurluklerde bu oran **%33.6**.

| evre | 1050x1050 (n=974) | digerleri (n=2688) |
|---|---|---|
| 0 No DR | 901 (%92.5) | 904 (%33.6) |
| 1 Mild | 19 (%2.0) | 351 (%13.1) |
| 2 Moderate | 39 (%4.0) | 960 (%35.7) |
| 3 Severe | 2 (%0.2) | 191 (%7.1) |
| 4 Proliferative DR | 13 (%1.3) | 282 (%10.5) |

### Ne anlama geliyor

APTOS verisi Hindistan'da birden fazla merkezde, farkli cihazlarla toplandi. Cozunurluk cihazin imzasi ve cihaz ile hastalik yayginligi arasinda guclu bir iliski var. Model yeniden boyutlandirilmis goruntuleri gorse de en-boy orani, keskinlik ve kenar geometrisi bu bilgiyi tasimaya devam eder.

Bu, sonuclarin gecersiz oldugu anlamina gelmez; ama **QWK 0.652'lik bir kismi teshis olmadan da elde edilebilir** demektir. Rapor edilen skorlar bu taban ile birlikte okunmalidir.

## 2. Sinif bazli goruntu ozellikleri

| evre | n | parlaklik | kontrast | megapiksel | kare olan |
|---|---|---|---|---|---|
| 0 No DR | 1805 | 71.4 | 43.3 | 1.10 | %50 |
| 1 Mild | 370 | 68.5 | 33.7 | 4.19 | %5 |
| 2 Moderate | 999 | 68.8 | 37.5 | 5.07 | %4 |
| 3 Severe | 193 | 64.2 | 35.3 | 5.07 | %1 |
| 4 Proliferative DR | 295 | 66.5 | 36.5 | 5.07 | %4 |

Kruskal-Wallis testi (siniflar arasi fark anlamli mi):

| ozellik | p | sonuc |
|---|---|---|
| brightness | 7.36e-05 | farkli |
| contrast_std | 1.73e-67 | farkli |
| megapixels | 0.00e+00 | farkli |

`No DR` goruntuleri medyan 1.10 megapiksel ve yarisi kare; hasta siniflar 4-5 megapiksel ve neredeyse hicbiri kare degil. Bu, yukaridaki kisayolun ta kendisi.

## 3. Etiket gurultusu

Ayni goruntu veri setinde birden fazla kez yer aliyorsa, etiketlerinin ayni olmasi beklenir. Olmadigi durumlar etiketleyiciler arasi uyumun alt sinirini verir.

| olcut | deger |
|---|---|
| Dogrulanmis duplicate grup | 131 |
| Ayni goruntu cifti | 148 |
| Etiketi celisen cift | 43 (%29.1) |
| Ortalama celiski buyuklugu | 1.26 evre |
| Uyusma orani | %70.9 |
| Tek etiketin dogru olma tahmini | %84.2 |

Celiski buyuklugu dagilimi: 1 evre: 33, 2 evre: 9, 3 evre: 1.

### Ne anlama geliyor

Tek bir etiketin dogru olma olasiligi kabaca **%84**. Bu, mukemmel bir modelin bile bu veri setinde ulasabilecegi accuracy tavanini sinirlar. Mevcut modelin test accuracy'si 0.82 civarinda - yani tavana yakin. Kalan hatanin bir kismi modelin degil etiketlerin.

Bu tahmin yalnizca duplicate goruntulerden turedigi icin **alt sinirdir**: tekrar etmeyen goruntulerdeki gurultuyu gormuyoruz.

---

Uretim: `python scripts/confound_analysis.py`

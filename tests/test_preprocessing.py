"""scripts/preprocessing.py icin testler.

Butun boru hatti bu modulun uzerinde duruyor; bir fonksiyonun sessizce bozulmasi
tum sonuclari etkiler. Testler sentetik goruntuler kullanir, veri setine bagimli
degildir - yani veri indirilmemis bir makinede de calisir.

Kullanim:
    pytest tests/                 (pytest kuruluysa)
    python tests/test_preprocessing.py   (kurulu degilse)
"""
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from preprocessing import (HARD_BRIGHT, HARD_DARK, apply_clahe, auto_crop,  # noqa: E402
                           brightness_outliers, dhash, image_quality,
                           pad_to_square, preprocess, to_square)


def sahte_fundus(h=400, w=600, yaricap=180, parlaklik=120, seed=0):
    """Siyah zemin uzerinde dairesel, dokulu bir 'retina' uretir."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= yaricap ** 2
    doku = rng.integers(parlaklik - 30, parlaklik + 30, size=(h, w, 3), dtype=np.int16)
    img[mask] = np.clip(doku, 0, 255).astype(np.uint8)[mask]
    return img


# ------------------------------------------------------------------ auto_crop

def test_auto_crop_siyah_cerceveyi_kaldirir():
    img = sahte_fundus(400, 600, yaricap=150)
    c = auto_crop(img)
    # dairenin cevresine tam oturmali: ~2*yaricap
    assert 290 <= c.shape[0] <= 310, c.shape
    assert 290 <= c.shape[1] <= 310, c.shape


def test_auto_crop_icerigi_kaybetmez():
    img = sahte_fundus()
    onceki = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 7).sum()
    sonraki = (cv2.cvtColor(auto_crop(img), cv2.COLOR_BGR2GRAY) > 7).sum()
    assert sonraki == onceki, f"doku kaybi: {onceki} -> {sonraki}"


def test_auto_crop_tamamen_siyah_goruntuyu_bozmaz():
    # Kirpilacak bir sey yoksa orijinali dondurmeli, bos dizi degil
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    assert auto_crop(img).shape == img.shape


def test_auto_crop_cercevesiz_goruntuyu_degistirmez():
    img = np.full((80, 90, 3), 100, dtype=np.uint8)
    assert auto_crop(img).shape == img.shape


def test_auto_crop_gri_goruntu_kabul_eder():
    gri = cv2.cvtColor(sahte_fundus(), cv2.COLOR_BGR2GRAY)
    c = auto_crop(gri)
    assert c.ndim == 2 and c.shape[0] < gri.shape[0]


# ----------------------------------------------------------------- kare hale

def test_pad_to_square_kare_dondurur():
    for shape in [(100, 200, 3), (200, 100, 3), (150, 150, 3)]:
        out = pad_to_square(np.full(shape, 90, dtype=np.uint8))
        assert out.shape[0] == out.shape[1] == max(shape[:2])


def test_pad_icerigi_korur():
    img = np.full((100, 200, 3), 90, dtype=np.uint8)
    out = pad_to_square(img)
    assert (out > 7).sum() == (img > 7).sum()


def test_to_square_iki_mod_da_istenen_boyutu_verir():
    img = sahte_fundus(300, 400)
    for mode in ("pad", "squash"):
        assert to_square(img, 256, mode=mode).shape == (256, 256, 3)


def test_squash_pad_den_daha_az_siyah_birakir():
    # Gercek fundus goruntuleri gibi DIKEYDE KESILMIS bir daire uretiyoruz
    # (yaricap > yukseklik/2). Kirpma sonrasi genis bir dikdortgen kalir;
    # pad siyah bant ekler, squash eklemez.
    img = sahte_fundus(300, 500, yaricap=200)
    c = auto_crop(img)
    siyah = {m: (cv2.cvtColor(to_square(c, 256, mode=m), cv2.COLOR_BGR2GRAY) <= 7).mean()
             for m in ("pad", "squash")}
    assert siyah["squash"] < siyah["pad"], siyah


def test_to_square_bilinmeyen_mod_hata_verir():
    try:
        to_square(sahte_fundus(), 128, mode="hatali")
    except ValueError:
        return
    raise AssertionError("bilinmeyen mod icin ValueError bekleniyordu")


# ---------------------------------------------------------------------- CLAHE

def test_clahe_bicimi_ve_tipini_korur():
    img = sahte_fundus()
    out = apply_clahe(img)
    assert out.shape == img.shape and out.dtype == img.dtype


def test_clahe_kontrasti_artirir():
    # Dusuk kontrastli bir goruntude CLAHE yayilimi buyutmeli
    rng = np.random.default_rng(1)
    img = rng.integers(100, 130, size=(300, 300, 3), dtype=np.int16).astype(np.uint8)
    once = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()
    sonra = cv2.cvtColor(apply_clahe(img, clip_limit=3.0), cv2.COLOR_BGR2GRAY).std()
    assert sonra > once, f"kontrast artmadi: {once:.2f} -> {sonra:.2f}"


def test_clahe_clip_limit_arttikca_daha_agresif():
    img = sahte_fundus(seed=2)
    stds = [cv2.cvtColor(apply_clahe(img, clip_limit=cl), cv2.COLOR_BGR2GRAY).std()
            for cl in (1.0, 4.0)]
    assert stds[1] > stds[0], stds


def test_clahe_gri_goruntu_kabul_eder():
    gri = cv2.cvtColor(sahte_fundus(), cv2.COLOR_BGR2GRAY)
    out = apply_clahe(gri)
    assert out.ndim == 2 and out.shape == gri.shape


# -------------------------------------------------------------- kalite kontrol

def test_image_quality_normal_goruntuyu_gecirir():
    ok, m = image_quality(sahte_fundus(parlaklik=120))
    assert ok and not m["too_dark"] and not m["too_bright"]


def test_image_quality_neredeyse_siyahi_yakalar():
    ok, m = image_quality(np.full((100, 100, 3), 3, dtype=np.uint8))
    assert not ok and m["too_dark"]


def test_image_quality_neredeyse_beyazi_yakalar():
    ok, m = image_quality(np.full((100, 100, 3), 253, dtype=np.uint8))
    assert not ok and m["too_bright"]


def test_hard_esikler_makul_aralikta():
    assert 0 < HARD_DARK < HARD_BRIGHT < 256


# ------------------------------------------------------------- ucdeger tespiti

def test_ucdeger_temiz_dagilimda_neredeyse_hicbir_sey_isaretlemez():
    # Temiz bir normal dagilimda k=3.5 esigini asan ornek beklenir ama nadirdir.
    # "Sifir" beklemek istatistiksel olarak yanlis olurdu; %1'in altini bekliyoruz.
    rng = np.random.default_rng(0)
    oran = brightness_outliers(rng.normal(70, 8, 500)).mean()
    assert oran < 0.01, f"temiz dagilimda %{oran*100:.1f} isaretlendi"


def test_ucdeger_sabit_dagilimda_hicbir_sey_isaretlemez():
    # MAD = 0 durumu; sifira bolme olmamali
    assert brightness_outliers([50.0] * 100).sum() == 0


def test_ucdeger_gercek_sapmayi_yakalar():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(70, 5, 200), [5.0, 250.0]])
    m = brightness_outliers(v)
    assert m[-2] and m[-1], "ucdegerler isaretlenmedi"
    assert m[:-2].sum() == 0, "normal degerler yanlislikla isaretlendi"


def test_ucdeger_k_kucuklukce_daha_cok_isaretler():
    rng = np.random.default_rng(3)
    v = rng.normal(70, 8, 400)
    assert brightness_outliers(v, k=1.0).sum() >= brightness_outliers(v, k=3.5).sum()


# ----------------------------------------------------------------------- dhash

def test_dhash_ayni_goruntu_ayni_hash():
    img = sahte_fundus()
    assert dhash(img) == dhash(img.copy())


def test_dhash_farkli_goruntu_farkli_hash():
    assert dhash(sahte_fundus(seed=1)) != dhash(sahte_fundus(seed=2))


def test_dhash_yeniden_boyutlandirmaya_dayanikli():
    # Algisal hash'in amaci bu: ayni goruntunun kucultulmus kopyasini yakalamak
    img = sahte_fundus(400, 400, yaricap=150)
    kucuk = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    h1, h2 = dhash(img), dhash(kucuk)
    fark = bin(int(h1, 16) ^ int(h2, 16)).count("1")
    assert fark <= 8, f"hamming mesafesi cok buyuk: {fark}"


def test_dhash_uzunlugu_sabit():
    assert len(dhash(sahte_fundus())) == 16


# ------------------------------------------------------------------- pipeline

def test_preprocess_uctan_uca(tmp_path=None):
    tmp = pathlib.Path(tmp_path) if tmp_path else pathlib.Path(".")
    yol = tmp / "_test_fundus.png"
    cv2.imwrite(str(yol), sahte_fundus())
    try:
        img, info = preprocess(yol, size=128, use_clahe=True)
        assert img is not None and img.shape == (128, 128, 3)
        assert info["orig_width"] == 600 and info["orig_height"] == 400
        assert info["crop_width"] < info["orig_width"]
    finally:
        yol.unlink(missing_ok=True)


def test_preprocess_normalize_modu():
    yol = pathlib.Path("_test_fundus_norm.png")
    cv2.imwrite(str(yol), sahte_fundus())
    try:
        arr, _ = preprocess(yol, size=64, normalize=True)
        assert arr.dtype == np.float32 and arr.shape == (64, 64, 3)
        # ImageNet normalizasyonu sonrasi degerler kabaca [-2.5, 2.5] araliginda
        assert -3.5 < arr.min() and arr.max() < 3.5
    finally:
        yol.unlink(missing_ok=True)


def test_preprocess_okunamayan_dosyada_none_doner():
    img, info = preprocess("var_olmayan_dosya.png")
    assert img is None and info["error"] == "okunamadi"


def test_preprocess_kalitesiz_goruntuyu_eler():
    yol = pathlib.Path("_test_siyah.png")
    cv2.imwrite(str(yol), np.full((100, 100, 3), 2, dtype=np.uint8))
    try:
        img, info = preprocess(yol, quality_check=True)
        assert img is None and "karanlik" in info["error"]
    finally:
        yol.unlink(missing_ok=True)


def test_preprocess_kalite_kontrolu_kapatilabilir():
    yol = pathlib.Path("_test_siyah2.png")
    cv2.imwrite(str(yol), np.full((100, 100, 3), 2, dtype=np.uint8))
    try:
        img, _ = preprocess(yol, size=32, quality_check=False)
        assert img is not None
    finally:
        yol.unlink(missing_ok=True)


# --------------------------------------------------------- pytest'siz calistir

def main():
    testler = [(ad, fn) for ad, fn in sorted(globals().items())
               if ad.startswith("test_") and callable(fn)]
    gecen, hatalar = 0, []
    for ad, fn in testler:
        try:
            fn()
            gecen += 1
        except Exception as e:
            hatalar.append((ad, f"{type(e).__name__}: {e}"))

    for ad, hata in hatalar:
        print(f"  BASARISIZ  {ad}\n             {hata}")
    print(f"\n{gecen}/{len(testler)} test gecti")
    return 1 if hatalar else 0


if __name__ == "__main__":
    sys.exit(main())

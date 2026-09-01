"""APTOS ön isleme fonksiyonlari - ortak modul.

Is bolumundeki her uc kisinin fonksiyonlari burada toplanir; boru hatti
(`preprocess`) bunlari sirayla uygular:

    Oku -> Kalite Kontrolu -> Auto-Crop -> CLAHE -> Resize -> Normalization

Diger scriptler bu modulu import eder, kopyalamaz.
"""
import cv2
import numpy as np

# ImageNet istatistikleri - on egitimli agirliklarla uyumlu olmasi icin
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------- 1. kisi: auto-crop

def auto_crop(img, tol=7):
    """Retina disindaki siyah cerceveyi keser.

    Fundus fotograflari dairesel bir alan iceren dikdortgen karelerdir; kenarlar
    tamamen siyahtir ve hem yer kaplar hem de sonraki adimlarda (ozellikle
    CLAHE'nin histograminda) sonucu bozar.

    img: BGR veya gri numpy dizisi
    tol: bu esigin uzerindeki piksel "icerik" sayilir
    """
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not mask.any():
        return img

    rows, cols = mask.any(1), mask.any(0)
    cropped = img[np.ix_(rows, cols)]
    # Tamami esigin altinda kalan patolojik durumda orijinali koru
    return img if 0 in cropped.shape[:2] else cropped


def pad_to_square(img):
    """En-boy oranini bozmadan kareye tamamlar (resize oncesi)."""
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    top, left = (size - h) // 2, (size - w) // 2
    out = np.zeros((size, size, 3) if img.ndim == 3 else (size, size), dtype=img.dtype)
    out[top:top + h, left:left + w] = img
    return out


# ------------------------------------------------------------- 2. kisi: CLAHE

def apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """Kontrasti yerel olarak artirir, damarlari belirginlestirir.

    CLAHE yalnizca LAB uzayinin L (parlaklik) kanalina uygulanir. Uc RGB kanalina
    ayri ayri uygulamak renk dengesini bozar ve fundus goruntulerinde yapay bir
    renklenme birakir.

    Not: bu fonksiyon auto_crop'tan SONRA cagrilmalidir. Siyah cerceve
    kirpilmadan uygulanirsa histogram o genis siyah alan yuzunden bozulur.
    """
    if img.ndim == 2:
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ----------------------------------------------------- 3. kisi: kalite kontrol

def image_quality(img, dark_threshold=12.0, bright_threshold=240.0):
    """Goruntunun temel kalite olculeri ve gecip gecmedigi.

    Doner: (gecti_mi, olculer sozlugu)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean = float(gray.mean())

    metrics = {
        "brightness": mean,
        "std": float(gray.std()),
        "too_dark": mean < dark_threshold,
        "too_bright": mean > bright_threshold,
    }
    metrics["ok"] = not (metrics["too_dark"] or metrics["too_bright"])
    return metrics["ok"], metrics


def dhash(img, size=8):
    """Algisal hash - ayni goruntunun yeniden boyutlandirilmis/sikistirilmis
    kopyalarini yakalar. Bit bit farkli olan dosyalar ayni hash'i verebilir,
    bu yuzden duplicate tespitinde dosya hash'inden daha kullanislidir.

    Doner: 16 haneli onaltilik dize
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return f"{int(''.join('1' if b else '0' for b in bits.flatten()), 2):016x}"


# ------------------------------------------------------------------ boru hatti

def preprocess(path, size=512, use_clahe=True, normalize=False,
               clip_limit=2.0, quality_check=True):
    """Tam boru hatti: oku -> kalite -> crop -> CLAHE -> resize -> normalize.

    path: goruntu dosyasi
    size: cikti kenar uzunlugu
    use_clahe: CLAHE uygulansin mi (karsilastirma kosulari icin kapatilabilir)
    normalize: True ise float32 ve ImageNet normalizasyonu doner (model girdisi);
               False ise uint8 BGR doner (diske yazmak icin)

    Doner: (goruntu | None, bilgi sozlugu). Okunamayan veya kaliteyi geciremeyen
    goruntulerde goruntu None doner ve sebep bilgi["error"] icinde olur.
    """
    info = {"path": str(path)}

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        info["error"] = "okunamadi"
        return None, info

    info["orig_height"], info["orig_width"] = img.shape[:2]

    if quality_check:
        ok, metrics = image_quality(img)
        info.update(metrics)
        if not ok:
            info["error"] = "asiri karanlik" if metrics["too_dark"] else "asiri parlak"
            return None, info

    img = auto_crop(img)
    info["crop_height"], info["crop_width"] = img.shape[:2]

    if use_clahe:
        img = apply_clahe(img, clip_limit=clip_limit)

    img = cv2.resize(pad_to_square(img), (size, size), interpolation=cv2.INTER_AREA)

    if normalize:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - IMAGENET_MEAN) / IMAGENET_STD, info

    return img, info

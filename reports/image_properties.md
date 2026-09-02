# Image Properties Summary

Raw APTOS-2019 fundus photographs, n=3662.

## Readability and colour mode

| check | result |
|---|---|
| Readable | 3662 / 3662 |
| Colour mode | RGB: 3662 |
| Channels | 3: 3662 |

Colour mode is judged by content rather than file format: an image whose three channels are identical is grayscale even if it was stored as RGB.

## Resolution

| measure | min | median | max |
|---|---|---|---|
| Width | 474 | 2144 | 4288 |
| Height | 358 | 1536 | 2848 |
| Megapixels | 0.17 | 3.15 | 12.21 |
| Aspect ratio | 1.000 | 1.333 | 1.506 |

There are **17 distinct resolutions**. The most common:

| resolution | count | share |
|---|---|---|
| 1050x1050 | 974 | 26.6% |
| 2416x1736 | 638 | 17.4% |
| 2588x1958 | 533 | 14.6% |
| 3216x2136 | 410 | 11.2% |
| 2048x1536 | 351 | 9.6% |
| 819x614 | 287 | 7.8% |
| 3388x2588 | 141 | 3.9% |
| 1504x1000 | 92 | 2.5% |

974 images (26.6%) are already square; the rest are brought to a square before resizing. This much variation in resolution makes a fixed input size mandatory.

## Pixel statistics

| measure | min | mean | max |
|---|---|---|---|
| Brightness | 15.0 | 66.8 | 129.6 |
| Contrast (std) | 9.6 | 38.8 | 75.8 |
| Black pixel ratio | 0.000 | 0.232 | 0.528 |

Brightness never exceeds 130 in this dataset; fundus photographs are inherently dark. A fixed "too bright" cutoff such as 240 would never fire here, which is why the quality report also uses a distribution-based outlier test.

## Auto-crop saving

Removing the black frame outside the retina discards **10.4%** of the area on average (min 0.0%, max 39.9%).

---

Charts: `reports/figures/06_image_properties.png`. Per-image detail: BigQuery `aptos_image_stats`.

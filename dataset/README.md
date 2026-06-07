# Dataset Folder

Place your image dataset here. The training script validates the folders before
training and stops with a clear message if images are missing.

## 3-Class Dataset

```text
dataset/
  train/
    mask/
    no_mask/
    incorrect_mask/
  val/
    mask/
    no_mask/
    incorrect_mask/
  test/
    mask/
    no_mask/
    incorrect_mask/
```

## 2-Class Dataset

```text
dataset/
  train/
    mask/
    no_mask/
  val/
    mask/
    no_mask/
  test/
    mask/
    no_mask/
```

## Notes

- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
- Keep class folder names exactly as shown.
- The `test` split is recommended for final results. If it is missing or empty,
  training still runs but saves validation evaluation instead.
- Do not place copyrighted datasets in this repository unless you have permission.

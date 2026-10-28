# Lab1 Computer Vision — CSE480/CSE482

**Ainour Hesham Elshamy — 23P0111**

This project implements the four experiments from `assignment/lab01_tasks.pdf` using the six supplied images. The notebook and Python script have been executed and verified locally; all result figures and processed images are included.

## Files

- `Lab1_Computer_Vision_23P0111.ipynb` — **executed notebook** with displayed output figures. Works in Colab after you upload the images ZIP when prompted.
- `code/run_lab.py` — runs all four tasks and exports PNGs.
- `data/` — original six supplied images.
- `outputs/` — mask, blend and processed image PNGs plus assembled result figures.
- `report/` — eight-page Word and PDF reports with cover page, table of contents, methods and execution results.
- `assignment/` — original lab assignment instructions.
- `lab1_images.zip` (separate download) — convenient image bundle for uploading in Colab.

## Run in Google Colab

1. Visit https://colab.research.google.com/ and use **File > Upload notebook** to upload `Lab1_Computer_Vision_23P0111.ipynb`.
2. Select **Runtime > Run all**.
3. If prompted, upload `lab1_images.zip` from the separate download. The notebook extracts the six images into `data/`.
4. View results under each task and in the runtime `outputs/` folder. To save outputs, download them before closing the runtime.

The notebook is already executed, so output figures are embedded if you simply open it in Jupyter or another compatible viewer.

## Run locally

From the project directory:

```bash
pip install -r requirements.txt
python code/run_lab.py
```

**Execution environment verified:** OpenCV 4.13.0; NumPy 2.3.5. All four tasks completed successfully.

## Notes

- Task 1 uses a red-dominance threshold; this does not perfectly segment apple objects.
- Task 2 resizes `cameraman.jpg` to match `zebra.png` (425 x 313 pixels), altering the cameraman aspect ratio.
- Task 3 applies a Gaussian-smoothed circular brightness mask.
- Task 4 swaps the zebra's green and red channel data. Because the zebra photograph is almost grayscale, the visual change is subtle; an optional colour-apple comparison shows a much stronger effect.
- The two cat photographs are included in an optional Task 2 blending example.
- The captured report figures are **real local Python execution results**, not screenshots of a Google Colab session.

## References

- Instructor's CSE480/CSE482 Lab Assignment #01 (provided in `assignment/`)
- OpenCV: https://docs.opencv.org/4.12.0/df/d9d/tutorial_py_colorspaces.html
- OpenCV image blending: https://docs.opencv.org/4.x/d5/dc4/tutorial_adding_images.html
- OpenCV Gaussian filtering: https://docs.opencv.org/4.x/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
- OpenCV split / merge: https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html

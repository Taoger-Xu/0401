# Appendix Figures

## Figure 1: Low-frequency residual visualization

`figs/figure1.pdf` qualitatively illustrates how the low-frequency residual identifies informative visual tokens. Each row contains an original image, its PCA-RGB visual features, the corresponding DCT low-pass reconstruction, and the residual heatmap. Homogeneous regions are reconstructed well and have low residuals, whereas object boundaries and fine-grained details produce higher residuals and should be preferentially retained.

## Figure 2: Examples under a 128-token visual budget

`figs/figure2.pdf` presents representative correct predictions under a visual budget of 128 tokens. The examples cover nine benchmarks: GQA, MMBench-EN, MME-P, MMStar, POPE, ScienceQA-IMG, TextVQA, VizWiz, and OCRBench. They demonstrate that the pruned visual tokens preserve information needed for general visual understanding, object recognition, scientific reasoning, and text/OCR tasks.

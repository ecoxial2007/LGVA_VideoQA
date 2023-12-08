

# GLIP Model Setup and Usage Guide

## Environment Setup

1. **Install GLIP Environment**: Follow the instructions in the [GLIP GitHub repository](https://github.com/microsoft/GLIP) readme. Note that MaskRCNN requires strict versions of Torch and CUDA.

## Model Download

2. **Download GLIP Model**: Obtain the GLIP model from its model zoo. We are using `glip_large_model.pth`.

## Video Processing

3. **Install skvideo for Video Processing**: Run `pip install skvideo` to install.

## Extract Bounding Boxes

4. **Extract Bounding Boxes**: Modify your video path and execute `extract_glip_bboxes.py`. The bbox results will be saved in a JSON file.

## Notes

- **Data Labels**: We did not use VG (Visual Genome) bbox labels but used COCO labels (80 classes).
- **Batch Processing**: We did not implement batch processing (due to laziness).
- **Post-Processing**: After obtaining bbox, we crop the images and use CLIP to extract features.

**Caution**: The entire process can be time-consuming. Good luck!

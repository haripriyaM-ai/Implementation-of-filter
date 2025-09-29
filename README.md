# EXP 05 : Implementation-of-filter
### NAME : HARI PRIYA M
### REG NO : 212224240047
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1 : Load and Prepare Image
  - Read the image using OpenCV.
  - Convert it from BGR to RGB for correct display.

### Step2 : Apply Average and Weighted Filters
   - Use convolution kernels to smooth the image.
   - Average filter gives uniform smoothing, weighted filter gives more importance to center pixels.


### Step3 : Apply Gaussian and Median Blur
 - Gaussian blur: smooths image with a Gaussian kernel for natural blur.
 - Median blur: removes noise by replacing each pixel with the median of its neighbors.

### Step4 : Apply Laplacian for Edge Detection
 - Use a custom kernel or the Laplacian operator to highlight edges.

### Step5 : Display Images
- Use Matplotlib to show the original and filtered images for comparison.
</br>

## Program:
### Developed By   : HARI PRIYA M
### Register Number: 212224240047
</br>

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
image1 = cv2.imread("/content/3d-countryside-landscape-with-tree-against-blue-sky.jpg")
image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
```

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
kernel_avg = np.ones((11,11), np.float32)/121  # 11*11=121
avg_filtered = cv2.filter2D(image_rgb, -1, kernel_avg)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(avg_filtered)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```
ii) Using Weighted Averaging Filter
```Python
kernel_weighted = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
weighted_filtered = cv2.filter2D(image_rgb, -1, kernel_weighted)

plt.imshow(weighted_filtered)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```
iii) Using Gaussian Filter
```Python
gaussian_blur = cv2.GaussianBlur(image_rgb, (33,33), 0)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```
iv)Using Median Filter
```Python
median_blur = cv2.medianBlur(image_rgb, 13)
plt.imshow(median_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
kernel_laplacian = np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
laplacian_custom = cv2.filter2D(image_rgb, -1, kernel_laplacian)
plt.imshow(laplacian_custom)
plt.title("Custom Laplacian Kernel")
plt.axis("off")
plt.show()
```
ii) Using Laplacian Operator
```Python
laplacian_op = cv2.Laplacian(image_rgb, cv2.CV_64F)
laplacian_op = cv2.convertScaleAbs(laplacian_op)  # Convert to uint8
plt.imshow(laplacian_op)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
<img width="1000" height="445" alt="Screenshot 2025-09-29 105533" src="https://github.com/user-attachments/assets/daf78be9-0307-4304-beb9-4709f1a3cb0d" />
<br>


ii)Using Weighted Averaging Filter
<br>
<img width="500" height="500" alt="Screenshot 2025-09-29 105545" src="https://github.com/user-attachments/assets/19c14fe9-b574-4687-8f0b-8554e2e6e96d" />

<br>

iii)Using Gaussian Filter
<br>

<img width="500" height="500" alt="Screenshot 2025-09-29 105555" src="https://github.com/user-attachments/assets/2116050b-6ac8-4229-9d04-24fe42520d1c" />
<br>


iv) Using Median Filter
<br>

<img width="500" height="500" alt="Screenshot 2025-09-29 105604" src="https://github.com/user-attachments/assets/f13a6300-bbb2-4bc8-93ee-9eedc93b1b78" />

<br>

### 2. Sharpening Filters

i) Using Laplacian Kernal

<img width="500" height="500" alt="Screenshot 2025-09-29 105614" src="https://github.com/user-attachments/assets/02334d46-3379-48d9-afb4-566faa695edb" />
<br>


ii) Using Laplacian Operator
<br>

<img width="500" height="500" alt="Screenshot 2025-09-29 105620" src="https://github.com/user-attachments/assets/92ea6b66-fbf6-4a94-80e0-5f28ffc3c0a0" />
<br>


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.

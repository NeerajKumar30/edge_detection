import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import os

windowName = "Edge Detection Toolkit"

# root window minimize
Tk().withdraw()

# open the file
file_path = askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
)

if not file_path:
    raise Exception("No image selected")

# Load grayscale image
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise Exception("Image not found. Check the file path!")

# Nothing function for trackbar callback
def nothing(x):
    pass

# Create window and trackbars
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
# Methods: 0=Sobel, 1=Laplacian, 2=Canny, 3=Prewitt, 4=Scharr
cv2.createTrackbar('Method', windowName, 0, 4, nothing)
cv2.createTrackbar('Threshold1', windowName, 100, 255, nothing)
cv2.createTrackbar('Threshold2', windowName, 150, 255, nothing)  # used by Canny
cv2.createTrackbar('Blur', windowName, 0, 20, nothing)  # kernel size (odd only)
cv2.createTrackbar('Binary (0/1)', windowName, 1, 1, nothing)  # 1: show binary(thresholded), 0: show magnitude/grad

method_name = ["Sobel", "Laplacian", "Canny", "Prewitt", "Scharr"]

# Prewitt kernels (float32)
#detecting vertical edges
prewitt_kx = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]], dtype=np.float32)

#detecting horizontal edges
prewitt_ky = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], dtype=np.float32)

# Output folder for saved images
out_dir = os.path.join(os.path.dirname(file_path), "edge_outputs")
os.makedirs(out_dir, exist_ok=True)

while True:
    # Get trackbar values
    method = cv2.getTrackbarPos('Method', windowName)
    t1 = cv2.getTrackbarPos('Threshold1', windowName)
    t2 = cv2.getTrackbarPos('Threshold2', windowName)
    blur_strength = cv2.getTrackbarPos('Blur', windowName)
    binary_mode = cv2.getTrackbarPos('Binary (0/1)', windowName)

    # Ensure kernel size is odd and at least 1
    if blur_strength <= 0:
        ksize = 1
    else:
        ksize = blur_strength if (blur_strength % 2 == 1) else blur_strength + 1

    # Apply blur if selected
    if ksize > 1:
        img_proc = cv2.GaussianBlur(image, (ksize, ksize), 0)
    else:
        img_proc = image.copy()

    # Prepare variable for display
    display = None

    try:
        if method == 0:  # Sobel
            gx = cv2.Sobel(img_proc, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_proc, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            if mag.max() > 0:
                mag_norm = (mag / mag.max()) * 255.0
            else:
                mag_norm = mag
            mag_uint8 = cv2.convertScaleAbs(mag_norm)
            display = (cv2.threshold(mag_uint8, t1, 255, cv2.THRESH_BINARY)[1]
                       if binary_mode else mag_uint8)

        elif method == 1:  # Laplacian
            lap = cv2.Laplacian(img_proc, cv2.CV_64F)
            lap_abs = np.abs(lap)
            if lap_abs.max() > 0:
                lap_norm = (lap_abs / lap_abs.max()) * 255.0
            else:
                lap_norm = lap_abs
            lap_uint8 = cv2.convertScaleAbs(lap_norm)
            display = (cv2.threshold(lap_uint8, t1, 255, cv2.THRESH_BINARY)[1]
                       if binary_mode else lap_uint8)

        elif method == 2:  # Canny
            display = cv2.Canny(img_proc, t1, t2)

        elif method == 3:  # Prewitt (robust)
            # Convert to float32 source for filtering
            src_f = img_proc.astype(np.float32)
            # Use CV_32F ddepth (-1 returns same depth as src, which is float32)
            px = cv2.filter2D(src_f, ddepth=cv2.CV_32F, kernel=prewitt_kx)
            py = cv2.filter2D(src_f, ddepth=cv2.CV_32F, kernel=prewitt_ky)
            # magnitude using OpenCV helper (handles shapes safely)
            pmag = cv2.magnitude(px, py)  # float32
            # normalize to 0-255 robustly (avoid division by zero)
            pmag_norm = np.zeros_like(pmag, dtype=np.float32)
            if np.isfinite(pmag).any():
                cv2.normalize(pmag, pmag_norm, 0, 255, cv2.NORM_MINMAX)
            # convert to uint8
            p_uint8 = cv2.convertScaleAbs(pmag_norm)
            display = (cv2.threshold(p_uint8, t1, 255, cv2.THRESH_BINARY)[1]
                       if binary_mode else p_uint8)

        else:  # method == 4 Scharr
            sx = cv2.Scharr(img_proc, cv2.CV_64F, 1, 0)
            sy = cv2.Scharr(img_proc, cv2.CV_64F, 0, 1)
            smag = np.sqrt(sx * sx + sy * sy)
            if smag.max() > 0:
                snorm = (smag / smag.max()) * 255.0
            else:
                snorm = smag
            s_uint8 = cv2.convertScaleAbs(snorm)
            display = (cv2.threshold(s_uint8, t1, 255, cv2.THRESH_BINARY)[1]
                       if binary_mode else s_uint8)

    except Exception as e:
        # If anything goes wrong, show a text overlay with the error (helps debug)
        display = np.zeros_like(image)
        err_msg = f"{type(e).__name__}: {str(e)}"
        cv2.putText(display, "ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        # second line (clamp to image width)
        cv2.putText(display, err_msg[:image.shape[1]-10], (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        print("Internal error in processing:", err_msg)  # also print to console for details

    # Ensure display is valid before imshow
    if display is None:
        display = np.zeros_like(image)

    # Show original and edges
    imgTitle = "Original Image"
    if cv2.getWindowProperty(imgTitle, cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow(imgTitle, cv2.WINDOW_NORMAL)
    cv2.imshow(imgTitle, image)

    window_title = f"{windowName} - {method_name[method]}"
    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, display)

    # Key handling: 'q' to quit, 's' to save current edge image
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{method_name[method]}_{ts}.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, display)
        print(f"Saved: {fpath}")

cv2.destroyAllWindows()

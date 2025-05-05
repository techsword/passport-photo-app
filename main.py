import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import io
import os # Required to find the cascade file

# --- Streamlit Page Configuration (MUST BE FIRST st command) ---
st.set_page_config(page_title="Passport Photo Generator", layout="centered")

# --- Configuration ---
# Define passport photo specifications (width, height in pixels at 300 DPI)
# Also define relative head height range (min_head_%, max_head_%)
# And target eye line position (% from bottom - approximate)

# Common DPI for calculations
DPI = 300

# Function to convert inches to pixels
def inches_to_pixels(inches, dpi=DPI):
    return int(inches * dpi)

# Function to convert mm to pixels
def mm_to_pixels(mm, dpi=DPI):
    # 1 inch = 25.4 mm
    return int((mm / 25.4) * dpi)

PASSPORT_SPECS = {
    "US": {
        "width_px": inches_to_pixels(2),   # 600
        "height_px": inches_to_pixels(2),  # 600
        "head_min_px": inches_to_pixels(1),    # 300 (Chin to Top of Hair)
        "head_max_px": inches_to_pixels(1 + 3/8), # 412
        "eye_line_from_bottom_min_px": inches_to_pixels(1 + 1/8), # 337
        "eye_line_from_bottom_max_px": inches_to_pixels(1 + 3/8), # 412
        "name": "US Passport (2x2 inch)"
    },
    "UK": {
        "width_px": mm_to_pixels(35),   # approx 413
        "height_px": mm_to_pixels(45),  # approx 531
        "head_min_px": mm_to_pixels(29),  # approx 342 (Chin to Crown)
        "head_max_px": mm_to_pixels(34),  # approx 401
        "name": "UK Passport (35x45 mm)"
    },
    "EU": {
        "width_px": mm_to_pixels(35),   # approx 413
        "height_px": mm_to_pixels(45),  # approx 531
        "head_min_px": mm_to_pixels(32),  # approx 378 (Chin to Crown)
        "head_max_px": mm_to_pixels(36),  # approx 425
        "name": "EU Schengen (35x45 mm)"
    }
}

# --- Face Detection Setup ---
# Ensure the path to the cascade file is correct. Put it in the same folder as the script.
CASCADE_FILE = "haarcascade_frontalface_default.xml"

# Load the cascade only once using Streamlit's caching for efficiency
# Check for file existence *inside* the cached function or just before use
@st.cache_resource
def load_cascade():
    if not os.path.exists(CASCADE_FILE):
        # Raise an error here to be caught later, or return None
        # st.error() cannot be reliably used inside @st.cache_resource for initial page load
        print(f"Error: Haar Cascade file not found at {os.path.abspath(CASCADE_FILE)}") # Log to console
        return None # Indicate failure
    try:
        cascade = cv2.CascadeClassifier(CASCADE_FILE)
        if cascade.empty():
             print(f"Error: Failed to load Haar Cascade from {CASCADE_FILE}") # Log to console
             return None # Indicate failure
        return cascade
    except Exception as e:
        print(f"Error loading Haar Cascade file: {e}") # Log to console
        return None # Indicate failure

face_cascade = load_cascade()

# --- Face Detection Function ---
def detect_face(image_pil, cascade):
    """Detects the largest face in a PIL image using the provided cascade."""
    if cascade is None:
        st.error("Face detection model (Haar Cascade) could not be loaded. Cannot process image.")
        return None # Cascade not available

    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    image_np = np.array(image_pil)
    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60) # Min face size to detect (adjust if needed)
    )

    if len(faces) == 0:
        return None # No face detected

    # Find the largest face based on area (w*h)
    largest_face = max(faces, key=lambda item: item[2] * item[3])
    return largest_face # Returns (x, y, w, h) of the largest face

# --- Image Processing Function ---
def create_passport_photo(image_pil, selected_format, cascade):
    """Crops and resizes the image according to passport specs."""
    specs = PASSPORT_SPECS[selected_format]
    target_w_px = specs["width_px"]
    target_h_px = specs["height_px"]
    target_aspect = target_w_px / target_h_px
    target_head_avg_px = (specs["head_min_px"] + specs["head_max_px"]) / 2

    # 1. Detect Face
    face = detect_face(image_pil, cascade) # Pass the loaded cascade
    if face is None:
        # Error/Warning is handled by detect_face or shown after this returns None
        return None

    x, y, w, h = face
    face_center_x = x + w / 2
    face_center_y = y + h / 2

    # --- Advanced Cropping Logic ---
    # Estimate head height based on detected face height (heuristic)
    estimated_head_h = h / 0.70 # Adjust divisor if needed
    # Estimate center Y aiming slightly above face center towards crown
    head_center_y = y + h * 0.40 # Adjust multiplier if needed

    # Calculate the scale factor needed
    scale = target_head_avg_px / estimated_head_h

    # Calculate the dimensions of the crop box in the *original* image coordinates
    crop_width_orig = target_w_px / scale
    crop_height_orig = target_h_px / scale

    # Calculate crop box coordinates centered on the estimated head center
    crop_x1 = int(face_center_x - crop_width_orig / 2)
    crop_y1 = int(head_center_y - crop_height_orig / 2)

    crop_x2 = int(crop_x1 + crop_width_orig)
    crop_y2 = int(crop_y1 + crop_height_orig)

    # 2. Validate and Adjust Crop Box
    img_w, img_h = image_pil.size
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

    # Recalculate width/height after clamping
    final_crop_w = crop_x2 - crop_x1
    final_crop_h = crop_y2 - crop_y1

    if final_crop_w < 50 or final_crop_h < 50 :
         st.warning("Face detected is too close to the edge or image is too small for required crop.")
         return None

    # 3. Crop the Image
    cropped_image_pil = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # 4. Resize the Cropped Image to Exact Passport Dimensions
    try:
        # Use LANCZOS (previously ANTIALIAS) for high-quality downsampling
        final_image = cropped_image_pil.resize((target_w_px, target_h_px), Image.Resampling.LANCZOS)
    except Exception as e:
        st.error(f"Error during image resizing: {e}")
        return None

    return final_image

# --- Streamlit App UI ---
# Title and intro placed *after* page config and potentially problematic setup code
st.title("Passport Photo Generator 📷")
st.write("Upload a photo and select a format to generate a compliant passport photo.")

# Check if cascade loaded correctly *before* allowing uploads/processing
if face_cascade is None:
    st.error(f"Critical Error: Could not load face detection model (haarcascade_frontalface_default.xml). Please ensure the file exists in the same directory as the script and is not corrupted.")
    st.stop() # Stop the app if the essential cascade isn't loaded

# Format Selection
format_options = list(PASSPORT_SPECS.keys())
format_display_names = [PASSPORT_SPECS[k]["name"] for k in format_options]
selected_format_display = st.selectbox(
    "Select Passport Photo Format:",
    format_display_names
)
# Get the key ('US', 'UK', 'EU') from the display name
selected_format_key = format_options[format_display_names.index(selected_format_display)]


# File Uploader
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded file into a PIL Image
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes))

        # --- Image Orientation Correction (Optional but Recommended) ---
        try:
            original_image = ImageOps.exif_transpose(original_image)
        except Exception:
            pass # Ignore if EXIF data is missing or corrupt


        st.image(original_image, caption="Original Uploaded Photo", use_column_width=True)

        st.write("---") # Separator

        # Process Button
        if st.button(f"Generate {selected_format_key} Passport Photo"):
            with st.spinner("Processing image..."):
                # Pass the loaded cascade to the processing function
                passport_photo = create_passport_photo(original_image, selected_format_key, face_cascade)

            if passport_photo:
                st.success("Passport photo generated successfully!")
                st.image(passport_photo, caption=f"Generated {selected_format_key} Passport Photo ({PASSPORT_SPECS[selected_format_key]['width_px']}x{PASSPORT_SPECS[selected_format_key]['height_px']}px)", use_column_width=True)

                # Prepare image for download
                buf = io.BytesIO()
                passport_photo.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download Passport Photo (PNG)",
                    data=byte_im,
                    file_name=f"passport_photo_{selected_format_key.lower()}.png",
                    mime="image/png"
                )
            else:
                # Display a generic failure message if create_passport_photo returned None
                # Specific warnings/errors might have already been shown inside the function
                 if detect_face(original_image, face_cascade) is None and face_cascade is not None:
                     st.warning("Could not detect a face in the uploaded image. Please try another photo with a clear, frontal view of the face.")
                 # else: a different error occurred (cascade loading, cropping issue, resize issue) - message likely shown already

    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")
        st.error("Please ensure the uploaded file is a valid image (JPG, PNG).")

else:
    st.info("Please upload an image file.")

st.write("---")
st.markdown("""
**Disclaimer:** This tool provides automated cropping based on standard face detection and predefined specifications.
It does **NOT** guarantee compliance with all passport photo rules (e.g., background color, facial expression, lighting, photo age, no glasses).
**Always double-check the generated photo against the official requirements of the issuing country before submission.**
""")

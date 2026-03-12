# Passport Photo Generator

A web-based tool that automatically detects faces in uploaded photos and crops/resizes them to meet passport photo specifications for multiple formats.

## Supported Photo Formats

| Format | Dimensions | Head Height Range |
|--------|-----------|-------------------|
| US Passport | 2x2 inches (600x600 px) | 1 - 1 3/8 inches |
| UK Passport | 35x45 mm (~413x531 px) | 29 - 34 mm |
| EU Schengen | 35x45 mm (~413x531 px) | 32 - 36 mm |
| Photo (3.5x4.5 cm) | 3.5x4.5 cm (~413x531 px) | 29 - 34 mm |

All images are rendered at 300 DPI.

## Prerequisites

- Python 3.11+
- `haarcascade_frontalface_default.xml` file in the project root directory

## Installation

```bash
# Install dependencies
pip install -e .
```

Or install dependencies manually:

```bash
pip install numpy opencv-python pillow streamlit
```

## Running the App

```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`.

## Usage

1. **Select a format** - Choose your target passport photo format from the dropdown (US, UK, EU Schengen, or 3.5x4.5 cm).
2. **Upload a photo** - Click "Choose a photo" and select a JPG, JPEG, or PNG image with a clear, frontal view of a face.
3. **Preview** - The original uploaded photo will be displayed.
4. **Generate** - Click the "Generate" button to process the image.
5. **Download** - If successful, the generated passport photo will appear with a download button for the PNG file.

## Tips for Best Results

- Use a photo with a **clear, well-lit face** looking directly at the camera
- Ensure the face is **not too close to the edges** of the photo
- Avoid photos with multiple faces (the largest detected face will be used)
- Use high-resolution source images for better output quality
- The background of the original photo is preserved (this tool does not change background color)

## Limitations

- Does not validate background color, facial expression, lighting, or photo age
- Does not handle glasses, head coverings, or other accessories
- Face detection may fail on low-quality, side-profile, or heavily filtered photos
- Always verify the generated photo against official requirements before submission

## Project Structure

```
passport-photo-app/
├── main.py                             # Main application (Streamlit)
├── haarcascade_frontalface_default.xml # OpenCV face detection model
├── pyproject.toml                      # Project configuration and dependencies
├── .python-version                     # Python version pin (3.11)
├── LICENSE                             # MIT License
└── README.md                           # This file
```

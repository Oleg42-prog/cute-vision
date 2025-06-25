# Cute Vision

A simple and intuitive Python library for computer vision tasks, designed to make prototyping and hypothesis testing effortless.

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)

## 🎯 Features

- **Multiple Camera Sources**: Support for webcams, RTSP streams, video files, screen capture, and static images
- **Simple API**: Intuitive interface with context managers and iterators
- **Image Processing**: Built-in filters and transformations
- **Easy Visualization**: Quick frame display and video playback
- **Pipeline Integration**: Seamless integration with pipes-and-filters for complex processing workflows
- **Rapid Prototyping**: Perfect for testing computer vision ideas quickly

## 📦 Installation

### From Source
```bash
git clone https://github.com/Oleg42-prog/cute-vision.git
cd cute-vision
pip install .
```

### Dependencies
The library requires Python 3.10+ and the following packages:
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `mss>=9.0.0`
- `pipes-and-filters`

## 🚀 Quick Start

### Basic Camera Usage

```python
import cv2
from cute_vision.cameras import DeviceCamera

# Simple frame capture
camera = DeviceCamera(device_index=0)
camera.connect()

while True:
    frame = camera.fetch_frame()
    if frame is not None:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.disconnect()
```

### Using Context Manager (Recommended)

```python
from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer

viewer = Viewer('My Camera')

with DeviceCamera(device_index=0) as camera:
    for frame in camera:
        if not viewer(frame):
            break
```

### Simple Iterator

```python
from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer

viewer = Viewer('Camera Stream')

for frame in DeviceCamera(device_index=0).frames():
    if not viewer(frame):
        break
```

## 📖 Examples

### Different Camera Types

#### Webcam
```python
from cute_vision.cameras import DeviceCamera
camera = DeviceCamera(device_index=0)
```

#### RTSP Stream
```python
from cute_vision.cameras import RTSPCamera
camera = RTSPCamera(url="rtsp://example.com/stream")
```

#### Video File
```python
from cute_vision.cameras import VideoCamera
camera = VideoCamera(path="video.mp4")
```

#### Screen Capture
```python
from cute_vision.cameras import ScreenCamera
camera = ScreenCamera(monitor_index=0)
```

### Image Processing Pipeline

```python
from pipes_and_filters import Pipeline, Pipe
from cute_vision.cameras import DeviceCamera
from cute_vision.filters import bgr_to_grayscale, flip_horizontal
from cute_vision.visualization import Viewer

pipeline = Pipeline(
    source=DeviceCamera(device_index=0).frames(),
    pipe=Pipe(
        flip_horizontal,
        bgr_to_grayscale
    )
)

viewer = Viewer('Processed Stream')
viewer.play(pipeline())
```

### Person Detection (Optional YOLO Integration)

```python
from ultralytics import YOLO  # pip install ultralytics
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer
from cute_vision.filters.reduces.xyxy_to_points import reduce_xyxy_to_points
from cute_vision.visualization.draw.points import draw_np_points
from cute_vision.visualization.draw.colors import BGR_RED
from cute_vision.utils import first

model = YOLO('yolov8n.pt')

splitter = Splitter(
    input_pipe=Pipe(
        lambda x: model(x, classes=[0], conf=0.6, verbose=False),
        first
    ),
    outputs_pipes=[
        Pipe(
            lambda result: result.boxes.xyxy.cpu().numpy(),
            lambda xyxy: reduce_xyxy_to_points(xyxy, 0.5, 0.5)
        ),
        Pipe(
            lambda result: result.plot()
        )
    ]
)

flow = Flow(
    source=DeviceCamera(device_index=0).frames(),
    splitter=splitter,
    sink=lambda points, frame: draw_np_points(frame, points, BGR_RED)
)

viewer = Viewer('Person Detection')
viewer.play(flow())
```

## 🏗️ Architecture

### Camera Classes

All camera classes inherit from `AbstractCamera` and provide:
- `connect()` / `disconnect()` methods
- `fetch_frame()` for getting frames
- Context manager support (`with` statement)
- Iterator interface for easy frame streaming
- Frame dimensions properties

### Available Camera Types

- **DeviceCamera**: USB/built-in webcams
- **RTSPCamera**: IP camera streams
- **VideoCamera**: Video file playback
- **ScreenCamera**: Desktop screen capture
- **ImageCamera**: Static image loading

### Filters and Transformations

- **Colorspace**: BGR to grayscale conversions
- **Flips**: Horizontal and vertical flipping
- **Reduces**: Bounding box to point conversions

### Visualization

- **Viewer**: Simple frame display with controls
- **Drawing utilities**: Points, colors, and annotations

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/your-username/cute-vision.git
cd cute-vision
pip install -e .[dev]
```

### Build Package

```bash
pip install build
python -m build
```

### Install Built Package

```bash
pip install dist/cute_vision-0.1.0-py3-none-any.whl
```

## ⚠️ Current Status

This project is in **alpha** stage. APIs may change between versions. Use with caution in production environments.

## 🤝 Contributing

Contributions are welcome! This library is designed for rapid prototyping and experimentation in computer vision. If you have ideas for improvements or new features, please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## 👤 Author

**Oleg Dudnik**
- Email: Oleggelo86@gmail.com

---

Perfect for computer vision developers who need a simple, clean interface for camera handling and basic image processing pipelines! 🎯

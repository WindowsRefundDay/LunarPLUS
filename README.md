# LunarPLUS
LunarPLUS is an advanced AI-powered aimbot with enhanced features, performance optimizations, and training capabilities.

## Features

### Core Features
- **Advanced AI Detection**: Uses YOLOv8 with optimized performance
- **CUDA Acceleration**: Full CUDA support with mixed precision training
- **TensorRT Optimization**: Enhanced inference speed with TensorRT
- **Smart Targeting**: 
  - Configurable aim height
  - Automatic closest target selection
  - Smooth aim interpolation
- **TriggerBot**: Auto-fire when crosshair is on target
- **Real-time Performance**: Uncapped FPS with performance optimizations
- **Auto Screen Resolution**: Automatic display configuration

### Advanced Controls
- **Hotkeys**:
  - `F1`: Toggle aimbot on/off
  - `F2`: Exit program
  - Right Mouse Button: Activate targeting
  - `C`: Capture training data (in training mode)
- **Training Mode**:
  - `python lunar.py collect_data`: Enter training data collection mode
  - Press 'C' while targeting to capture training samples
  - `python train.py`: Train model on collected data

### Performance Features
- **CUDA Optimizations**:
  - TensorRT acceleration
  - torch.compile optimization
  - Mixed precision inference
  - cuDNN benchmark mode
  - TF32 acceleration
- **Memory Optimizations**:
  - Pre-allocated frame buffers
  - Optimized numpy operations
  - Efficient screen capture
- **Processing Optimizations**:
  - Vectorized calculations
  - Bit-shift operations
  - Asynchronous key detection
  - Smooth FPS calculation

## Technical Specifications
- PyTorch 2.5.1 with CUDA 11.8
- TorchVision 0.20.1
- Ultralytics 8.0.0+ (YOLOv8)
- OpenCV 4.9.0
- MSS for efficient screen capture
- CUDA-optimized neural network processing
- Custom Win32 API integration

## Installation

### Prerequisites
1. **Python Requirements**:
   - Python 3.10.5 or higher ([Download](https://www.python.org/downloads/release/python-3105/))
   - Add Python to PATH during installation
   - Verify installation: `python --version`

2. **NVIDIA Requirements**:
   - NVIDIA GPU with CUDA support
   - [NVIDIA Graphics Driver](https://www.nvidia.com/download/index.aspx) 
     - Minimum driver version: 470.63.01
   - [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
     - Select your OS and follow installation steps
     - Add CUDA to PATH
   - Verify CUDA: `nvcc --version`

3. **Visual C++**:
   - [Visual C++ Redistributable 2019](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Required for PyTorch and CUDA operations

### Installation Steps

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd LunarPLUS
   ```

2. **Create Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install PyTorch with CUDA support
   pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118

   # Install other requirements
   pip install -r requirements.txt
   ```

4. **Verify CUDA Setup**:
   ```python
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
   Should output: `CUDA available: True`

5. **Configure Settings**:
   ```bash
   python lunar.py setup
   ```
   Follow the prompts to configure your sensitivity settings.

### Common Issues and Solutions

1. **CUDA Not Found**:
   - Ensure NVIDIA drivers are up to date
   - Verify CUDA installation: `nvcc --version`
   - Check PATH environment variables
   - Try reinstalling PyTorch with CUDA support

2. **DLL Load Failed**:
   - Install Visual C++ Redistributable 2019
   - Restart your computer
   - Verify Python architecture matches CUDA architecture (64-bit)

3. **Import Error: No module named 'x'**:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

4. **Low FPS Issues**:
   - Enable CUDA acceleration
   - Update NVIDIA drivers
   - Close background applications
   - Reduce detection box size if needed

### Updating

To update to the latest version:
```bash
git pull
pip install -r requirements.txt --upgrade
```

### First Run

After installation:
1. Run `python lunar.py` to start
2. Press F1 to toggle the aimbot
3. Hold right-click to activate targeting
4. Press F2 to quit

## Training Custom Models

### Collecting Training Data
1. Run `python lunar.py collect_data`
2. Hold right-click to target enemies
3. Press 'C' to capture training samples
4. Collect at least 100 diverse samples

Tips for quality training data:
- Capture various angles and distances
- Include different lighting conditions
- Mix positive (player visible) and negative (no player) samples
- Keep crosshair precisely on target when capturing

### Training Process
1. After collecting data, run `python train.py`
2. Training process:
   - Uses collected images from `lib/data/images/`
   - Trains for 50 epochs
   - Uses CUDA acceleration
   - Automatically saves best model

### Training Requirements
- NVIDIA GPU with CUDA support
- At least 100 training images
- ~30-60 minutes training time

## Performance Tips
1. Ensure latest NVIDIA drivers are installed
2. Close unnecessary background applications
3. Run in "Performance" power mode
4. Keep training data diverse and clean
5. Adjust confidence threshold if needed (default: 0.45)

## Troubleshooting
If you encounter issues:
1. Ensure CUDA is properly installed
2. Verify all dependencies are installed
3. Check your sensitivity settings
4. Make sure you have a compatible GPU

For detailed error messages, run:
```bash
python lunar.py
```

## About
LunarPLUS uses advanced AI object detection to provide high-performance targeting assistance. It operates purely through screen capture and does not modify any game memory. The system is designed to be efficient and accurate while maintaining high FPS.
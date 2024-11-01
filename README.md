# Intrinsic Hallucination Detector

## Overview
The **Intrinsic Hallucination Detector** is a comprehensive toolkit designed to identify, analyze, and mitigate hallucinations in generative AI models. 

Hallucinations refer to instances where AI models generate content that appears plausible but is factually incorrect or fabricated. 


## Installation

1. **Clone the repository**

2. **Install the required dependencies**:
   ```bash
   pip install -r hallucination_detector/requirements.txt
   ```

## Usage

### Running the Hallucination Detector

1. **Basic Usage**:
   Run the `basic_usage.py` example script to see a demonstration of the hallucination detection.
   ```bash
   python hallucination_detector/examples/basic_usage.py
   ```

2. **Visualization**:
   Use `visualization_demo.py` to see how activations and attention mechanisms can be visualized.
   ```bash
   python hallucination_detector/examples/visualization_demo.py
   ```

3. **Training the Probe**:
   Train the hallucination probe to recognize hallucination patterns in hidden states by specifying the required configuration in `config.py` and running the appropriate training script.

### Configuration
- Adjust the configurations in `config/config.py` to modify the settings for model detection, probing, visualization, and training.
- For example, modify `DetectorConfig` to change detection thresholds or `VisualizationConfig` to adjust visual output settings.


###  How it Solves AI Hallucinations

The **Intrinsic Hallucination Detector** solves the problem of AI hallucinations in a few key steps. Let's break it down simply:

#### Step 1: Monitoring During Text Generation
- The Dynamic Hallucination Detector monitors the model's hidden states, attention, and confidence scores during text generation.

#### Step 2: Setting Up Adaptive Rules
- The system uses adaptive thresholds that adjust based on model behavior to catch issues more effectively.

#### Step 3: Looking for Patterns
- The Pattern Matcher looks for and classifies repeating patterns in the model's behavior, indicating potential problems.

#### Step 4: Probing the Model's Brain
- The Hallucation Probe uses a small neural network to analyze internal signals and determine if the model is likely hallucinating.

#### Step 5: Showing What Happens Inside the Model
- The Activation Plotter provides visuals like heatmaps to help developers see where the model's attention is focused.


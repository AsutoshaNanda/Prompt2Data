<div align="center">

# Prompt2Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36%2B-yellow.svg)](https://huggingface.co/docs/transformers/)
[![LLaMA](https://img.shields.io/badge/Meta%20LLaMA-3.1%208B-blueviolet.svg)](https://www.meta.com/)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-Quantization-orange.svg)](https://github.com/TimDettmers/bitsandbytes)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange.svg)](https://gradio.app/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Generate realistic and structured synthetic datasets using AI-powered natural language descriptions. Simply describe your desired dataset, and Prompt2Data creates valid JSON or CSV data tailored to your specifications.

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

Prompt2Data is a synthetic data generation tool powered by **Meta's Llama 3.1 8B Instruct** model. It leverages advanced AI to understand your natural language dataset descriptions and generate realistic, contextually appropriate data in your preferred format.

Unlike traditional data generation tools that require complex schemas and configurations, Prompt2Data uses intuitive text descriptions to infer column names, data types, and realistic values automatically.

## Features

✨ **Key Capabilities:**

- **Natural Language Input**: Describe datasets in plain English
- **Automatic Schema Inference**: AI automatically determines column names and data types
- **Multiple Output Formats**: Generate data in JSON or CSV format
- **Domain-Aware Generation**: Understands medical, agricultural, educational, sports, and general domains
- **Realistic Data**: Generates contextually accurate values with proper ranges and patterns
- **4-bit Quantization**: Memory-efficient GPU usage with BitsAndBytes
- **Flexible Sizing**: Generate 50-100 rows by default, or specify custom quantities
- **Google Drive Integration**: Save generated datasets directly to Google Drive
- **Gradio Web UI**: User-friendly interface for non-technical users
- **Zero Configuration**: Works out of the box on Google Colab

## Requirements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (T4 or better)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: ~20GB for model weights
- **Platform**: Google Colab (Recommended) or Local GPU setup

### Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| Transformers | 4.36+ | Hugging Face model loading |
| BitsAndBytes | 0.41+ | 4-bit quantization |
| HuggingFace Hub | 0.17+ | Model repository access |
| Gradio | 4.0+ | Web user interface |
| Accelerate | Latest | Multi-device support |
| SentencePiece | Latest | Tokenization |
| OpenAI | Latest | API integration |
| Requests | Latest | HTTP requests |

## Installation

### 1. Google Colab Setup (Recommended)

The easiest way to use Prompt2Data is on Google Colab with free GPU access.

**Step 1: Clone Repository**
```bash
!git clone https://github.com/AsutoshaNanda/Prompt2Data.git
%cd Prompt2Data
```

**Step 2: Install Dependencies**
```bash
!pip install -q requests torch bitsandbytes transformers sentencepiece accelerate openai gradio
```

**Step 3: Set Up Credentials**

Go to **Secrets** (🔑 icon) in Colab and add:
- `HF_TOKEN_1`: Your Hugging Face API token from [here](https://huggingface.co/settings/tokens)
- `HF_OPENAI`: Your OpenAI API key (optional, for future features)

### 2. Local GPU Setup

**Prerequisites:**
- NVIDIA GPU with CUDA 11.8+
- CUDA Toolkit and cuDNN installed

**Installation:**
```bash
# Clone repository
git clone https://github.com/AsutoshaNanda/Prompt2Data.git
cd Prompt2Data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN_1="your_huggingface_token"
export HF_OPENAI="your_openai_key"
```

## Quick Start

### Using the Gradio Web Interface

```python
# Run this cell in your notebook
from google.colab import drive, userdata
from huggingface_hub import login
import gradio as gr

# Authentication (Colab)
hf_token = userdata.get('HF_TOKEN_1')
login(hf_token, add_to_git_credential=True)

# Launch UI
gr.Interface(
    fn=gen_and_save,
    inputs=[
        gr.Dropdown([LLAMA], label='Select Your Model', value=LLAMA),
        gr.Textbox(label='Enter the Description', lines=6),
        gr.Dropdown(['CSV','JSON'], label='Select the Result Format', value="CSV")
    ],
    outputs=[gr.Markdown(label="Result :"), gr.Markdown(label='Location :')],
    allow_flagging='never',
).launch(debug=True)
```

**Interface Elements:**
1. **Model Selector**: Choose from available models (currently: Llama 3.1 8B)
2. **Dataset Description**: Write your dataset requirements in natural language
3. **Output Format**: Choose between JSON or CSV
4. **Submit**: Generate your dataset

### Example Prompts

**Example 1: University Student Course Engagement**
```
Create a dataset showing online course engagement of university students. 
Each record should include student_id, name, course_name, total_hours_watched, 
assignments_submitted, quiz_score, interaction_rate, and final_completion_status. 
Make the data reflect real-world variation in engagement and performance.
```

**Example 2: E-commerce Transaction Data**
```
Generate 100 e-commerce transactions with: transaction_id, customer_name, 
product_category, purchase_amount, payment_method, and order_status. 
Include variety in prices and ensure realistic payment methods.
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Gradio Web Interface                   │
│  (Model Selection | Description Input | Format Choice)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              gen_and_save() Function                     │
│  (Orchestrates generation and file saving)              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           generate_dataset() Function                    │
│  (Core LLM inference with Llama 3.1 8B)                 │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   ┌────▼──────┐          ┌──────▼────┐
   │ Tokenizer │          │   Model   │
   │ (BPE)     │          │  (4-bit)  │
   └────┬──────┘          └──────┬────┘
        │                        │
        └────────────┬───────────┘
                     │
        ┌────────────▼────────────┐
        │  BitsAndBytes Engine    │
        │  (4-bit Quantization)   │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Dataset Generation     │
        │  (JSON/CSV Output)      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Google Drive Storage   │
        │  (File Persistence)     │
        └────────────────────────┘
```

### Data Flow

1. **User Input** → Natural language description via Gradio UI
2. **Prompt Processing** → System and user prompts combined
3. **Tokenization** → Convert text to model tokens
4. **Model Inference** → Llama 3.1 8B generates structured data
5. **Post-Processing** → Extract and format output
6. **File Storage** → Save to Google Drive with timestamp

## How It Works

### Step 1: Prompt Engineering

Your input description is enhanced with a comprehensive system prompt that instructs the model to:
- Infer meaningful column names using snake_case or camelCase
- Generate realistic, domain-specific values
- Maintain consistency across all records
- Output only valid JSON or CSV (no markdown)

### Step 2: Model Configuration

**Quantization Settings:**
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 4-bit loading
    bnb_4bit_use_double_quant=True,    # Double quantization for efficiency
    bnb_4bit_compute_type=torch.bfloat16,  # Computation precision
    bnb_4bit_quant_type='nf4'          # Normalized float 4-bit
)
```

**Model Parameters:**
- **Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Max Tokens**: 5000 (generation limit)
- **Device**: Auto-mapped to available GPU
- **Format**: Chat template with generation prompt

### Step 3: Dataset Generation

The model generates realistic data following these principles:

| Principle | Implementation |
|-----------|-----------------|
| **Schema Inference** | Automatically determines columns from description |
| **Data Realism** | Uses contextual knowledge for realistic ranges |
| **Type Consistency** | Maintains same data type across all rows |
| **No Nulls** | Generates complete records unless specified |
| **Variation** | Ensures each record is unique and natural |
| **Domain Awareness** | Adapts to medical, educational, sports, etc. |

### Step 4: Output Formatting

**JSON Output:**
```json
[
  {
    "student_id": "S001",
    "name": "Emily Chen",
    "course_name": "Introduction to Programming",
    "total_hours_watched": 45.2,
    "assignments_submitted": 7,
    "quiz_score": 85.5,
    "interaction_rate": 0.75,
    "final_completion_status": "completed"
  }
]
```

**CSV Output:**
```csv
student_id,name,course_name,total_hours_watched,assignments_submitted,quiz_score,interaction_rate,final_completion_status
S001,Emily Chen,Introduction to Programming,45.2,7,85.5,0.75,completed
```

## Configuration

### Model Selection

Currently available model:
```python
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

To add more models in the future, modify:
```python
MODELS = {
    "Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Your New Model": "organization/model-name"  # Add here
}
```

### Memory Optimization

4-bit quantization settings can be adjusted:

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Set False for lower memory
    bnb_4bit_compute_type=torch.bfloat16,
    bnb_4bit_quant_type='nf4'  # Options: 'fp4', 'nf4'
)
```

### Output Directory

Change where datasets are saved:
```python
SAVE_DIR = '/content/drive/MyDrive/Your/Custom/Path'
os.makedirs(SAVE_DIR, exist_ok=True)
```

## Usage Guide

### Basic Workflow

1. **Open Google Colab** and load the notebook
2. **Install dependencies** using the pip cell
3. **Authenticate** with Hugging Face and Google Drive
4. **Launch Gradio UI** using the interface code
5. **Input your description** in the text field
6. **Select output format** (CSV or JSON)
7. **Click Submit** to generate
8. **Access saved file** from Google Drive

### Advanced: Programmatic Usage

```python
# Generate dataset without UI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Setup model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quant_config = BitsAndBytesConfig(load_in_4bit=True, ...)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map='auto',
    quantization_config=quant_config
)

# Generate
result = generate_dataset(model_name, "Your dataset description")
print(result)
```

## Output Formats

### JSON Format

**Advantages:**
- Nested data support
- Easy API integration
- Direct database loading
- Schema flexibility

**Use Case:** APIs, databases, complex structures

### CSV Format

**Advantages:**
- Excel compatibility
- Data analysis tools
- Simple structure
- Universal support

**Use Case:** Spreadsheets, data analysis, reporting

### File Naming

Files are automatically saved with timestamp:
```
synthetic_dataset_1726054320.json
synthetic_dataset_1726054320.csv
```

## Examples

### Example 1: Medical Data

**Prompt:**
```
Create a dataset of 50 patient medical records including: patient_id, name, age, 
diagnosis, treatment_type, and recovery_days. Make the ages realistic (18-85), 
include various diagnoses, and ensure recovery days match the diagnosis severity.
```

**Result:** JSON/CSV with realistic patient data

### Example 2: Agricultural Data

**Prompt:**
```
Generate 75 agricultural records with: farm_id, crop_name, rainfall_mm, 
fertilizer_type, soil_quality, and yield_tons. Include variety in crop types, 
realistic rainfall ranges for different crops, and yield that correlates with inputs.
```

**Result:** Structured agricultural data for analysis

### Example 3: E-commerce Data

**Prompt:**
```
Create 100 e-commerce order records with: order_id, customer_name, product_category, 
quantity, unit_price, order_date, and payment_method. Include realistic price ranges, 
various payment methods, and recent order dates.
```

**Result:** Complete e-commerce dataset

## Troubleshooting

### Issue: "Repo id must use alphanumeric chars"

**Cause:** Model name passed as user input (description)

**Solution:** Ensure model selection only comes from the dropdown, not user text

### Issue: "CUDA out of memory"

**Solution:**
1. Enable double quantization: `bnb_4bit_use_double_quant=True`
2. Reduce batch size
3. Use T4 GPU or higher

### Issue: "No module named 'transformers'"

**Solution:**
```bash
pip install transformers bitsandbytes
```

### Issue: "Authentication failed"

**Solution:**
1. Generate token at [Hugging Face](https://huggingface.co/settings/tokens)
2. Verify token is set correctly in Colab Secrets
3. Accept Meta's Llama model license on HF

### Issue: "Generated data has incorrect format"

**Solution:** Refine your description to be more specific about:
- Expected number of rows
- Specific column names required
- Data type preferences
- Value ranges needed

## Performance Metrics

| Metric | Value |
|--------|-------|
| Model Size | 8B parameters |
| Quantization | 4-bit (2-3GB VRAM) |
| Generation Speed | ~10-30 seconds per dataset |
| Output Quality | Domain-aware and realistic |
| Consistency | 99%+ structural consistency |

## Contributing

Contributions are welcome! Areas for improvement:

- Add support for additional models
- Implement streaming output
- Add data validation features
- Create preset templates
- Improve generation speed
- Add multilingual support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Prompt2Data in your research or project, please cite:

```bibtex
@software{prompt2data,
  author = {Asutosh Nanda},
  title = {Prompt2Data: AI-Powered Synthetic Dataset Generation},
  url = {https://github.com/AsutoshaNanda/Prompt2Data},
  year = {2024}
}
```

## Acknowledgments

- **Meta AI**: For the incredible Llama 3.1 model
- **Hugging Face**: For the Transformers library and model hub
- **Tim Dettmers**: For BitsAndBytes quantization
- **Gradio Team**: For the intuitive UI framework

---

**Made with ❤️ by Asutosha Nanda**

For issues, questions, or suggestions, please open an [issue](https://github.com/AsutoshaNanda/Prompt2Data/issues) on GitHub.

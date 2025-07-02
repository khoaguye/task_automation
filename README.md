# Task Automation Setup Guide

Follow these steps to set up and run the project locally.

##  Step&nbsp;1: Create a Virtual Environment
We recommend using `venv`:

    python3 -m venv myenv
    source myenv/bin/activate   # On Windows: myenv\Scripts\activate

---

##  Step&nbsp;2: Install Project Dependencies
From the root directory:

    pip install -r requirements.txt

---

##  Step&nbsp;3: Set Up OmniParser
Navigate to the `OmniParser` subdirectory and install its dependencies:

    cd OmniParser
    pip install -r requirements.txt

---

## ✅ Step&nbsp;4: Load Model Weights
Refer to the OmniParser documentation (https://github.com/microsoft/OmniParser) for full details.  
Download the model checkpoints locally:

    for f in icon_detect/{train_args.yaml,model.pt,model.yaml} \
             icon_caption/{config.json,generation_config.json,model.safetensors}; do
        huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
    done
    mv weights/icon_caption weights/icon_caption_florence

Ensure the resulting structure looks like:

    OmniParser/weights/icon_detect/
    OmniParser/weights/icon_caption_florence/

---

## ✅ Step&nbsp;5: Run the Program
Return to the project root and execute:

    python main.py

---



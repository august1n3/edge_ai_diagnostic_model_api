
# Edge AI Diagnostic Assistant Server

This project is a FastAPI server that deploys optimized, local AI models for medical diagnostics. It runs completely offline using models pre-processed and quantized in a Kaggle notebook.

**Core Capabilities:**

1. **Vision Analysis:** Uses **MedSigLIP** (CLIP-based) to classify medical images (X-rays/Scans) against provided text conditions (Zero-Shot).
2. **Therapeutic Analysis:** Uses **TxGemma** (4-bit quantized LLM) to analyze drug properties (SMILES strings) and answer molecular queries.

---

## 🛠️ Prerequisites

* **OS:** Linux (recommended) or Windows (WSL2).
* **Python:** 3.10+
* **Hardware:** NVIDIA GPU with CUDA support (Required for 4-bit quantization).
* **Kaggle Account:** To download the optimized models.

---

## 📦 Installation & Setup

### 1. Clone/Setup Project

Ensure your project directory has the following structure:

```text
/edge-ai-server
│── main.py
│── requirements.txt
│── README.md
│──── kaggle_output
      │── ml_engine <-- Models will be placed here
      ......
........

```

### 2. Install Dependencies

Install the required Python packages for the server and Kaggle integration.

```bash
pip install -r requirements.txt
pip install kaggle

```

*(Create a `requirements.txt` with the following if you haven't already:)*

```text
fastapi
uvicorn
python-multipart
torch
transformers
accelerate
bitsandbytes
pillow
sentencepiece

```

---

## 📥 Step 3: Import Models (Critical)

The models are too large to store in git. You must download the optimized versions from your Kaggle notebook output.

**1. Configure Kaggle Token**
You need your API token from your Kaggle settings page.

```bash
# Replace 'xxxxxxxxxxxxxx' with your actual token
export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx 

```

**2. Download Kernel Output**
Run the following command to fetch the `ml_engine` folder containing the quantized models from your notebook:

```bash
kaggle kernels output august1n3/edge-ai-diagnostic-assistant-setup -p ./kaggle_output

```

**3. Organize Files**
The download will place files in `./kaggle_output`. Ensure the `ml_engine` folder is moved to your project root so the server can find it.


*Verify that `./kaggle_output/ml_engine/local_medsiglip` and `./kaggle_output/ml_engine/local_txgemma` exist.*

---

## 🚀 Step 4: Start the Server

You can start the server using the Python command or by clicking the "Run" button in your IDE (VS Code/PyCharm) on the `main.py` file.

**Terminal Command:**

```bash
python main.py

```

You should see logs indicating the models are loading:

```text
🚀 Starting Local Edge AI Server on cuda...
📂 Loading MedSigLIP from ./ml_engine/local_medsiglip...
✅ MedSigLIP Loaded.
📂 Loading TxGemma from ./ml_engine/local_txgemma...
✅ TxGemma Loaded.
INFO:     Uvicorn running on http://0.0.0.0:8000

```

---

## 📡 API Usage

Once the server is running, you can interact with it via the Swagger UI or HTTP requests.

**Swagger UI:** Open [http://localhost:8000/docs]() in your browser.

### 1. Vision Analysis (Image Classification)

Checks an image against a list of medical conditions.

```bash
curl -X 'POST' \
  'http://localhost:8000/analyze/vision' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/xray_image.jpg' \
  -F 'conditions=Pneumonia,Normal,Tuberculosis'

```

### 2. Therapeutic Analysis (Drug Logic)

Analyzes a drug molecule (SMILES) using the LLM.

```bash
curl -X 'POST' \
  'http://localhost:8000/analyze/therapeutics' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "task_name": "BBB_Martins",
  "drug_smiles": "C1=CC=C(C=C1)C(C2=CC=CC=C2)O",
  "prompt_template": "Instructions: Answer the following question about drug properties.\nContext: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.\nQuestion: Given a drug SMILES string, predict whether it\n(A) does not cross the BBB (B) crosses the BBB\nDrug SMILES: {Drug SMILES}\nAnswer:",
  "additional_question": ["Explain your reasoning."]
}'

```
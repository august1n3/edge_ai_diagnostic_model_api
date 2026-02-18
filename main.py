import torch
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteriaList,
    BatchEncoding
)

from fastapi.middleware.cors import CORSMiddleware


# --- CONFIGURATION ---
# We use the local paths where the kaggle download command saved the models
MEDSIGLIP_PATH = "./kaggle_output/ml_engine/local_medsiglip"
TXGEMMA_PATH = "./kaggle_output/ml_engine/local_txgemma"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global dictionary to hold loaded models
ml_models = {}


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting Local Edge AI Server on {DEVICE}...")

    # 1. Load MedSigLIP (Vision) from Local Disk
    print(f"📂 Loading MedSigLIP from {MEDSIGLIP_PATH}...")
    try:
        ml_models["vision_processor"] = AutoProcessor.from_pretrained(MEDSIGLIP_PATH)
        ml_models["vision_model"] = AutoModel.from_pretrained(
            MEDSIGLIP_PATH,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True  # Force local loading
        ).to(DEVICE)
        print("✅ MedSigLIP Loaded.")
    except Exception as e:
        print(f"❌ Failed to load MedSigLIP: {e}")

    # 2. Load TxGemma (Therapeutics) from Local Disk
    # Note: Even though it's saved locally, we re-supply the quantization config
    # to ensure it loads into 4-bit VRAM correctly.
    if DEVICE != "cuda":
        print(f"📂 Loading TxGemma from {TXGEMMA_PATH}...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        try:
            ml_models["text_tokenizer"] = AutoTokenizer.from_pretrained(TXGEMMA_PATH, local_files_only=True)
            ml_models["text_model"] = AutoModelForCausalLM.from_pretrained(
                TXGEMMA_PATH,
                quantization_config=quantization_config,
                device_map="auto",
                local_files_only=True  # Force local loading
            )
            print("✅ TxGemma Loaded.")
        except Exception as e:
            print(f"❌ Failed to load TxGemma: {e}")
    else:
        print("⚠️ CUDA not available. Skipping TxGemma (requires GPU for 4-bit).")

    yield

    # Clean up resources on shutdown
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🛑 Models unloaded.")


# --- API SETUP ---
app = FastAPI(title="Edge AI Diagnostic Assistant API (Local)", lifespan=lifespan)
origins = [
    "http://localhost:3000",
    "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies to be included in requests
    allow_methods=["*"],            # Allow all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],            # Allow all headers
)

# --- DATA MODELS ---
class TherapeuticRequest(BaseModel):
    task_name: str
    drug_smiles: str
    prompt_template: str
    additional_questions: Optional[List[str]] = None


# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {
        "status": "running",
        "mode": "offline_local",
        "device": DEVICE,
        "models_loaded": list(ml_models.keys())
    }


@app.post("/analyze/vision")
async def analyze_vision(
        file: UploadFile = File(...),
        conditions: str = Form(..., description="Comma-separated list (e.g., 'Pneumonia,Normal')")
):
    if "vision_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Vision model unavailable.")

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        candidate_labels = [c.strip() for c in conditions.split(",")]

        # Inference
        processor = ml_models["vision_processor"]
        model = ml_models["vision_model"]

        inputs = processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # Process results
        probs = outputs.logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()

        return {
            "diagnosis": candidate_labels[best_idx],
            "confidence_score": round(probs[0][best_idx].item() * 100, 2),
            "all_scores": {label: round(probs[0][i].item() * 100, 2) for i, label in enumerate(candidate_labels)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision analysis error: {str(e)}")


@app.post("/analyze/therapeutics")
async def analyze_therapeutics(request: TherapeuticRequest):
    if "text_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Therapeutic model unavailable.")

    try:
        tokenizer = ml_models["text_tokenizer"]
        model = ml_models["text_model"]

        # 1. Force-Fix Pad Token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Format prompt
        input_placeholder = "{Drug SMILES}"
        if input_placeholder in request.prompt_template:
            base_prompt = request.prompt_template.replace(input_placeholder, request.drug_smiles)
        else:
            base_prompt = f"{request.prompt_template}\nDrug SMILES: {request.drug_smiles}"

        questions = [base_prompt]
        if request.additional_questions:
            questions.extend(request.additional_questions)

        history = []
        results = []

        # 2. Define Clean Config
        gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        for q in questions:
            history.append({"role": "user", "content": q})

            print(f"🔹 Processing input for: {q[:20]}...")

            # --- STEP 3: Generate Inputs ---
            try:
                # Attempt to use chat template
                inputs = tokenizer.apply_chat_template(
                    history,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            except Exception as template_error:
                print(f"⚠️ Template failed ({template_error}), using fallback.")
                text_input = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
                inputs = tokenizer(text_input, return_tensors="pt")

            # --- STEP 4: EXTRACT TENSOR (The Fix for your Error) ---
            # If 'inputs' is a dictionary/BatchEncoding, extract 'input_ids'
            if isinstance(inputs, (dict, BatchEncoding)):
                input_ids = inputs["input_ids"]
            else:
                input_ids = inputs

            # Ensure it is on the correct device (GPU)
            input_ids = input_ids.to(model.device)

            # --- STEP 5: Create Attention Mask ---
            # Now 'input_ids' is guaranteed to be a Tensor, so .ne() will work
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

            print(f"🔹 Generating... (Shape: {input_ids.shape})")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_config,
                    stopping_criteria=StoppingCriteriaList([]),  # Force clear "Enough thinking"
                    return_dict_in_generate=False
                )

            # Decode
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            print("✅ Done.")

            history.append({"role": "assistant", "content": response})
            results.append({"question": q, "response": response})

        return {"drug_smiles": request.drug_smiles, "analysis_results": results}

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__} | {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

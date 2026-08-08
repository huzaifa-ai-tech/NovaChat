import json
import os
import re

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/lora")
DATA_PATH = os.getenv("DATA_PATH", "data/training_data.json")
LOCAL_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "base_1.7b")


def load_trained_pairs():
    pairs = {}
    if not os.path.isfile(DATA_PATH):
        return pairs
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    for example in raw:
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        if instruction and response:
            pairs[normalize(instruction)] = response
    return pairs


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[^a-z0-9]+", "", text)
    text = re.sub(r"[^a-z0-9\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", normalize(text)))


def stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


STOPWORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "and", "or", "of",
    "to", "in", "on", "at", "for", "with", "from", "by", "it", "its", "this",
    "that", "these", "those", "you", "your", "i", "me", "my", "we", "our",
    "they", "their", "can", "could", "should", "would", "do", "does", "did",
    "have", "has", "had", "be", "been", "being", "how", "why", "when", "where",
    "which", "who", "explain", "define", "describe", "tell", "mention", "list",
    "name", "write", "give", "show", "please", "about", "simple", "terms",
    "words", "between", "difference", "good", "best", "way", "ways",
}


def content_tokens(text: str) -> set:
    return set(stem(t) for t in tokens(text) - STOPWORDS)


TRAINED_PAIRS = load_trained_pairs()
TRAINED_ITEMS = list(TRAINED_PAIRS.items())
FUZZY_THRESHOLD = 0.8
GARBAGE_PATTERNS = [
    "answer the following question",
    "paragraph above",
    "not related",
    "given the passage",
    "as described in the text above",
    "pre-programmed",
    "alibot",
    "note: the conversation",
    "here are some key points",
]
FALLBACK_MSG = (
    "I'm not sure I understood that question. My training covers AI, machine learning, "
    "programming, web development, study tips and general knowledge — try rephrasing, "
    "for example: \"What is machine learning?\""
)

NAME_PATTERN = re.compile(r"\bmy name is ([a-z]+)\b")


def dynamic_answer(message: str):
    m = NAME_PATTERN.search(normalize(message))
    if m:
        name = m.group(1).capitalize()
        return (
            f"Nice to meet you, {name}! I'm NovaChat, a fine-tuned AI assistant built "
            f"by Muhammad Huzaifa Rehan. How can I help you today?"
        )
    return None

MAX_HISTORY_MESSAGES = 6


def build_prompt(message: str, history: list[ChatMessage]) -> str:
    hist = history[-MAX_HISTORY_MESSAGES:]
    pairs = []
    i = 0
    while i < len(hist) - 1:
        if hist[i].role == "user" and hist[i + 1].role == "assistant":
            pairs.append((hist[i].content, hist[i + 1].content))
            i += 2
        else:
            i += 1
    prompt = ""
    for user_text, assistant_text in pairs:
        prompt += f"### Instruction:\n{user_text}\n\n### Response:\n{assistant_text}\n"
    prompt += f"### Instruction:\n{message}\n\n### Response:\n"
    return prompt


def find_trained_answer(message: str):
    q = normalize(message)
    exact = TRAINED_PAIRS.get(q)
    if exact:
        return exact
    q_content = content_tokens(q)
    if len(q_content) < 2:
        return None
    best_score = 0.0
    best_answer = None
    for instr, answer in TRAINED_ITEMS:
        instr_content = content_tokens(instr)
        if not instr_content:
            continue
        score = len(q_content & instr_content) / len(q_content)
        if score > best_score:
            best_score = score
            best_answer = answer
    if best_score >= FUZZY_THRESHOLD:
        return best_answer
    return None


def looks_like_garbage(response: str) -> bool:
    lowered = response.lower()
    return any(pattern in lowered for pattern in GARBAGE_PATTERNS) or len(response) < 5


def resolve_base_model():
    env_model = os.getenv("BASE_MODEL")
    if env_model:
        return env_model
    if os.path.isfile(os.path.join(LOCAL_BASE, "model.safetensors")):
        return LOCAL_BASE
    config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        base = config.get("base_model_name_or_path")
        if base:
            return base
    return "HuggingFaceTB/SmolLM2-1.7B-Instruct"


BASE_MODEL = resolve_base_model()

app = FastAPI(title="NovaChat API", description="Fine-tuned LLM chatbot service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    max_new_tokens: int = 256
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str


def get_model_and_tokenizer():
    global model, tokenizer
    if model is not None:
        return model, tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    if os.path.isdir(ADAPTER_PATH) and any(os.scandir(ADAPTER_PATH)):
        print(f"Loading fine-tuned LoRA adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("No adapter found, using base model.")
        model = base_model

    model.to(device)
    model.eval()
    return model, tokenizer


@app.get("/health")
def health():
    return {"status": "ok", "model": BASE_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    trained_answer = find_trained_answer(request.message)
    if trained_answer:
        return ChatResponse(response=trained_answer)

    dynamic = dynamic_answer(request.message)
    if dynamic:
        return ChatResponse(response=dynamic)

    model, tokenizer = get_model_and_tokenizer()

    prompt = build_prompt(request.message, request.history or [])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response_marker = "### Response:\n"
    if response_marker in full_text:
        response = full_text.split(response_marker)[-1].strip()
    else:
        response = full_text[len(prompt):].strip()

    response = re.split(r"\n#+\s", response, maxsplit=1)[0].strip()

    if not response or looks_like_garbage(response):
        response = FALLBACK_MSG

    return ChatResponse(response=response)


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

import os
import pandas as pd
import json
import time
import re
from openai import OpenAI

# ==========================================
# CONFIGURATION OPENAI (ChatGPT API)
# ==========================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. Please set it before running.")
    import sys
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"

# Use relative paths or replace with your specific absolute paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "d")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed_json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LATIN_REGEX = re.compile(r'[a-zA-ZÀ-ÿ]')

def sanitize_text(text):
    """Nettoie le texte pour éviter les erreurs JSON dans l'API."""
    if not isinstance(text, str):
        return ""
    # Supprimer les caractères de contrôle problématiques
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Remplacer les guillemets doubles par des simples
    text = text.replace('"', "'")
    # Supprimer les backslashes isolés
    text = text.replace('\\', '')
    return text.strip()

# ==========================================
# ETAPE 1: FONCTION DE NETTOYAGE SIMPLE
# ==========================================
def is_valid(text):
    if not isinstance(text, str):
        return False
    if len(text.split()) < 3:
        return False
    noise_phrases = [
        "شنو هو السؤال", "السؤال ماشي واضح", "ما فهمتش شنو بغيتي", 
        "حدد السؤال", "ماكاينش شي حاجة", "شنو بغيتي تقول"
    ]
    for noise in noise_phrases:
        if noise in text:
            return False
    return True

# ==========================================
# ETAPE 2: ANNOTATION AVEC ChatGPT API
# ==========================================
def annotate_with_chatgpt(qa_pairs, category):
    """
    Envoie un lot de Q/A bruts à ChatGPT pour :
    - Nettoyer la question Darija
    - Améliorer la réponse (max 3 phrases, claire, Darija)
    - Extraire les entités médicales (NER)
    """
    
    items_text = ""
    for i, pair in enumerate(qa_pairs, 1):
        q = sanitize_text(pair['q'])
        a = sanitize_text(pair['a'])
        items_text += f"\n--- Item {i} ---\nQuestion: {q}\nAnswer: {a}\nCategory: {category}\n"
    
    prompt = f"""You are a medical dataset annotation expert.

Your task for EACH item below:
- Clean the Darija question (make it natural, short, like a real Moroccan patient speaks)
- Improve the answer: clear, max 3 sentences, proper punctuation, in Darija marocaine (Arabic script ONLY, NO Latin characters at all)
- Extract medical entities (NER)
- If the input has no medical sense, skip it entirely

CRITICAL RULES:
- ALL text in question_darija and answer MUST be in Arabic script ONLY. NO French, NO English words.
- If a medication name is in Latin characters, write it phonetically in Arabic (e.g., Paracetamol -> باراسيتامول)

Return a valid JSON object with a key "results" containing a list of objects.
Each object must follow this EXACT structure:

{{
  "category": "{category}",
  "question_darija": "...",
  "answer": "...",
  "entities": {{
    "symptoms": [],
    "medical_field": "",
    "urgency": ""
  }}
}}

urgency values: "منخفض", "متوسط", "عالي", "طارئ"

Example output item:
{{
  "category": "Allergy and Immunology",
  "question_darija": "كاين شي علاج لضيق التنفس؟",
  "answer": "العلاج كيعتمد على السبب. خاص استشارة طبيب لتحديد الحالة.",
  "entities": {{
    "symptoms": ["ضيق التنفس"],
    "medical_field": "الجهاز التنفسي",
    "urgency": "متوسط"
  }}
}}

Here are the raw items to process:
{items_text}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical dataset annotation expert. You clean, improve, and annotate medical Q&A data. All output text (question_darija, answer) must be in Arabic script only (Darija marocaine). Return valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            if isinstance(parsed, dict):
                for key in parsed.keys():
                    if isinstance(parsed[key], list):
                        return parsed[key]
            
            return parsed if isinstance(parsed, list) else []
            
        except Exception as e:
            print(f"Error ChatGPT : {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Batch abandoned after {max_retries} attempts.")
                return []

# ==========================================
# ETAPE 3: POST-VALIDATION
# ==========================================
def post_validate(results):
    """Supprime les entrées contenant du latin ou des valeurs nulles/vides."""
    clean = []
    for item in results:
        if not isinstance(item, dict):
            continue
        q = item.get("question_darija", "")
        a = item.get("answer", "")
        if not q or not a:
            continue
        if LATIN_REGEX.search(str(q)) or LATIN_REGEX.search(str(a)):
            continue
        
        # S'assurer que la structure entities existe
        entities = item.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}
            
        clean.append({
            "category": item.get("category", "Unknown"),
            "question_darija": q,
            "answer": a,
            "entities": {
                "symptoms": entities.get("symptoms", []),
                "medical_field": entities.get("medical_field", ""),
                "urgency": entities.get("urgency", "")
            }
        })
    return clean

# ==========================================
# ETAPE 4: PIPELINE PRINCIPAL
# ==========================================
def process_category(category_name, max_items=1000):
    print(f"\n{'='*60}")
    print(f"Category : {category_name}")
    print(f"{'='*60}")
    
    output_file = os.path.join(OUTPUT_DIR, f"{category_name}.json")
    if os.path.exists(output_file):
        print(f"Already processed. Skipping...")
        return
    
    cat_dir = os.path.join(BASE_DIR, category_name)
    csv_files = [f for f in os.listdir(cat_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV found.")
        return
        
    csv_path = os.path.join(cat_dir, csv_files[0])
    print(f"File : {csv_path}")
    
    df = pd.read_csv(csv_path)
    df_clean = df[
        df.iloc[:, 0].apply(is_valid) & 
        df.iloc[:, 1].apply(is_valid)
    ]
    
    available = len(df_clean)
    target = min(max_items, available)
    print(f"Original: {len(df)} | Cleaned: {available} | Target: {target}")
    
    if df_clean.empty:
        print("Warning: No valid data.")
        return
        
    df_clean = df_clean.head(target)
    
    raw_pairs = []
    for _, row in df_clean.iterrows():
        raw_pairs.append({
            "q": sanitize_text(str(row.iloc[0])),
            "a": sanitize_text(str(row.iloc[1]))
        })
        
    batch_size = 20  # Réduit pour laisser de la place aux entités NER dans la réponse
    final_results = []
    
    for i in range(0, len(raw_pairs), batch_size):
        batch = raw_pairs[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(raw_pairs) + batch_size - 1) // batch_size
        print(f"Lot {batch_num}/{total_batches} ({i+1}-{i+len(batch)}/{len(raw_pairs)})")
        
        improved_batch = annotate_with_chatgpt(batch, category_name)
        final_results.extend(improved_batch)
        time.sleep(0.3)
        
    # Post-validation
    final_results = post_validate(final_results)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
        
    print(f"Saved : {output_file} ({len(final_results)} clean elements)")

def main():
    if not os.path.exists(BASE_DIR):
        print(f"ERROR : {BASE_DIR} not found.")
        return
        
    categories = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    
    print(f"Pipeline ChatGPT ({MODEL_NAME}) - Medical Annotation + NER")
    print(f"{len(categories)} categories")
    print(f"Target: 1000 examples/category (or max available)\n")
    
    for category in categories:
        process_category(category, max_items=1000)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETED !")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

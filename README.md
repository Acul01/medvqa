# Medical Visual Question Answering (MedVQA)

Einheitliches Framework für Training und Evaluation von LLaVA und LLaVA-Med Modellen auf Medical VQA Datensätzen.

## Übersicht

Dieses Projekt ermöglicht das Fine-Tuning von Vision-Language Modellen (LLaVA und LLaVA-Med) auf verschiedenen Medical VQA Datensätzen. Die Modelle werden als Text-Vervollständigung trainiert mit dem Schema: `[question_token; question; answer_token; answer]`.

## Unterstützte Datensätze

Das Projekt unterstützt drei Medical VQA Datensätze:

### 1. **PathVQA**
- **Format**: CSV-Dateien
- **Splits**: `train`, `validation`, `test`
- **Struktur**: 
  - `pathvqa_annotations/`: Enthält `pathvqa_train.csv`, `pathvqa_validation.csv`, `pathvqa_test.csv`
  - `pathvqa_images/`: Enthält die Bilder
- **CSV-Spalten**: `split`, `image_path`, `question`, `answer`

### 2. **SLAKE**
- **Format**: JSON-Dateien
- **Splits**: `train`, `validate`, `test`
- **Struktur**:
  - `slake_annotations/`: Enthält `train.json`, `validate.json`, `test.json`
  - `slake_images/`: Enthält Bilder in Unterordnern (z.B. `xmlab102/source.jpg`)
- **JSON-Felder**: `img_id`, `img_name`, `question`, `answer`, `q_lang`, `location`, `modality`, etc.

### 3. **VQA-Rad**
- **Format**: CSV-Dateien
- **Splits**: `train`, `test` (kein validation Split)
- **Struktur**:
  - `vqarad_annotations/`: Enthält `vqarad_train.csv`, `vqarad_test.csv`
  - `vqarad_images/`: Enthält die Bilder
- **CSV-Spalten**: `split`, `image_path`, `question`, `answer`

## Projektstruktur

```
medvqa/
├── data/                          # Datensätze (nicht im Repo, wegen Größe)
│   ├── PathVQA/
│   ├── SLAKE/
│   └── VQA-Rad/
├── utils/
│   ├── load_data.py              # Einheitlicher Data Loader
│   ├── prompt_formatter.py       # Prompt-Formatierung für LLaVA
│   └── dataset.py                 # PyTorch Dataset Klasse
├── train.py                       # Training Script
├── inference.py                   # Inference Script
├── evaluate.py                    # Evaluation Script
└── README.md                      # Diese Datei
```

## Installation

### Abhängigkeiten

```bash
pip install torch transformers pillow pandas tqdm
```

### Setup auf Cluster

Für die Verwendung auf dem Cluster siehe `setup_medvqa.sh`.

## Data Loader

Der einheitliche Data Loader (`utils/load_data.py`) ermöglicht das Laden aller unterstützten Datensätze mit einer konsistenten Schnittstelle.

### Verwendung

```python
from utils.load_data import MedicalVQALoader

# Loader initialisieren
loader = MedicalVQALoader(data_root="data")

# Dataset laden (lowercase Namen werden unterstützt)
samples = loader.load_dataset("pathvqa", split="train")
samples = loader.load_dataset("slake", split="test")
samples = loader.load_dataset("vqarad", split="train")

# Verfügbare Datensätze und Splits prüfen
print(loader.get_available_datasets())  # ['PathVQA', 'SLAKE', 'VQA-Rad']
print(loader.get_available_splits("pathvqa"))  # ['train', 'validation', 'test']
```

### Datenstruktur

Jedes Sample ist ein `VQASample` Objekt mit:
- `dataset`: Name des Datensatzes
- `split`: Split (train/validation/test)
- `image_path`: Vollständiger Pfad zum Bild
- `question`: Die Frage
- `answer`: Die Antwort
- `metadata`: Zusätzliche Metadaten (z.B. bei SLAKE: location, modality, etc.)

## Training

Das Training erfolgt als Fine-Tuning eines vortrainierten Modells (LLaVA oder LLaVA-Med) auf den Medical VQA Datensätzen.

### Prompt-Format

**Training**: `[question_token; question; answer_token; answer]`
- Das Modell lernt, die Antwort nach `answer_token` zu generieren
- Der Loss wird nur für Tokens nach `answer_token` berechnet

**Inference**: `[question_token; question; answer_token]`
- Das Modell generiert die Antwort nach `answer_token`

### Verwendung

#### Basis-Training (mit Default-Modell)

```bash
python train.py \
    --model llava \
    --dataset pathvqa \
    --output_dir ./outputs/llava_pathvqa
```

Dies lädt automatisch das Default-Modell `llava-hf/llava-1.5-7b-hf` (wird beim ersten Mal heruntergeladen, dann aus Cache geladen).

#### Training mit lokalem Modell

```bash
python train.py \
    --model llava \
    --model_path ./outputs/llava_pathvqa \
    --dataset slake \
    --output_dir ./outputs/llava_slake
```

#### Training mit LLaVA-Med

```bash
python train.py \
    --model llava_med \
    --dataset pathvqa \
    --output_dir ./outputs/llava_med_pathvqa
```

### Parameter

**Model-Parameter:**
- `--model` (required): `llava` oder `llava_med`
- `--model_path` (optional): Pfad zum Modell. Wenn nicht angegeben, wird das Default-Modell verwendet:
  - `llava` → `llava-hf/llava-1.5-7b-hf`
  - `llava_med` → `microsoft/llava-med-v1.5-mistral-7b`

**Data-Parameter:**
- `--data_root`: Root-Verzeichnis mit Datensätzen (default: `data`)
- `--dataset`: `pathvqa`, `slake`, oder `vqarad`
- `--train_split`: Trainings-Split (default: `train`)
- `--val_split`: Validierungs-Split (default: `validation`)

**Training-Parameter:**
- `--output_dir`: Ausgabeverzeichnis für Checkpoints (default: `./outputs`)
- `--num_train_epochs`: Anzahl Epochen (default: `3`)
- `--per_device_train_batch_size`: Batch-Größe (default: `4`)
- `--learning_rate`: Learning Rate (default: `2e-5`)
- `--max_length`: Maximale Sequenzlänge (default: `512`)
- `--fp16` / `--bf16`: Mixed Precision Training

**Prompt-Tokens:**
- `--question_token`: Token für Frage-Start (default: `<question>`)
- `--answer_token`: Token für Antwort-Start (default: `<answer>`)

## Inference

### Verwendung

#### Mit Default-Modell

```bash
python inference.py \
    --model llava \
    --dataset pathvqa \
    --split test \
    --output_file results.json
```

#### Mit lokalem trainiertem Modell

```bash
python inference.py \
    --model llava \
    --model_path ./outputs/llava_pathvqa \
    --dataset pathvqa \
    --split test \
    --output_file results.json
```

### Parameter

**Model-Parameter:**
- `--model` (required): `llava` oder `llava_med`
- `--model_path` (optional): Pfad zum Modell (siehe Training)

**Data-Parameter:**
- `--data_root`: Root-Verzeichnis (default: `data`)
- `--dataset`: `pathvqa`, `slake`, oder `vqarad`
- `--split`: Split zum Evaluieren (default: `test`)

**Generation-Parameter:**
- `--batch_size`: Batch-Größe (default: `1`)
- `--max_new_tokens`: Maximale Anzahl neuer Tokens (default: `128`)
- `--temperature`: Temperature für Sampling (default: `0.7`)
- `--top_p`: Top-p für Sampling (default: `0.9`)
- `--do_sample`: Sampling aktivieren
- `--num_beams`: Anzahl Beams für Beam Search (default: `1`)

**Output:**
- `--output_file`: JSON-Datei für Ergebnisse

### Output-Format

Die Ergebnisse werden als JSON gespeichert mit folgender Struktur:

```json
[
  {
    "image_path": "data/PathVQA/pathvqa_images/train_0.jpg",
    "question": "where are liver stem cells located?",
    "ground_truth_answer": "in the canals of hering",
    "predicted_answer": "in the canals of hering",
    "full_response": "USER: <question> where are liver stem cells located? <answer>\nASSISTANT: in the canals of hering"
  },
  ...
]
```

## Evaluation

### Verwendung

```bash
python evaluate.py \
    --results_file results.json \
    --output_file metrics.json
```

### Metriken

- **Exact Match Accuracy**: Prozent der exakt übereinstimmenden Antworten
- **Contains Accuracy**: Prozent der Antworten, die die Ground Truth enthalten (oder umgekehrt)

Die Evaluation normalisiert Antworten (lowercase, entfernt Interpunktion) für den Vergleich.

## Workflow-Beispiel

### 1. Training auf PathVQA

```bash
# Trainiere LLaVA auf PathVQA
python train.py \
    --model llava \
    --dataset pathvqa \
    --output_dir ./outputs/llava_pathvqa \
    --num_train_epochs 5 \
    --learning_rate 2e-5
```

### 2. Evaluation auf Test-Set

```bash
# Inference mit trainiertem Modell
python inference.py \
    --model llava \
    --model_path ./outputs/llava_pathvqa \
    --dataset pathvqa \
    --split test \
    --output_file results_pathvqa.json

# Evaluation
python evaluate.py \
    --results_file results_pathvqa.json \
    --output_file metrics_pathvqa.json
```

### 3. Vergleich mit Base-Modell

```bash
# Inference mit Base-Modell (ohne Fine-Tuning)
python inference.py \
    --model llava \
    --dataset pathvqa \
    --split test \
    --output_file results_base.json

# Evaluation
python evaluate.py \
    --results_file results_base.json \
    --output_file metrics_base.json
```

## Modelle

### Default-Modelle

- **LLaVA**: `llava-hf/llava-1.5-7b-hf` (7B Parameter)
- **LLaVA-Med**: `microsoft/llava-med-v1.5-mistral-7b` (Medical-spezifisch)

Modelle werden beim ersten Aufruf automatisch von HuggingFace Hub heruntergeladen und dann lokal gecacht (`~/.cache/huggingface/hub/`).

### Lokale Modelle

Nach dem Training werden Modelle im `--output_dir` gespeichert und können direkt für Inference verwendet werden.

## Technische Details

### Label-Behandlung beim Training

Beim Training werden Labels so gesetzt, dass der Loss nur für Tokens nach `answer_token` berechnet wird:

```python
# Beispiel für Labels:
# Input:  [question_token; question; answer_token; answer]
# Labels: [-100, -100, ..., -100, token_id("answer"), ...]
#         ↑ ignoriert im Loss    ↑ nur diese zählen
```

Dies stellt sicher, dass das Modell nur lernt, Antworten zu generieren, nicht die Frage zu wiederholen.

### Prompt-Formatierung

Das Projekt verwendet ein einheitliches Prompt-Format für beide Modelle:

- **Training**: `USER: <question> {question} <answer> {answer}\nASSISTANT: {answer}`
- **Inference**: `USER: <question> {question} <answer>\nASSISTANT:`

Die Tokens `<question>` und `<answer>` können über Parameter angepasst werden.

## Troubleshooting

### Modell wird nicht gefunden

- Prüfe, ob der Modellname korrekt ist
- Bei lokalen Pfaden: Prüfe, ob der Pfad existiert
- Bei HuggingFace Modellen: Prüfe Internetverbindung für ersten Download

### Dataset nicht gefunden

- Prüfe, ob `--data_root` korrekt gesetzt ist
- Prüfe, ob die Dataset-Struktur korrekt ist (siehe Datensätze-Sektion)
- Prüfe, ob Split-Namen korrekt sind (z.B. `validate` für SLAKE, nicht `validation`)

### Out of Memory

- Reduziere `--per_device_train_batch_size`
- Erhöhe `--gradient_accumulation_steps`
- Aktiviere `--fp16` oder `--bf16` für Mixed Precision

## Lizenz

Siehe LICENSE-Datei (falls vorhanden).

## Kontakt

Für Fragen oder Probleme, bitte ein Issue erstellen.

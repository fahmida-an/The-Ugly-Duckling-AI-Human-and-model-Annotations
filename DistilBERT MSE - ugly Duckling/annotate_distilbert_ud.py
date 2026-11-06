"""Score the 24 passages in story.txt with a local VA checkpoint."""
import csv
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

FOLDER = Path(__file__).resolve().parent
INPUT = FOLDER / "story.txt"
OUTPUT = FOLDER / "ugly_duckling_distilbert_annotations.csv"

# Illustrative anchors on the signed VA scale, not learned model parameters.
ANCHORS = {
    "Anger": (-0.7, 0.8),
    "Contempt": (-0.6, 0.1),
    "Disgust": (-0.8, 0.4),
    "Fear": (-0.8, 0.9),
    "Happiness": (0.8, 0.5),
    "Neutral": (0.0, 0.0),
    "Sadness": (-0.7, -0.4),
    "Surprise": (0.0, 0.8),
}


def convert_score(raw):
    
    if not math.isfinite(raw):
        raise ValueError(f"Non-finite model output: {raw}")

    score_01 = torch.nn.functional.hardsigmoid(
        torch.tensor(3.0 * raw)
    ).item()
    return 2.0 * score_01 - 1.0, abs(raw) > 1.0

def emotion_from_va(valence, arousal):
    return min(
        ANCHORS,
        key=lambda label: (valence - ANCHORS[label][0]) ** 2
        + (arousal - ANCHORS[label][1]) ** 2,
    )


def main():
    sections = INPUT.read_text(encoding="utf-8-sig").strip().split("\n\n")
    if len(sections) != 24 or any(not section.strip() for section in sections):
        raise ValueError("Expected 24 sections in story.txt, separated by one blank line.")

    tokenizer = AutoTokenizer.from_pretrained(FOLDER, local_files_only=True)
    model, loading = DistilBertForSequenceClassification.from_pretrained(
        FOLDER, local_files_only=True, output_loading_info=True
    )
    if loading["missing_keys"] or model.config.num_labels != 2:
        raise RuntimeError("Use the fine-tuned DistilBERT VA checkpoint with two outputs.")
    model.eval()

    token_limit = min(tokenizer.model_max_length, model.config.max_position_embeddings)
    rows = []
    with torch.inference_mode():
        for index, section in enumerate(sections, start=1):
            passage_id = f"S{index:03d}"
            passage_text = section.strip()
            inputs = tokenizer(passage_text, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > token_limit:
                raise ValueError(f"{passage_id} exceeds the model's {token_limit}-token limit.")

            # Match the output order in your working DistilBERT VA script.
            raw_valence, raw_arousal = model(**inputs).logits[0].tolist()
            valence, v_clipped = convert_score(raw_valence)
            arousal, a_clipped = convert_score(raw_arousal)
            rows.append([
                passage_id, raw_valence, raw_arousal, valence, arousal,
                emotion_from_va(valence, arousal), v_clipped, a_clipped, passage_text,
            ])

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Passage ID", "Raw valence", "Raw arousal", "Valence (-1 to +1)",
            "Arousal (-1 to +1)", "VA-derived emotion", "Valence clipped",
            "Arousal clipped", "Passage text",
        ])
        writer.writerows(rows)
    print(f"Saved {len(rows)} passages to {OUTPUT}")


if __name__ == "__main__":
    main()
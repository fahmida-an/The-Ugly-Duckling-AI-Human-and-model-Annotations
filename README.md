# The-Ugly-Duckling-AI-Human-and-model-Annotations

# prompt that use to get AI Annotations

Annotate the three passages below from The Ugly Duckling.

Rate the overall emotional tone conveyed by each passage.
Use the wording and events in the passage as evidence. Do not
predict how a particular reader would feel.

Give two scores from -1 to +1:
- Valence: -1 = very unpleasant, 0 = neutral, +1 = very pleasant.
- Arousal: -1 = very calm or inactive, 0 = moderately activated,
  +1 = highly activated or tense.

Use one decimal place. If the emotional tone changes within a
passage, give one overall rating and mention the shift in your reason.

Return one row per passage with these columns:
Passage ID | Valence | Arousal | Brief reason



# The Ugly Duckling — Text Emotion Annotations
 We divided the English version of *The Ugly Duckling* into **24 reading passages** and estimated the emotional tone of each passage using ChatGPT and a fine-tuned multilingual DistilBERT model.

Each method gives two scores:

| Dimension | −1 | 0 | +1 |
|---|---|---|---|
| **Valence** | Very unpleasant | Neutral | Very pleasant |
| **Arousal** | Very calm or inactive | Moderately activated | Highly activated or intense |

## Where to find the files

| File | What it contains |
|---|---|
| `AI Annotations/ugly_duckling_ai_annotations.csv` | ChatGPT scores, emotion labels, short explanations, and the full passage text |
| `DistilBERT MSE - ugly Duckling/story.txt` | The 24 story passages in reading order |
| `DistilBERT MSE - ugly Duckling/annotate_distilbert_ud.py` | The Python script that runs DistilBERT |
| `DistilBERT MSE - ugly Duckling/ugly_duckling_distilbert_annotations.csv` | DistilBERT scores, rule-based emotion labels, and the full passage text |

To view the results, download the two CSV files and open them in Excel. Match rows by **Passage ID**, from `S001` to `S024`.

## How we created the ChatGPT annotations

1. We grouped the story into 24 passages, keeping connected dialogue together and preserving the wording and order.
2. We asked ChatGPT to rate the **overall emotional tone** of each passage based on its wording and events. It gave one valence score and one arousal score, each rounded to one decimal place.
3. We asked for a short reason and, where relevant, an explanation of how the tone changed within the passage.
4. We asked ChatGPT to choose one of eight emotion labels using the passage text and the scores. The choices were **anger, contempt, disgust, fear, happiness, neutral, sadness, and surprise**.

## How we ran DistilBERT

The script reads the same 24 passages from `story.txt`. It assigns IDs `S001` to `S024` in order and writes the model results to a separate CSV. It uses the story text as input and generates its own scores independently of the ChatGPT ratings.

### Run it on Windows

1. Clone or download this repository and open it in VS Code.
2. Right-click the **`DistilBERT MSE - ugly Duckling`** folder in VS Code and choose **Open in Integrated Terminal**. Use PowerShell for the commands below.
3. Download the **DistilBERT MSE** checkpoint and place `pytorch_model.bin` in that folder, next to `config.json`, `story.txt`, and `annotate_distilbert_ud.py`. [Authors’ model downloads](https://drive.google.com/drive/folders/1a3ToFHaGQxxAPI4dXc_shUrjROj7OrKt?usp=share_link)
4. With Python 3.11 installed, create a virtual environment and install the dependencies. You only need to do this once.

   ```powershell
   py -3.11 -m venv .venv
   & ".\.venv\Scripts\python.exe" -m pip install "torch>=2.6" "transformers>=4.50,<5"
   ```

5. Run the script. If the results CSV is already open in Excel, close it first so the script can write the new results.

   ```powershell
   & ".\.venv\Scripts\python.exe" ".\annotate_distilbert_ud.py"
   ```

6. When the terminal prints `Saved 24 passages`, open the results.

   ```powershell
   Invoke-Item ".\ugly_duckling_distilbert_annotations.csv"
   ```

If the script reports the wrong number of passages, check that `story.txt` contains only the story text, with **one blank line between passages**. The script adds the passage IDs automatically.

## What the DistilBERT columns mean

- **Raw valence** and **Raw arousal** are the numbers from the model before activation. These numbers can be negative.
- **Valence (-1 to +1)** and **Arousal (-1 to +1)** are the final scores. The script applies the authors’ `hardsigmoid(3 × raw output)` to get a score on the 0–1 scale. It then applies `2 × score − 1` to put the result on the same scale as the ChatGPT ratings. For example, a raw valence of `−0.46` becomes about `0.27` on the 0–1 scale and then `−0.46` on the final scale.
- **Valence clipped** and **Arousal clipped** are `TRUE` when the raw output is outside −1 to +1 and the activation limits the final score to a boundary. `FALSE` means the raw output was within that interval.
- **VA-derived emotion** is an approximate label assigned by the script. It picks the closest of eight example emotion points using the final valence and arousal scores. ChatGPT also considers the passage wording when choosing its label.

## How to compare the two sets of results

1. Match each passage by ID and check that the passage text is the same in both CSVs.
2. Compare the final **valence** scores and final **arousal** scores separately.
3. Look at passages where the scores disagree and read the ChatGPT explanation alongside the story text.

Both CSVs contain estimates of the **text’s emotional tone**. Independent human ratings of the same passages can help us assess how each method agrees with human judgments. Reader self-reports and webcam estimates are separate measures of the reader’s response.

## Model reference

Gonçalo Azevedo Mendes and Bruno Martins (2023). *Quantifying Valence and Arousal in Text with Multilingual Pre-trained Transformers*.

[Paper PDF](https://arxiv.org/pdf/2302.14021) · [Authors’ GitHub repository](https://github.com/gmendes9/multilingual_va_prediction)
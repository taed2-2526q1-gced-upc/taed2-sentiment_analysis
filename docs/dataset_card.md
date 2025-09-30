# Dataset Card — RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

**Homepage:** https://zenodo.org/record/1188976  
**Repository:** Distributed via Zenodo (no GitHub repository)  
**Paper:** Livingstone, S.R. & Russo, F.A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLOS ONE 13(5): e0196391.  
**Leaderboard:** None available.  
**Point of Contact:** Steven R. Livingstone (Ryerson University), steven.livingstone@ryerson.ca (as listed in the original paper).

## Dataset Summary
RAVDESS is an audio dataset containing speech and song recordings annotated with eight emotional categories. It includes 24 professional actors (12 female, 12 male) performing two lexically neutral sentences in North American English, recorded under controlled studio conditions.  
In our project, we use the **speech subset** to train models that predict emotions from audio.

## Supported Tasks and Leaderboards
### `speech-emotion-recognition`
- **Task:** Classify an audio clip into one of 8 discrete emotion labels (neutral, calm, happy, sad, angry, fearful, surprise, disgust).  
- **Metrics:** Unweighted Average Recall (UAR), accuracy, macro-F1.  
- **Suggested models:** CNNs over log-Mel spectrograms, MFCC+SVM, CRNNs.

### `automatic-speech-recognition` *(planned use)*
- **Task:** Transcribe the two spoken sentences (“Kids are talking by the door” / “Dogs are sitting by the door”).  
- **Metrics:** Word Error Rate (WER).  
- **Suggested models:** Whisper (tiny/base), wav2vec2, Vosk.

No leaderboard is currently maintained for this dataset.

## Languages
- **Language:** English (North American), BCP-47 code: `en-US`.  
- **Domain:** Acted studio speech (not conversational or spontaneous).  
- **Register:** Controlled, scripted, semantically neutral utterances.

## Dataset Structure
### Data Instances
Example instance metadata:
```json
{
  "filename": "03-01-05-02-02-02-12.wav",
  "actor_id": 12,
  "gender": "male",
  "modality": "speech",
  "emotion": "angry",
  "intensity": "strong",
  "statement": "Dogs are sitting by the door",
  "repetition": 2
}
```

### Data Fields
- `filename` *(string)*: path to the audio file.  
- `actor_id` *(int)*: ID of the actor (1–24).  
- `gender` *(string)*: gender of the speaker (male/female).  
- `modality` *(string)*: speech or song.  
- `emotion` *(string)*: one of 8 categories.  
- `intensity` *(string)*: normal or strong.  
- `statement` *(string)*: spoken sentence (2 possible options).  
- `repetition` *(int)*: repetition index (1 or 2).

### Data Splits
The dataset is released as a whole without predefined splits. In our project we defined a **speaker-independent** split:  
- **Train:** 70% of actors  
- **Validation:** 10% of actors  
- **Test:** 20% of actors

## Dataset Creation
### Curation Rationale
Provide a balanced, studio-quality benchmark for emotion recognition in speech and song.

### Source Data
- **Collection:** Recorded in a professional sound studio.  
- **Protocol:** Two fixed sentences delivered by each actor under multiple emotional conditions.  
- **Quality:** Normalized audio levels and consistent recording quality.

### Source Language Producers
- **24 professional actors** (12 female, 12 male), native North American English speakers, compensated.

### Annotations
- **Process:** Emotions were acted following a script.  
- **Validation:** An independent perception validation study confirmed recognizability.  
- **Annotators:** Human participants in validation studies; demographics not fully disclosed in the paper.

## Personal and Sensitive Information
- Contains voice data from actors but no personally identifying information beyond metadata.  
- Gender is labeled explicitly (male/female).  
- No sensitive data such as political, financial, or medical information.

## Considerations for Using the Data
### Social Impact
- **Positive:** Enables research in HCI, emotion-aware systems, accessibility, and education.  
- **Risks:** Potential misuse for surveillance, profiling, or inferring psychological states.

### Discussion of Biases
- Limited to 24 actors, North American English only.  
- **Acted** (not spontaneous) emotions → may differ from real-world expression.  
- Only two sentences → limited lexical diversity.

### Other Known Limitations
- Not robust to noisy, cross-lingual, or naturalistic data.  
- Common confusions: fear vs. surprise, sad vs. calm.

## Additional Information
### Dataset Curators
Steven R. Livingstone and Frank A. Russo, Ryerson University (now Toronto Metropolitan University).

### Licensing Information
Distributed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

### Citation
```bibtex
@article{livingstone2018ravdess,
  author    = {Livingstone, Steven R. and Russo, Frank A.},
  title     = {The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  journal   = {PLOS ONE},
  year      = {2018},
  volume    = {13},
  number    = {5},
  pages     = {e0196391},
  doi       = {10.1371/journal.pone.0196391}
}
```

### Contributions
Thanks to @livingstone and @russo for creating and releasing the dataset. Adapted for use in our project by **Tomàs, Arnau, Mathys and Arnau**.

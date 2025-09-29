# Model Card for RAVDESS Speech Emotion Recognition Model

## Model Summary
This model performs emotion recognition from speech audio, classifying recordings into 8 categories: **neutral, calm, happy, sad, angry, fearful, surprise, disgust**.  
It is trained on the **RAVDESS** speech subset (studio-quality recordings from professional actors).

## Model Details
- **Developed by:** Tomàs, Arnau, Mathys and Arnau (Data Science Project Team)  
- **Model type:** Convolutional Neural Network (CNN) or Convolutional Recurrent Neural Network (CRNN) *(TBD during development)*  
- **Language(s):** English (North American)  
- **Task:** Speech Emotion Recognition  
- **Domain:** Acted speech audio  
- **Development Status:** In planning phase  
- **Repository:** https://github.com/Arnaubiosca15/taed2-cars  
- **Dataset:** See the accompanying *RAVDESS Dataset Card*  
- **Reference paper:** Livingstone & Russo (2018), PLoS ONE 13(5): e0196391.

## Intended Uses
### Direct Use
- Classify emotion in English speech recordings for **research** purposes.  
- Serve as a **baseline** for affective computing.  
- Educational demonstrations.  
- Prototype development for emotion-aware applications.

### Downstream Use
- Fine-tuning or integration into voice assistants, mental health monitoring (with appropriate validation), educational systems, or accessibility tools.

### Out-of-Scope Use
- Real-time **clinical diagnosis** or mental health assessment.  
- **Surveillance** or profiling without consent.  
- High-stakes decisions.  
- **Non-English** languages or spontaneous speech without validation.  
- Noisy/low-quality audio; cross-cultural recognition w/o validation.

## Bias, Risks, and Limitations
### Key Limitations
- Trained on **acted** emotions; may differ from spontaneous emotions.  
- Limited to **24 actors** (North American English).  
- Only **two sentences**, limiting lexical diversity.  
- May not generalize to real-world, noisy speech.  
- Potential confusion between similar classes (fear/surprise, sad/calm).

### Bias Considerations
- Gender representation is balanced (binary), but limited.  
- Cultural bias toward **North American** expression patterns.  
- Studio recording conditions differ from real-world audio.

### Risks
- Misclassification could lead to misunderstanding of emotional states.  
- Potential misuse for unauthorized emotional profiling.  
- Risk of perpetuating stereotypes.

### Recommendations for Users
- Validate on your target population/use case.  
- Consider acted vs. spontaneous differences and cultural context.  
- Use **confidence thresholds** / uncertainty estimation.  
- Communicate limitations to end users and implement safeguards.

## How to Get Started
> *To be added after implementation.*  
> Final API may differ based on design decisions.

## Training Details
### Training Data
RAVDESS speech subset: 1,440 audio files (24 actors × 2 sentences × 8 emotions × 2 intensities × 2 repetitions).  
Balanced gender (12 male, 12 female); studio-quality recordings.

**Proposed speaker-independent split:**  
- Train: 70% of actors  
- Validation: 10%  
- Test: 20%

### Preprocessing *(planned)*
- Resample to 22,050 Hz or 16,000 Hz  
- Features: **log-Mel spectrograms** (primary), **MFCCs** (baseline), or raw audio  
- Normalization to be evaluated  
- Augmentations under consideration: time-stretch, pitch-shift, background noise, **SpecAugment**

### Hyperparameters *(TBD)*
- Optimizer: Adam or SGD  
- Loss: Cross-Entropy  
- LR / batch size / epochs: to be tuned  
- Precision: fp32 or mixed precision depending on HW

### Speeds, Sizes, Times *(estimated)*
- Model size & training time: TBD by architecture/HW  
- **Inference target:** near real-time

## Evaluation
### Protocol
- **Speaker-independent** test set (20% actors). No actor overlap.  
- Disaggregate by **gender**, **emotion**, **intensity**, **statement**, and **actor groups**.

### Metrics
- **Unweighted Average Recall (UAR)** *(primary)*  
- **Accuracy**, **Macro F1-score**  
- Per-class **precision/recall** and **confusion matrix**

### Target Performance Goals
- UAR > 70%  
- Accuracy > 75%  
- Macro F1 > 70%

> **Results:** placeholders to be filled after training.

## Environmental Impact *(planned)*
- Carbon estimated with ML Impact calculator.  
- Compute region: Barcelona, Spain (local development).  
- Consider model compression and efficiency when possible.

## Technical Specifications *(planned)*
### Architecture Options
- Primary: **CNN** on log-Mel spectrograms  
- Alternative: **CRNN** (CNN + LSTM/GRU)  
- Baseline: **MFCC + SVM**  
- Input: e.g., 128 mel bands (window/hop TBD)  
- Output: 8-class probabilities (Cross-Entropy)

### Compute Infrastructure
- **Framework:** PyTorch (primary) or TensorFlow  
- **Libs:** librosa, scikit-learn, numpy, pandas  
- **Python:** 3.10+  
- **Tools:** Jupyter, **DVC** for data versioning

## More Information
**Status:** Planning & design phase.  
**Next steps:** (1) Data prep (2) Baseline (3) Arch experiments (4) HPO (5) Evaluation.  
**Authors:** Tomàs, Arnau, Mathys, Arnau.

> *Disclaimer: Draft model card for a project in development. Specs and targets may change during implementation.*

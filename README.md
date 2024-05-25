
# Voice Cloning Project Plan Document

## Introduction
The goal of this project is to create a system that can clone a voice by capturing the unique characteristics of a speaker's voice and synthesizing new audio that sounds like the original speaker. This document outlines the approach, key technical decisions, and core components of the project.

## Data Collection and Preprocessing

### Data Sources
The LJ Speech Dataset consists of 13,100 audio clips of a single speaker reading passages from seven public domain non-fiction books. For this project, only 100 clips are selected. Each clip lasts 1 to 10 seconds. Metadata includes transcriptions and normalized transcriptions of the spoken words. The dataset is in the public domain and useful for training speech synthesis models.

### Preprocessing Steps

#### Initial Setup and Dependencies
- Ensure that the required Python packages are installed.
- Unzip the `wavs.zip` file containing the audio data.
- Import necessary libraries for audio processing (`os`, `shutil`, `taglib`, `torch`, `torchaudio`, `librosa`, `soundfile`, `transformers`).

#### Define Paths
- Set the paths for input audio files, output preprocessed files, and metadata updated files.
- Ensure that the output directories exist, creating them if necessary.

#### Renaming .wav Files
- Traverse through the input directory and rename all `.wav` files in a sequential numerical order (e.g., 1.wav, 2.wav, etc.) to organize and manage the audio files systematically.

#### Preprocessing .wav Files
1. **Load the Audio:** Use the `librosa.load` function to load the audio file.
2. **Trim Silence:** Utilize `librosa.effects.trim` to remove silence from the beginning and end of the audio.
3. **Normalize Audio:** Normalize the audio volume to ensure consistency across all files.
4. **Save the Processed Audio:** Save the preprocessed audio to the output directory using `soundfile.write`.

#### Metadata Handling
- Create and update metadata for each audio file using `taglib`.
- Save the updated metadata information in a specified metadata directory.

#### Logging
- Maintain a log of all processed files in a `list.txt` file within the metadata directory.

**Code:** [`data_prepro.ipynb`](data_prepro.ipynb)

## Model Selection

### Model Architectures Compared
- **XTTS:** 
  - Strengths: Often more modular, allowing for easier experimentation with different components.
  - Weaknesses: Can be more complex to train due to modularity, requiring careful tuning of each part.
- **WaveNet:** 
  - Strengths: Produces high-quality, natural-sounding audio.
  - Weaknesses: Computationally intensive, slow to train and generate audio due to autoregressive nature.
- **StyleTTS2:** 
  - Strengths: Focuses on style transfer and expressive speech synthesis.
  - Weaknesses: More complex and may require additional data for style embeddings.
- **Transformer-Based Models:** 
  - Strengths: Excellent at capturing long-range dependencies in data, potentially improving prosody and coherence.
  - Weaknesses: Can be resource-intensive and require large datasets to achieve optimal performance.

### Chosen Model
- **Model Name:** Tacotron2 for training, Tacotron2 + HiFi-GAN for inference
- **Justification:** Tacotron2 is selected due to its balance of performance, computational efficiency, and ease of training. It integrates well with HiFi-GAN, which is used for converting Mel spectrograms into high-quality audio waveforms. This combination ensures high-quality audio synthesis while remaining relatively straightforward to implement and train compared to other architectures.

**Code:** [`Train_Voices.ipynb`](Train_Voices.ipynb)

## Training Pipeline

### Data Preparation
- Collect high-quality audio data along with corresponding transcriptions.
- Preprocess data to ensure consistency in sampling rate, volume normalization, and noise reduction.
- Extract features such as Mel spectrograms from the audio files.

### Model Architecture
- **Tacotron2:** An encoder-decoder architecture with attention mechanisms that convert text sequences into Mel spectrograms.
  - **Encoder:** Processes input text into a hidden representation.
  - **Decoder with Attention:** Converts hidden representation into Mel spectrogram frames, attending to relevant parts of the text sequence.

### Training Process
- **Loss Functions:** Define a combination of L1 loss for Mel spectrogram prediction and binary cross-entropy for stop token prediction.
- **Optimizers:** Use Adam optimizer with appropriate learning rates and schedules.
- **Teacher Forcing:** Implement teacher forcing during initial training stages, gradually reducing its use as the model learns.

### Post-Processing
- Use a vocoder (e.g., WaveGlow) to convert predicted Mel spectrograms into audible waveforms.
- Evaluate synthesized audio quality using metrics like Mean Opinion Score (MOS), Mel Cepstral Distortion (MCD), and similarity to the original voice.

**Code:** [`Train_Voices.ipynb`](Train_Voices.ipynb)

## Voice Synthesis and Evaluation

### Synthesis Process
- The Tacotron2 model processes the input text and generates a Mel spectrogram. This represents the frequency content of the audio over time.
- Apply super-resolution to the Mel spectrogram to improve quality and detail.
- Use HiFi-GAN, a generative adversarial network, to convert the enhanced Mel spectrogram into a waveform. This model generates high-quality audio that matches the input spectrogram.
- The generated waveform is saved as a `.wav` file and can be played back to verify the quality and accuracy of the synthesized voice.

### Evaluation Metrics
- **Mean Opinion Score (MOS):** Subjective listening tests where human listeners rate the naturalness and quality of the synthesized speech on a scale (e.g., 1 to 5).
- **Mel Cepstral Distortion (MCD):** Objective metric measuring the difference between the Mel frequency cepstral coefficients (MFCCs) of the original and synthesized speech. Lower values indicate higher similarity.
- **Perceptual Evaluation of Speech Quality (PESQ):** Algorithmic evaluation that provides a score reflecting the perceived audio quality, considering factors like distortion and noise.
- **Word Error Rate (WER):** Measures the accuracy of speech recognition systems by comparing the transcriptions of synthesized speech against reference transcriptions. Lower WER indicates better intelligibility.
- **Short-Time Objective Intelligibility (STOI):** Metric assessing the intelligibility of speech, particularly in noisy environments. Higher STOI scores indicate clearer speech.

**Code:** [`Infer_voices.ipynb`](Infer_voices.ipynb)

## Deployment Strategy

### Containerization
- The model is containerized into a Docker container and available on Docker Hub. It can be tested using the `docker-compose` file in the repository.
- The Docker container includes a Flask API for serving the model and handling inference requests.

### Deployment as Web App
- **Flask API:** The Flask API handles incoming requests, processes the input text, runs the inference using Tacotron2 and HiFi-GAN, and returns the synthesized audio.
- **Web Interface:** A simple web interface (`index.html`) allows users to input text and receive synthesized audio in real-time.
- **Docker Configuration:** The Dockerfile and `docker-compose.yml` ensure that the application can be easily deployed and scaled.

### Potential Platforms
- **Cloud Services:** Suitable for deployment on platforms such as AWS, Google Cloud Platform (GCP), or Azure for scalable deployment.
- **Edge Devices:** Deployable on edge devices like Raspberry Pi for on-device inference in low-latency applications.

### Scalability and Performance
- **Scalability:** Use containerization (Docker) and orchestration (Kubernetes) for scalable deployments.
- **Performance Concerns:** Optimize model inference speed and reduce latency with efficient data pipelines.

**Code:**
- [`dockerfile`](dockerfile)
- [`docker-compose.yml`](docker-compose.yml)
- [`app.py`](app.py)
- [`inference.py`](inference.py)
- [`requirements.txt`](requirements.txt)
- [`index.html`](index.html)

## Ethical Considerations and Mitigation

### Potential Misuse
- Unauthorized voice cloning could lead to identity theft, fraud, or defamation.
- Synthesized voices could be used to create fake audio clips for malicious purposes, such as spreading misinformation or impersonating individuals.

### Safeguards
- **Usage Monitoring:** Implement logging and monitoring to track the use of the voice cloning system. Analyze logs to detect and prevent misuse.
- **Watermarking:** Embed inaudible watermarks in the synthesized audio to trace and identify misuse.
- **Access Control:** Restrict access to the voice cloning system to authorized users only. Implement authentication and authorization mechanisms.

### Privacy and Consent
- **Data Anonymization:** Anonymize all voice data used for training and inference to protect the privacy of individuals.
- **Consent:** Obtain explicit consent from individuals before using their voice data for cloning. Ensure that users are aware of how their data will be used and provide options to opt-out.

### Legal and Ethical Guidelines
- **Compliance:** Ensure that the voice cloning system complies with relevant legal and ethical guidelines, such as GDPR for data protection and privacy.
- **Transparency:** Maintain transparency with users about the capabilities and limitations of the voice cloning system. Provide clear disclaimers about the potential risks and ethical considerations.

## Prototype Implementation

### Core Component
The core component implemented includes the data preprocessing scripts, model training code, and a basic synthesis script to demonstrate voice cloning.
![Screenshot 2024-05-25 185311](https://github.com/DhruvPatel96/voices/assets/80629263/0c8f3591-a8b0-4eaf-9e9c-53323568c3e1)


**Files:**
- [`data_prepro.ipynb`](data_prepro.ipynb)
- [`Train_Voices.ipynb`](Train_Voices.ipynb)
- [`Infer_voices.ipynb`](Infer_voices.ipynb)
- [`dockerfile`](dockerfile)
- [`docker-compose.yml`](docker-compose.yml)
- [`app.py`](app.py)
- [`index.html`](index.html)
- [`inference.py`](inference.py)
- [`requirements.txt`](requirements.txt)

## References
- [ARPAtaco2](https://github.com/justinjohn0306/ARPAtaco2)
- [NVIDIA Tacotron2](https://github.com/NVIDIA/tacotron2)
- [gdown](https://github.com/IAHispano/gdown)
- [TTS-TT2](https://github.com/justinjohn0306/TTS-TT2)
- [HiFi-GAN](https://github.com/justinjohn0306/hifi-gan)
- [FakeYou Tacotron2 Notebook](https://github.com/justinjohn0306/FakeYou-Tacotron2-Notebook)
- [ARPAtaco2](https://github.com/justinjohn0306/ARPAtaco2.git)
- [TTS-TT2](https://github.com/justinjohn0306/TTS-TT2.git)
- [Tacotron2](https://github.com/justinjohn0306/tacotron2)
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)


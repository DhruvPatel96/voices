import logging
import os
import sys
import torch
import numpy as np
import json
import resampy
import scipy.signal
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm
import gdown
import re


# Define the AttrDict class
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# Constants
MAX_WAV_VALUE = 32768.0

# Set Matplotlib backend to non-interactive Agg
matplotlib.use('Agg')

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)

# Local directory path for Tacotron2 model
tacotron_local_path = "./first_model"  # Change this to your local model path
output_directory = "./output"  # Change this to your desired output directory path

HIFIGAN_ID = "universal"  # Leave blank or enter "universal" for universal model

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clone the HiFi-GAN repository if it doesn't exist
if not os.path.exists('hifi-gan'):
    os.system('git clone https://github.com/justinjohn0306/hifi-gan.git')

# Clone the TTS-TT2 repository if it doesn't exist
if not os.path.exists('TTS-TT2'):
    os.system('git clone https://github.com/justinjohn0306/TTS-TT2.git')

# Append the paths to sys.path
sys.path.append('hifi-gan')
sys.path.append('TTS-TT2')

# Install TensorFlow, inflect, and IPython if not installed
try:
    import tensorflow as tf
except ImportError:
    os.system('pip install tensorflow')
    import tensorflow as tf

try:
    import inflect
except ImportError:
    os.system('pip install inflect')
    import inflect

try:
    from IPython.display import Audio, display
except ImportError:
    os.system('pip install ipython')
    from IPython.display import Audio, display

from models import Generator
from denoiser import Denoiser
from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
from meldataset import mel_spectrogram  # Import mel_spectrogram

# Setup Pronunciation Dictionary
gdown.download(
    'https://github.com/justinjohn0306/FakeYou-Tacotron2-Notebook/releases/download/CMU_dict/merged.dict.txt',
    'merged.dict.txt', quiet=False)
thisdict = {}
for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
    thisdict[(line.split(" ", 1))[0]] = (line.split(" ", 1))[1].strip()


def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word = word_
        end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation:
                end_chars = word[-1] + end_chars
                word = word[:-1]
            else:
                break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError:
            pass
        out = (out + " " + word + end_chars).strip()
    if EOS_Token and out[-1] != ";":
        out += ";"
    return out


def get_hifigan(MODEL_ID, conf_name):
    # Download HiFi-GAN
    hifigan_pretrained_model = 'hifimodel_' + conf_name
    if MODEL_ID == 1:
        gdown.download('https://github.com/justinjohn0306/tacotron2/releases/download/assets/Superres_Twilight_33000',
                       hifigan_pretrained_model, quiet=False)
    elif MODEL_ID == "universal":
        gdown.download('https://github.com/justinjohn0306/tacotron2/releases/download/assets/g_02500000',
                       hifigan_pretrained_model, quiet=False)
    else:
        gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', hifigan_pretrained_model, quiet=False)

    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    if not os.path.isfile(conf):
        raise FileNotFoundError(f"Configuration file {conf_name}.json not found in hifi-gan directory.")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(device)
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=device)
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser


# Download character HiFi-GAN
hifigan, h, denoiser = get_hifigan(HIFIGAN_ID, "config_v1")
# Download super-resolution HiFi-GAN
hifigan_sr, h2, denoiser_sr = get_hifigan(1, "config_32k")


def has_MMI(STATE_DICT):
    return any(True for x in STATE_DICT.keys() if "mi." in x)


def get_Tactron2(model_path):
    # Load Tacotron2 and Config
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000  # Max Duration
    hparams.gate_threshold = 0.25  # Model must be 25% sure the clip is over before ending generation
    model = Tacotron2(hparams)
    state_dict = torch.load(model_path, map_location=device)['state_dict']
    if has_MMI(state_dict):
        raise Exception("ERROR: This notebook does not currently support MMI models.")
    model.load_state_dict(state_dict)
    _ = model.to(device).eval().half()
    return model, hparams


model, hparams = get_Tactron2(tacotron_local_path)
previous_tt2_id = tacotron_local_path


# Function to sanitize file names
def sanitize_filename(text):
    # Remove non-alphanumeric characters
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Limit length to avoid overly long file names
    return sanitized[:50]


# Extra Info
def end_to_end_infer(text, pronounciation_dictionary, show_graphs):
    if not pronounciation_dictionary:
        if text[-1] != ";":
            text = text + ";"
    else:
        text = ARPA(text)

    with torch.no_grad():  # save VRAM by not including gradients
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        if show_graphs:
            plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                       alignments.float().data.cpu().numpy()[0].T))
        y_g_hat = hifigan(mel_outputs_postnet.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]

        # Resample to 32k
        audio_denoised = audio_denoised.cpu().numpy().reshape(-1)

        normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
        audio_denoised = audio_denoised * normalize
        wave = resampy.resample(
            audio_denoised,
            h.sampling_rate,
            h2.sampling_rate,
            filter="sinc_window",
            window=scipy.signal.windows.hann,
            num_zeros=8,
        )
        wave_out = wave.astype(np.int16)

        # HiFi-GAN super-resolution
        wave = wave / MAX_WAV_VALUE
        wave = torch.FloatTensor(wave).to(device)
        new_mel = mel_spectrogram(
            wave.unsqueeze(0),
            h2.n_fft,
            h2.num_mels,
            h2.sampling_rate,
            h2.hop_size,
            h2.win_size,
            h2.fmin,
            h2.fmax,
        )
        y_g_hat2 = hifigan_sr(new_mel)
        audio2 = y_g_hat2.squeeze()
        audio2 = audio2 * MAX_WAV_VALUE
        audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]

        # High-pass filter, mixing and denormalizing
        audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
        b = scipy.signal.firwin(
            101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
        )
        y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
        y *= superres_strength
        y_out = y.astype(np.int16)
        y_padded = np.zeros(wave_out.shape)
        y_padded[: y_out.shape[0]] = y_out
        sr_mix = wave_out + y_padded
        sr_mix = sr_mix / normalize

        # Save the audio file
        sanitized_text = sanitize_filename(text)
        output_path = os.path.join(output_directory, f"{sanitized_text}.wav")
        write_wav(output_path, h2.sampling_rate, sr_mix.astype(np.int16))

        print(f"Audio saved at: {output_path}")
        display(Audio(sr_mix.astype(np.int16), rate=h2.sampling_rate))


def plot_data(data, figsize=(9, 3.6)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', interpolation='none', cmap='inferno')
    plt.savefig(os.path.join(output_directory, "plot.png"))  # Save the plot as an image file


# Main Loop
pronounciation_dictionary = False  # Disables automatic ARPAbet conversion
show_graphs = True  # Show graphs
max_duration = 20  # Max duration in seconds
model.decoder.max_decoder_steps = max_duration * 80
stop_threshold = 0.5  # Stop threshold
model.decoder.gate_threshold = stop_threshold
superres_strength = 10  # Super-resolution strength

print(
    f"Current Config:\npronounciation_dictionary: {pronounciation_dictionary}\nshow_graphs: {show_graphs}\nmax_duration (in seconds): {max_duration}\nstop_threshold: {stop_threshold}\nsuperres_strength: {superres_strength}\n\n")

# time.sleep(1)
# print("Enter/Paste your text.")
# while True:
#     try:
#         print("-" * 50)
#         line = input().strip()
#         if line:
#             end_to_end_infer(line, not pronounciation_dictionary, show_graphs)
#     except EOFError:
#         break
#     except KeyboardInterrupt:
#         print("Stopping...")
#         break

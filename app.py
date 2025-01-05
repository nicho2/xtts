import sys
import os
import subprocess
import random
import uuid
import time
import torch
import torchaudio

# Coqui TTS
# pip install TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

# LangID (pour la détection de la langue sur du texte plus long)
# pip install langid
import langid

import chardet
import re
from tqdm import tqdm

# Gradio pour l’interface Web locale
# pip install gradio
import gradio as gr
from scipy.io.wavfile import write
from pydub import AudioSegment

###################################################
# Partie 1 : Chargement du modèle et configuration
###################################################

# Spécifiez le nom du modèle Coqui TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

print("Téléchargement du modèle (si nécessaire) : ", model_name)
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

print("Chargement de la configuration...")
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

print("Initialisation du modèle XTTS...")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(model_path, "model.pth"),
    vocab_path=os.path.join(model_path, "vocab.json"),
    eval=True,
    use_deepspeed=True,
)
model.cuda()

supported_languages = config.languages
print("Langues supportées :", supported_languages)

###################################################
# Partie 2 : Fonction de prédiction (TTS + Voice Clone)
###################################################


def split_into_sentences(text):
    # Découpe approximative par ponctuation
    # et préserve la ponctuation finale dans chaque morceau
    sentences = re.split(r'([.!?])', text)
    # sentences renvoie quelque chose comme ["Ceci est une phrase", ".", " Ceci est une autre phrase", ".", ""]
    # Il faut recoller la ponctuation
    merged = []
    for i in range(0, len(sentences)-1, 2):
        merged.append(sentences[i].strip() + sentences[i+1])
    # 3) Nettoyage des guillemets et des sauts de ligne
    final_sentences = []
    for s in merged:
        # Retire espaces superflus
        s = s.strip()
        if not s:
            continue

        # a) Supprimer les guillemets (ici " normaux, éventuellement “ ” « » si besoin)
        s = re.sub(r'[\"“”«»]', '', s)

        # b) Remplacer les doubles sauts de ligne (ou multiples) par des ":"
        #    Ici, \n{2,} capture deux ou plus retours à la ligne successifs
        s = re.sub(r'\n{2,}', '..', s)

        # c) Supprimer les retours à la ligne restants
        s = s.replace('\n', '')

        # On retire les espaces finaux à nouveau
        s = s.strip()
        if s:
            final_sentences.append(s)

    return final_sentences


def predict(
    prompt,
    language,
    audio_file_pth,
    mic_file_path,
    use_mic,
    voice_cleanup,
    no_lang_auto_detect,
    agree,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Fonction principale pour générer la voix à partir du texte et d’un échantillon audio.
    """
    if not agree:
        # L’utilisateur doit accepter la licence Coqui TTS (CPML)
        gr.Warning("Veuillez accepter les conditions d’utilisation (CPML).")
        return None, None, None, None

    # Contrôle si la langue choisie est supportée
    if language not in supported_languages:
        gr.Warning(
            f"Langue '{language}' non supportée. Langues disponibles : {supported_languages}"
        )
        return None, None, None, None

    # Détection automatique de la langue dans le texte
    # La détection n’est faite que si le texte dépasse 15 caractères et
    # si l'utilisateur n'a pas coché "Do not use language auto-detect".
    language_predicted = langid.classify(prompt)[0].strip()
    if language_predicted == "zh":
        language_predicted = "zh-cn"  # Ajustement pour le chinois

    print(f"Langue détectée : {language_predicted}, langue choisie : {language}")

    if len(prompt) > 15 and not no_lang_auto_detect:
        if language_predicted != language:
            gr.Warning(
                "Le texte ne semble pas correspondre à la langue sélectionnée."
                " Cochez 'Do not use language auto-detect' si vous êtes sûr(e)."
            )
            return None, None, None, None

    # Détermine le fichier de référence (micro ou fichier uploadé)
    if use_mic:
        if mic_file_path is not None:
            speaker_wav = mic_file_path
        else:
            gr.Warning("Aucun fichier micro trouvé. Décochez 'Use Microphone' ou enregistrez.")
            return None, None, None, None
    else:
        speaker_wav = audio_file_pth

    # Nettoyage éventuel du fichier de référence audio
    if voice_cleanup and speaker_wav is not None:
        try:
            out_filename = speaker_wav + "_" + str(uuid.uuid4()) + ".wav"
            # Filtrage simple avec ffmpeg (nécessite ffmpeg installé localement)
            lowpass_highpass = "lowpass=8000,highpass=75,"
            trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
            shell_command = (
                f"ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}"
            ).split()
            subprocess.run(shell_command, capture_output=True, check=True)
            speaker_wav = out_filename
            print("Référence audio nettoyée avec succès.")
        except subprocess.CalledProcessError as e:
            print("Problème lors du nettoyage audio, utilisation du fichier original.")
            print(e)

    # Vérification de la longueur du prompt
    if len(prompt) < 2:
        gr.Warning("Le texte est trop court. Veuillez entrer un texte plus long.")
        return None, None, None, None
    #if len(prompt) > 200:
    #    gr.Warning("Limite de 200 caractères dépassée pour cette démonstration.")
    #    return None, None, None, None

    metrics_text = ""
    try:
        # Extraction de l'embedding du haut-parleur
        t_latent = time.time()
        (
            gpt_cond_latent,
            speaker_embedding,
        ) = model.get_conditioning_latents(
            audio_path=speaker_wav,
            gpt_cond_len=30,
            gpt_cond_chunk_len=4,
            max_ref_length=60
        )
        latent_calculation_time = time.time() - t_latent

        # Correction provisoire ponctuation
        #prompt = re.sub(r"([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", prompt)
        #print(prompt)
        # Génération audio
        print("Génération de la voix...")
        # 1) Découper le texte
        chunks = split_into_sentences(prompt)  # ou split_into_sentences
        print(chunks)
        # 2) Pour chaque segment, générer l’audio
        wav_out_list = []

        t0 = time.time()

        for segment in tqdm(chunks, desc="Synthèse TTS", unit="segment", colour="green"):
            t1 = time.time()
            out = model.inference(
                segment,
                language,
                gpt_cond_latent,
                speaker_embedding,
                repetition_penalty=5.0,
                temperature=0.75,
            )
            inference_time = time.time() - t1
            print(f"Temps de génération audio (ms) : {round(inference_time*1000)}  ")
            real_time_factor = (time.time() - t1) / out['wav'].shape[-1] * 24000
            print(f"Facteur temps réel (RTF) : {real_time_factor:.2f}\n")

            #progress(i + 1, total=len(chunks), desc=f"Synthèse segment {i + 1}/{len(chunks)}")

            wav_out_list.append(torch.tensor(out["wav"]))

        inference_time = time.time() - t0
        metrics_text += f"Temps de génération audio (ms) : {round(inference_time*1000)}\n"

        # 3) Concaténer l’audio si nécessaire
        if len(wav_out_list) == 1:
            final_wav = wav_out_list[0]
        else:
            final_wav = torch.cat(wav_out_list, dim=0)

        # Sauvegarde du fichier WAV
        torchaudio.save("output.wav", final_wav.unsqueeze(0), 24000)

    except RuntimeError as e:
        print("Erreur d’exécution (RuntimeError) :", str(e))
        gr.Warning("Une erreur est survenue, réessayez.")
        return None, None, None, None
    except Exception as e:
        print("Erreur inattendue :", str(e))
        gr.Warning("Une erreur inattendue est survenue.")
        return None, None, None, None

    # Retourne la forme d’onde, le fichier audio, les métriques et l’audio de référence
    return (
        "output.wav",
        "output.wav",
        metrics_text,
        speaker_wav,
    )


def read_txt_file(txt_file):
    """
    Lit le fichier .txt si fourni,
    renvoie son contenu sous forme de str.
    """
    if txt_file is None:
        return ""
    # txt_file est un dictionnaire si type="file",
    # ou une chaîne de caractères (chemin) si type="filepath".
    # Ici on suppose type="filepath" pour un usage local :
    # Lecture en binaire pour détecter l'encodage
    with open(txt_file, 'rb') as f:
        raw_data = f.read()

    # Détection de l'encodage
    detection = chardet.detect(raw_data)
    detected_encoding = detection['encoding']

    # Parfois, chardet peut renvoyer None ou se tromper ;
    # on peut mettre un encodage par défaut (utf-8, cp1252, etc.)
    if detected_encoding is None:
        detected_encoding = 'utf-8'

    # Lecture avec l'encodage détecté
    with open(txt_file, 'r', encoding=detected_encoding, errors='replace') as f:
        text = f.read()

    return text
###################################################
# Partie 3 : Interface Gradio locale
###################################################

title = "Coqui🐸 XTTS (Local)"
description = """
Cette démonstration locale utilise le modèle Coqui TTS XTTS v2 pour la synthèse vocale multilingue et le clonage de voix.  
Veuillez vous assurer que ffmpeg est installé, ainsi que les bibliothèques Python requises (TTS, Gradio, etc.).  
"""

article = """
<p>En utilisant cette démonstration, vous acceptez les termes de la licence Coqui Public Model License (CPML) : <a href="https://coqui.ai/cpml" target="_blank">Coqui.ai</a>.</p>
"""

# Exemples : ajustez selon vos propres fichiers WAV.
examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
]

with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("# " + title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Text Prompt",
                value="Bonjour, ceci est un essai de voix clonée."
            )
            # Fichier texte
            input_txt_file_gr = gr.File(
                label="Fichier texte (optionnel)",
                type="filepath"
            )

            # Dès que le fichier change, on lit son contenu,
            # et on injecte le résultat dans text_box
            input_txt_file_gr.change(fn=read_txt_file, inputs=input_txt_file_gr, outputs=input_text_gr)

            language_gr = gr.Dropdown(
                label="Language",
                choices=supported_languages,
                value="fr"
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                type="filepath",
                value="examples/female.wav",
            )
            mic_gr = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Use Microphone for Reference",
            )
            use_mic_gr = gr.Checkbox(
                label="Use Microphone",
                value=False,
            )
            clean_ref_gr = gr.Checkbox(
                label="Cleanup Reference Voice (FFMPEG)",
                value=False,
            )
            auto_det_lang_gr = gr.Checkbox(
                label="Do not use language auto-detect",
                value=False,
            )
            tos_gr = gr.Checkbox(
                label="Agree to CPML terms",
                value=False,
            )

            tts_button = gr.Button("Lancer la synthèse")

        with gr.Column():
            video_gr = gr.Video(label="Waveform Visual")
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            out_text_gr = gr.Text(label="Metrics")
            ref_audio_gr = gr.Audio(label="Reference Audio Used")

    gr.Examples(
        examples=examples,
        inputs=[
            input_text_gr,
            language_gr,
            ref_gr,
            mic_gr,
            use_mic_gr,
            clean_ref_gr,
            auto_det_lang_gr,
            tos_gr
        ],
        outputs=[video_gr, audio_gr, out_text_gr, ref_audio_gr],
        fn=predict,
        cache_examples=False
    )

    tts_button.click(
        fn=predict,
        inputs=[
            input_text_gr,
            language_gr,
            ref_gr,
            mic_gr,
            use_mic_gr,
            clean_ref_gr,
            auto_det_lang_gr,
            tos_gr
        ],
        outputs=[video_gr, audio_gr, out_text_gr, ref_audio_gr]
    )

    gr.Markdown(article)

# Lancement local
demo.queue()
demo.launch(
    debug=True,
    show_api=True,        # Mettre à False si vous ne voulez pas afficher l’API
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)

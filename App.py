import json
import streamlit as st
import time
import librosa
import io
import tempfile
import numpy as np
import keras

def spectrogram_matrice(file_in,numbers_of_bins=128):
    try:
        y, sr = librosa.load(file_in,sr=22050,duration=10)
        if len(y) == 0:
            return None
        mel = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=numbers_of_bins)
        mel = repeat_matrices(mel)
        log_mel = librosa.power_to_db(mel,ref=np.max)
        return log_mel.astype(np.float32)
    except Exception as e:(
        print(e))
    return None

def repeat_matrices(mel,target=431):
    if mel.shape[1] < target :
        repeats = int(np.ceil(target / mel.shape[1]))
        mel = np.tile(mel,(1,repeats))
    return mel[:, :target]

if "progression" not in st.session_state:
    st.session_state.progression = False
if "resultat" not in st.session_state:
    st.session_state.resultat = False
if "nouveau" not in st.session_state:
    st.session_state.nouveau = False
if "audio" not in st.session_state:
    st.session_state.audio = None
if "oiseau" not in st.session_state:
    st.session_state.oiseau = "Colibri"
if "duration" not in st.session_state:
    st.session_state.duration = 0
if "format" not in st.session_state:
    st.session_state.format = ""
if "init" not in st.session_state:
    st.session_state.init = False

if st.session_state.init:
    path_json = "json_info"
    model_test_after_training = keras.models.load_model("modele.keras")
    st.session_state.init = True

# À la réinitialisation de l'application
if st.session_state.nouveau:
    st.session_state.resultat = False
    st.session_state.progression = False
    st.session_state.nouveau = False
    st.rerun()

# Affichage de l'écran principal
boite = st.empty()
if not st.session_state.progression and not st.session_state.resultat:
    with boite.container():
        audio_record = st.audio_input("Enregistrez un son (seules les 10 premières secondes seront analysées) :", sample_rate=22050)
        audio_upload = st.file_uploader("Ou téléversez un fichier :", type=["mp3", "wav"])
        oui = st.button("Soumettre l'enregistrement audio")

        # Enregistrement envoyé via le bouton
        if oui:
            # Dans le cas où aucun enregistrement n'est fourni
            if audio_record is None and audio_upload is None:
                st.error("Aucun enregistrement n'a été détecté !")

            # Dans le cas d'un fichier téléversé
            elif audio_record is None:
                if audio_upload.type in ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav"]:
                    st.session_state.audio = audio_upload
                    st.session_state.format = audio_upload.type
                    st.session_state.progression = True
                    boite.empty()
                    st.rerun()
                else:
                    st.error("Le format du fichier n'est pas valide")

            # Dans le cas d'un fichier enregistré
            else:
                st.session_state.audio = audio_record
                st.session_state.format = audio_record.type
                st.session_state.progression = True
                boite.empty()
                st.rerun()

time.sleep(0.2)
# Affichage de la barre de chargement
if st.session_state.progression:
    st.write("Veuillez patienter...")
    progress_bar = st.progress(0)

    if st.session_state.format == "audio/mp3":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(st.session_state.audio.getvalue())
            st.session_state.audio = tmp_file.name

    y, sr = librosa.load(st.session_state.audio)
    st.session_state.duration = librosa.get_duration(y=y, sr=sr)

    audio_transform = st.session_state.audio.getvalue()
    audio_buffer = io.BytesIO(audio_transform)

    matrice_verif = spectrogram_matrice(audio_buffer)
    time.sleep(0.8)
    progress_bar.progress(10)
    # Insérer l'intégration du modèle ici éventuellement
    matrice_verif = matrice_verif.reshape((1,*matrice_verif.shape,1))
    prediction = model_test_after_training.predict(matrice_verif,verbose=0)
    pred_class = prediction.argmax()
    with open(f"{path_json}Classes_To_True.json", "r") as f:
        infos = json.load(f)
    print(infos[str(pred_class)])
    st.session_state.oiseau = "Colibri" # À modifier pour le renvoi du modèle
    progress_bar.progress(90)
    st.session_state.progression = False
    st.session_state.resultat = True
    progress_bar.progress(100)
    time.sleep(0.8)
    st.rerun()

if st.session_state.resultat:
    st.write(f"L'espèce d'oiseau détectée est : {st.session_state.oiseau}")
    st.write(f"L'enregistrement avait une durée de {st.session_state.duration} secondes")
    reset = st.button("Détecter un nouvel oiseau")
    if reset:
        st.session_state.nouveau = True
        st.rerun()

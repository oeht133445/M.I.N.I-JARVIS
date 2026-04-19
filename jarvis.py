"""
╔══════════════════════════════════════════╗
║              J.A.R.V.I.S                 ║
║    Assistant vocal personnel (FR)        ║
╚══════════════════════════════════════════╝
"""

import asyncio
import ctypes
import json
import os
import re
import subprocess
import time
import threading
import urllib.parse
import webbrowser
from threading import Event

import edge_tts
import numpy as np
import psutil
import pygame
import pyautogui
import pygetwindow as gw
import scipy.io.wavfile as wav
import screen_brightness_control as sbc
import sounddevice as sd
import whisper
from rapidfuzz import fuzz

# ══════════════════════════════════════════
# ⚙️  CONFIGURATION
# ══════════════════════════════════════════
MIC_ID         = 1           # Index du microphone — à adapter selon le matériel
SAMPLE_RATE    = 16_000      # Hz
TTS_VOICE      = "fr-FR-HenriNeural"
WHISPER_MODEL  = "small"

CHUNK_DURATION = 0.5         # secondes par bloc de détection de bruit
SILENCE_LIMIT  = 20.0        # secondes sans bruit avant mise en veille
SOUND_THRESHOLD = 0.01       # seuil RMS normalisé [0..1]
WAKE_POLL_INTERVAL = 2.0     # secondes entre chaque vérification en mode veille

WAKE_WORDS = ["jarvis", "jarvi", "jervis", "j'arrive", "service"]

APPS: dict[str, str] = {
    "spotify": r"C:\Users\ADMIN\AppData\Roaming\Spotify\Spotify.exe",
    "discord": "discord://",
    "brave":   "brave",
    "steam":   r"C:\Program Files (x86)\Steam\steam.exe",
}

QUICK_SITES: dict[str, str] = {
    "gemini":    "https://gemini.google.com",
    "météo":     "https://meteofrance.com",
    "actualité": "https://news.google.com",
    "twitch":    "https://www.twitch.tv/",
}

STREMIO_PATH  = r"C:\Users\ADMIN\AppData\Local\Programs\Stremio\stremio-shell-ng.exe"
CINEMA_KILL   = ["chrome.exe", "brave.exe", "msedge.exe", "discord.exe", "spotify.exe"]

# ══════════════════════════════════════════
# 🛠️  UTILITAIRES
# ══════════════════════════════════════════
def safe_print(*args) -> None:
    """print() résistant aux erreurs d'encodage et de pipe fermé."""
    try:
        text = " ".join(map(str, args))
        print(text.encode("utf-8", errors="ignore").decode("utf-8"))
    except Exception:
        pass


def normalize_text(text: str) -> str:
    """Minuscules, sans ponctuation superflue ni formules de politesse."""
    text = text.lower()
    text = re.sub(r"[^\w\sàâäéèêëîïôöùûüç'-]", " ", text)
    text = re.sub(r"\b(s'il te plaît|s'il vous plaît|svp|stp|s v p)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_number(text: str) -> int | None:
    """Retourne le premier entier trouvé dans le texte, ou None."""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


# ══════════════════════════════════════════
# 🔊 SYNTHÈSE VOCALE (edge-tts + pygame)
# ══════════════════════════════════════════
_tts_lock = threading.Lock()

def speak(text: str) -> None:
    """Synthétise et joue le texte en voix française."""
    safe_print("Jarvis:", text)

    async def _generate(path: str) -> None:
        await edge_tts.Communicate(text, TTS_VOICE).save(path)

    def _tts_thread(path: str) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_generate(path))
        except Exception as e:
            safe_print(f"[TTS] Erreur génération : {e}")
        finally:
            loop.close()

    mp3_path = "response.mp3"
    with _tts_lock:
        try:
            t = threading.Thread(target=_tts_thread, args=(mp3_path,), daemon=True)
            t.start()
            t.join()

            pygame.mixer.init()
            pygame.mixer.music.load(mp3_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.quit()
        except Exception as e:
            safe_print(f"[TTS] Erreur lecture audio : {e}")
        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass


# ══════════════════════════════════════════
# 🎤 ENREGISTREMENT MICRO
# ══════════════════════════════════════════
def record_audio(duration: float = 4.0) -> str:
    """Enregistre `duration` secondes depuis le micro et sauvegarde en WAV."""
    safe_print(f"\n[MIC] J'écoute ({duration}s)…")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=MIC_ID,
    )
    sd.wait()
    wav.write("audio.wav", SAMPLE_RATE, audio)
    return "audio.wav"


# ══════════════════════════════════════════
# 🧠 WHISPER (chargement unique au démarrage)
# ══════════════════════════════════════════
safe_print(f"[Whisper] Chargement du modèle '{WHISPER_MODEL}'…")
_whisper_model = whisper.load_model(WHISPER_MODEL)


def transcribe(audio_path: str) -> str:
    """Transcrit un fichier audio en texte français."""
    result = _whisper_model.transcribe(audio_path, language="fr", fp16=False)
    return result.get("text", "").lower().strip()


# ══════════════════════════════════════════
# 🔥 MOT DE RÉVEIL
# ══════════════════════════════════════════
def is_wake_word(text: str) -> bool:
    text = text.lower().strip()
    return any(fuzz.partial_ratio(w, text) > 80 for w in WAKE_WORDS)


# ══════════════════════════════════════════
# 🔇 DÉTECTION D'ACTIVITÉ SONORE
# ══════════════════════════════════════════
def poll_for_sound() -> bool:
    """Retourne True si le RMS du micro dépasse SOUND_THRESHOLD."""
    frames = int(CHUNK_DURATION * SAMPLE_RATE)
    try:
        rec = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=MIC_ID)
        sd.wait()
        data = rec.astype(np.float32).flatten() / 32_768.0
        rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
        safe_print(f"[VAD] rms={rms:.4f}, seuil={SOUND_THRESHOLD}")
        return rms > SOUND_THRESHOLD
    except Exception as e:
        safe_print(f"[VAD] Erreur : {e}")
        return False


# ══════════════════════════════════════════
# 📱 GESTION D'APPLICATIONS
# ══════════════════════════════════════════
def find_app(text: str) -> tuple[str | None, str | None]:
    """Recherche floue du nom d'une application dans le dictionnaire APPS."""
    text = normalize_text(text)
    best_name, best_score = None, 0
    for name in APPS:
        score = fuzz.partial_ratio(name, text)
        if score > best_score:
            best_score, best_name = score, name
    if best_score >= 65 and best_name:
        return best_name, APPS[best_name]
    return None, None


def open_app(app_text: str) -> None:
    app_text = normalize_text(app_text)
    if not app_text:
        speak("Quelle application veux-tu ouvrir ?")
        return
    name, cmd = find_app(app_text)
    if cmd:
        os.system(f'start "" "{cmd}"')
        speak(f"J'ouvre {name}")
    else:
        os.system(f'start "" "{app_text}"')
        speak(f"J'essaie d'ouvrir {app_text}")


# ══════════════════════════════════════════
# 🖥️  MULTI-ÉCRANS
# ══════════════════════════════════════════
def move_to_screen(screen_number: int) -> None:
    """Déplace la fenêtre active vers l'écran 1 (gauche) ou 2 (droite)."""
    try:
        time.sleep(0.5)
        window = gw.getActiveWindow()
        if not window:
            speak("Aucune fenêtre active trouvée.")
            return
        screen_w, _ = pyautogui.size()
        x = screen_w if screen_number == 2 else 0
        window.moveTo(x, 0)
        speak(f"Fenêtre déplacée sur l'écran {screen_number}.")
    except Exception as e:
        safe_print(f"[Écran] Erreur : {e}")
        speak("Je n'ai pas pu déplacer la fenêtre.")


# ══════════════════════════════════════════
# 🔊 VOLUME
# ══════════════════════════════════════════
def set_volume(percent: int) -> None:
    percent = max(0, min(100, percent))
    for _ in range(50):
        pyautogui.press("volumedown")
    for _ in range(percent // 2):
        pyautogui.press("volumeup")
    speak(f"Volume à {percent} pourcent.")


# ══════════════════════════════════════════
# 💡 LUMINOSITÉ
# ══════════════════════════════════════════
def set_brightness(percent: int) -> None:
    percent = max(0, min(100, percent))
    try:
        sbc.set_brightness(percent)
        speak(f"Luminosité réglée à {percent} pourcent.")
    except Exception as e:
        safe_print(f"[Luminosité] Erreur : {e}")
        speak("Je n'ai pas pu modifier la luminosité.")


# ══════════════════════════════════════════
# 🎬 MODE CINÉMA
# ══════════════════════════════════════════
def cinema_mode() -> None:
    speak("Préparation de la salle, Monsieur.")

    # Luminosité maximale
    try:
        sbc.set_brightness(100)
    except Exception:
        pass

    # Désactiver l'écran secondaire (MultiMonitorTool optionnel)
    if os.path.exists("MultiMonitorTool.exe"):
        os.system("MultiMonitorTool.exe /disable 2")

    # Fermer les applications parasites
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info.get("name", "").lower() in CINEMA_KILL:
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Lancer Stremio
    if os.path.exists(STREMIO_PATH):
        try:
            os.startfile(STREMIO_PATH)
            time.sleep(5)
            pyautogui.press("f11")
            speak("Stremio est prêt. Bon film, Monsieur.")
        except Exception:
            speak("Erreur lors du lancement de Stremio.")
    else:
        speak("Stremio introuvable, mais le reste est prêt.")


# ══════════════════════════════════════════
# ⚡ MOTEUR DE COMMANDES
# ══════════════════════════════════════════
def execute(command: str) -> None:
    """Interprète et exécute une commande vocale."""
    command = command.lower()
    safe_print("[CMD]", command)

    # — Volume ————————————————————————————
    if "volume" in command:
        if "maximum" in command:
            set_volume(100)
        else:
            n = extract_number(command)
            if n is not None:
                set_volume(n)
            else:
                speak("À quel niveau dois-je régler le volume ?")
        return

    # — Luminosité ————————————————————————
    if "luminosité" in command:
        n = extract_number(command)
        if n is not None:
            set_brightness(n)
        else:
            speak("À quel pourcentage dois-je régler la luminosité ?")
        return

    # — Multi-écrans ——————————————————————
    screen_keywords = {
        2: ["deuxième écran", "écran 2", "écran droit"],
        1: ["premier écran",  "écran 1", "écran gauche"],
    }
    for screen_num, keywords in screen_keywords.items():
        if any(kw in command for kw in keywords):
            move_to_screen(screen_num)
            return

    # — Ouvrir application ————————————————
    if "ouvre" in command:
        open_app(command.replace("ouvre", "").strip())
        return

    # — Dossier Téléchargements ———————————
    if "téléchargements" in command or "téléchargements" in command:
        path = os.path.join(os.path.expanduser("~"), "Downloads")
        try:
            os.startfile(path)
            speak("Voici vos téléchargements, Monsieur.")
        except Exception as e:
            safe_print(f"[Downloads] Erreur : {e}")
            speak("Je n'ai pas pu ouvrir le dossier téléchargements.")
        return

    # — Sites rapides —————————————————————
    for keyword, url in QUICK_SITES.items():
        if fuzz.partial_ratio(keyword, command) > 80:
            webbrowser.open(url)
            speak(f"Ouverture de {keyword}.")
            return

    # — Recherche web (Google) ————————————
    if "recherche" in command or "cherche" in command:
        query = re.sub(r"\b(recherche pour moi|recherche|cherche)\b", "", command).strip()
        if query:
            webbrowser.open(f"https://google.com/search?q={urllib.parse.quote(query)}")
            speak(f"Voici les résultats pour {query}.")
        else:
            speak("Que dois-je chercher ?")
        return

    # — YouTube ———————————————————————————
    if "youtube" in command:
        query = re.sub(r"\b(cherche sur|youtube)\b", "", command).strip()
        if query:
            webbrowser.open(f"https://youtube.com/results?search_query={urllib.parse.quote(query)}")
            speak("Voici ce que j'ai trouvé sur YouTube.")
        else:
            speak("Que veux-tu regarder sur YouTube ?")
        return

    # — Perplexity ————————————————————————
    if "perplexity" in command:
        query = command.replace("perplexity", "").strip()
        if query:
            webbrowser.open(f"https://www.perplexity.ai/search?q={urllib.parse.quote(query)}")
            speak(f"Je lance la recherche sur Perplexity pour : {query}.")
        else:
            speak("Monsieur, quelle est votre question pour Perplexity ?")
        return

    # — Médias ————————————————————————————
    if "suivant" in command:
        pyautogui.press("nexttrack")
        speak("Titre suivant.")
        return
    if "pause" in command or "reprend" in command:
        pyautogui.press("playpause")
        speak("Fait.")
        return

    # — Performance système ———————————————
    perf_triggers = ["est-ce que tu rames", "charge de travail", "performance", "état du système"]
    if any(t in command for t in perf_triggers):
        cpu  = psutil.cpu_percent(interval=0.5)
        ram  = psutil.virtual_memory().percent
        if cpu > 80:
            msg = f"Oui Monsieur, le processeur est à {cpu} pourcent — ça commence à chauffer !"
        else:
            msg = f"Tout va bien, Monsieur. CPU à {cpu} pourcent, mémoire à {ram} pourcent."
        speak(msg)
        return

    # — Vider la corbeille ————————————————
    if "corbeille" in command:
        try:
            ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 1 + 2 + 4)
            speak("Corbeille vidée, Monsieur.")
        except Exception:
            speak("Je n'ai pas pu vider la corbeille — elle est peut-être déjà vide.")
        return

    # — Mode cinéma ———————————————————————
    if "mode cinéma" in command:
        cinema_mode()
        return

    # — Fin du mode cinéma ————————————————
    if "rallume l'écran" in command or "fin du film" in command:
        speak("Bien Monsieur, je réactive l'écran secondaire.")
        if os.path.exists("MultiMonitorTool.exe"):
            os.system("MultiMonitorTool.exe /enable 2")
        else:
            speak("MultiMonitorTool est introuvable.")
        return

    # — Fermer fenêtre ————————————————————
    if "ferme" in command:
        pyautogui.hotkey("alt", "f4")
        speak("Je ferme la fenêtre.")
        return

    # — Arrêt ——————————————————————————
    if any(kw in command for kw in ("stop", "arrête", "éteins")):
        speak("À plus tard, Monsieur.")
        raise SystemExit(0)

    # — Fallback ——————————————————————————
    speak("Je n'ai pas bien compris la commande.")


def describe_capabilities() -> None:
    speak(
        "Je peux : ouvrir des applications comme Spotify, Discord ou Brave ; "
        "déplacer une fenêtre sur le deuxième écran ; rechercher sur Google, YouTube ou Perplexity ; "
        "régler le volume et la luminosité ; fermer des fenêtres ; activer le mode cinéma ; "
        "et me mettre en veille si aucun bruit n'est détecté."
    )


# ══════════════════════════════════════════
# 🚀 BOUCLE PRINCIPALE
# ══════════════════════════════════════════
stop_event = Event()


def enter_sleep_mode() -> None:
    speak("Je me mets en veille. Appelle-moi en parlant pour me réveiller.")
    while not stop_event.is_set():
        time.sleep(WAKE_POLL_INTERVAL)
        if poll_for_sound():
            speak("Je suis réveillé. Que puis-je faire ?")
            return


def listen_loop() -> None:
    silent_time = 0.0

    while not stop_event.is_set():
        try:
            if not poll_for_sound():
                silent_time += CHUNK_DURATION
                if silent_time >= SILENCE_LIMIT:
                    enter_sleep_mode()
                    silent_time = 0.0
                continue

            # Bruit détecté : transcription
            silent_time = 0.0
            text = transcribe(record_audio(duration=4.0))
            if text:
                safe_print("[STT] Entendu :", text)

            if not is_wake_word(text):
                continue

            # Mot de réveil détecté : on écoute la commande
            speak("Oui ?")
            command = transcribe(record_audio(duration=5.0))
            if not command:
                speak("Je n'ai rien compris, annulation.")
                continue

            safe_print("[CMD] Reçu :", command)
            if re.search(r"que\s+peux[- ]?tu\s+faire", command):
                describe_capabilities()
            else:
                execute(command)

        except SystemExit:
            stop_event.set()
            break
        except Exception as e:
            safe_print(f"[Loop] Erreur : {e}")


# ══════════════════════════════════════════
# ▶  POINT D'ENTRÉE
# ══════════════════════════════════════════
if __name__ == "__main__":
    speak("Jarvis est prêt et à votre écoute.")
    try:
        listen_loop()
    except KeyboardInterrupt:
        stop_event.set()
        speak("Arrêt de Jarvis.")

import streamlit as st
import cv2
import time
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import deque
import threading

# --- Seitenkonfiguration ---
st.set_page_config(page_title="Phone Guard Timer", layout="wide")

# --- Titel & Beschreibung ---
st.title("📱 Phone Guard Timer")
st.markdown("**Konzentriert arbeiten:** Der Timer läuft nur, wenn **KEIN Handy** erkannt wird.")
st.divider()

# --- 1. Modell laden (mit Caching) ---
@st.cache_resource
def load_model():
    """Lädt das YOLOv8 Modell (nano für Geschwindigkeit)"""
    # 'yolov8n.pt' wird automatisch heruntergeladen, falls nicht vorhanden
    return YOLO('yolov8n.pt')

model = load_model()

# --- 2. Pomodoro Logik ---
class PomodoroTimer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.work_duration = 25 * 60  # 25 Minuten in Sekunden
        self.break_duration = 5 * 60   # 5 Minuten Pause
        self.current_time = self.work_duration
        self.is_work_phase = True
        self.is_running = False
        self.start_time = 0
        self.phone_detected_recently = False

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            
    def pause(self):
        if self.is_running:
            # Speichere die verstrichene Zeit
            elapsed = time.time() - self.start_time
            self.current_time -= int(elapsed)
            self.is_running = False
            
    def update(self):
        if not self.is_running:
            return self.current_time
            
        elapsed = int(time.time() - self.start_time)
        remaining = self.current_time - elapsed
        
        if remaining <= 0:
            # Phase ist vorbei
            self.is_running = False
            # Automatischer Wechsel? Hier pausieren wir erstmal
            return 0
        return remaining
        
    def switch_phase(self):
        """Wechselt zwischen Arbeit und Pause"""
        if self.is_work_phase:
            # Wechsel zu Pause
            self.is_work_phase = False
            self.current_time = self.break_duration
        else:
            # Wechsel zu Arbeit
            self.is_work_phase = True
            self.current_time = self.work_duration
        self.is_running = False # Timer pausieren bis zum manuellen Start
        self.phone_detected_recently = False

# Session State initialisieren
if 'timer' not in st.session_state:
    st.session_state.timer = PomodoroTimer()
if 'last_phone_status' not in st.session_state:
    st.session_state.last_phone_status = False
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None

# --- 3. UI Layout ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("⏱️ Timer")
    timer_placeholder = st.empty()
    
    # Timer Steuerung
    col_start, col_stop, col_reset = st.columns(3)
    with col_start:
        start_btn = st.button("▶️ Start", use_container_width=True)
    with col_stop:
        stop_btn = st.button("⏸️ Pause", use_container_width=True)
    with col_reset:
        reset_btn = st.button("🔁 Reset", use_container_width=True)
        
    if start_btn:
        st.session_state.timer.start()
    if stop_btn:
        st.session_state.timer.pause()
    if reset_btn:
        st.session_state.timer.reset()
        st.rerun()
        
    st.subheader("📊 Status")
    status_text = st.empty()
    phone_status_text = st.empty()

with col2:
    st.subheader("📸 Live-Erkennung")
    # Hier kommt das Kamerabild hin
    camera_placeholder = st.empty()

# --- 4. Hilfsfunktion für Handy-Erkennung ---
def detect_phone(frame):
    """Führt die YOLO-Erkennung durch und gibt zurück: (frame_annotiert, phone_detected)"""
    # YOLO Inference
    results = model(frame, verbose=False, classes=[67]) # Klasse 67 = cell phone
    phone_detected = False
    
    # Ergebnisse verarbeiten
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 67:  # cell phone
                    phone_detected = True
                    # Bounding Box zeichnen
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "HANYDETECTED", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Wenn kein Handy, grünen Rahmen zeichnen
    if not phone_detected:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 5)
        
    return frame, phone_detected

# --- 5. Hauptloop für Webcam & Timer ---
def main_loop():
    # Kamera initialisieren
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Keine Kamera gefunden! Bitte Webcam anschließen.")
        return
    
    # Für bessere Performance: Auflösung reduzieren
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        # Frame lesen
        ret, frame = cap.read()
        if not ret:
            st.warning("Kamera-Fehler")
            break
            
        # 1. Handy-Erkennung
        annotated_frame, phone_detected = detect_phone(frame)
        
        # 2. Timer-Logik basierend auf Handy-Status
        timer = st.session_state.timer
        
        if phone_detected:
            # Handy erkannt -> Timer pausieren
            if timer.is_running:
                timer.pause()
                st.session_state.last_phone_status = True
        else:
            # Kein Handy -> Timer laufen lassen, wenn er gestartet wurde
            if timer.is_running == False and st.session_state.last_phone_status == True:
                # Timer war vorher wegen Handy pausiert -> Jetzt weiter
                # Setze Startzeit neu, basierend auf aktueller Restzeit
                timer.start()
                st.session_state.last_phone_status = False
            elif timer.is_running == False and st.session_state.last_phone_status == False:
                # Timer ist manuell pausiert -> nichts tun
                pass
        
        # Timer aktualisieren
        remaining = timer.update()
        
        # 3. UI Updates
        # Timer anzeigen
        minutes = remaining // 60
        seconds = remaining % 60
        phase = "🍅 Arbeiten" if timer.is_work_phase else "☕ Pause"
        
        timer_placeholder.markdown(f"""
        <div style='text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='margin:0'>{phase}</h2>
            <h1 style='font-size: 4rem; margin:0; font-family: monospace;'>{minutes:02d}:{seconds:02d}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Status anzeigen
        if phone_detected:
            phone_status_text.error("⚠️ HANDY ERKANNT! Timer pausiert.")
            status_text.warning("Status: ⏸️ Pausiert (Handy im Bild)")
        else:
            phone_status_text.success("✅ Kein Handy erkannt.")
            if timer.is_running:
                status_text.success("Status: ▶️ Läuft")
            else:
                if timer.current_time == timer.work_duration or timer.current_time == timer.break_duration:
                    status_text.info("Status: ⏹️ Gestoppt")
                else:
                    status_text.warning("Status: ⏸️ Pausiert (manuell)")
        
        # Frame in Streamlit anzeigen (BGR zu RGB konvertieren)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Kleine Pause für Stabilität
        time.sleep(0.03)
        
        # Prüfen ob Timer abgelaufen ist
        if remaining <= 0 and timer.is_running == False:
            st.balloons()
            st.warning(f"🎉 {phase} beendet!")
            # Automatisch umschalten? Der Benutzer muss bestätigen.
            if st.button("➡️ Nächste Phase starten"):
                timer.switch_phase()
                st.rerun()
            break
            
    cap.release()

# App ausführen
if __name__ == "__main__":
    main_loop()

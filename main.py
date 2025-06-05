import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models.resnet_emotion import EmotionResNet
from config import load_config
from game_agent import play_game_ui, check_winner, is_full, get_best_move
import threading
import streamlit as st
import tempfile
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_game_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

cfg = load_config()
device = torch.device("cpu")
model = EmotionResNet(num_classes=cfg['training']['num_classes'],
                      pretrained=cfg['training']['pretrained']).to(device)
checkpoint = torch.load(cfg['test']['ckpt'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((cfg['training']['image_size'], cfg['training']['image_size'])),
    transforms.Grayscale(num_output_channels=3) if cfg['training']['grayscale'] else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class_names = cfg['dataset']['class_names']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_state = {'emotion': 'neutral', 'last_update': time.time()}

st.session_state.setdefault('emotion_state', emotion_state)

def detect_emotion_stream():
    cap = cv2.VideoCapture(0)
    camera_placeholder = st.empty()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    
    while not st.session_state.get("stop_emotion", False):
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.5)  
                continue
            
            current_time = time.time()
            if current_time - emotion_state['last_update'] >= 0.5:
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])  # width * height
                    (x, y, w, h) = largest_face
                    
                    face_img = frame[y:y+h, x:x+w]
                    image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    image = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(image)
                        probs = F.softmax(output, dim=1)
                        _, pred = torch.max(probs, dim=1)
                        pred_class = class_names[pred.item()]
                        confidence = probs[0][pred.item()].item()
                        
                        if emotion_state['emotion'] != pred_class:
                            st.session_state['emotion_state'] = {
                                'emotion': pred_class.lower(),
                                'last_update': current_time
                            }
                            logger.info(f"updated to {st.session_state['emotion_state']}")
                            emotion_state['emotion'] = pred_class
                            emotion_state['last_update'] = current_time
                            
                            max_depth = 5 if pred_class in ['Happy', 'Neutral'] else 2
                            logger.info(f"Player emotion changed to {pred_class} (confidence: {confidence:.2f}) -- AI agent max depth: {max_depth}")
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f'Emotion: {pred_class} ({confidence:.2f})', 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No face detected', (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            with camera_placeholder.container():
                st.image(frame, channels="BGR", 
                        caption=f"Current Emotion: {emotion_state['emotion']} | Updated: {time.strftime('%H:%M:%S', time.localtime(emotion_state['last_update']))}", 
                        use_column_width=True)
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            time.sleep(1) 
    
    cap.release()
    camera_placeholder.empty()


def main():
    st.set_page_config(page_title="Emotion-Aware Tic-Tac-Toe")
    st.title(" Emotion-Aware Tic-Tac-Toe Game")

    if 'stop_emotion' not in st.session_state:
        st.session_state['stop_emotion'] = False
    if 'game_started' not in st.session_state:
        st.session_state['game_started'] = False
    if 'show_game' not in st.session_state:
        st.session_state['show_game'] = False
    if 'emotion_thread' not in st.session_state:
        st.session_state['emotion_thread'] = None

    if not st.session_state['game_started']:
        st.info(" This game adapts AI difficulty based on your emotions!")
        st.markdown("""
        - **Happy/Neutral**: AI plays at maximum difficulty
        - **Other emotions**: AI plays easier to keep the game fun
        """)
        
        if st.button(" Start Game", type="primary", use_container_width=True):
            st.session_state['game_started'] = True
            st.session_state['show_game'] = True
            st.session_state['stop_emotion'] = False
            logger.info("Game started - Emotion detection initiated")
            st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(" Emotion Detection")
        
        with col2:
            if st.button(" Stop Game", type="secondary"):
                st.session_state['stop_emotion'] = True
                st.session_state['game_started'] = False
                st.session_state['show_game'] = False
                
                # Wait for thread to finish
                if st.session_state.get('emotion_thread') and st.session_state['emotion_thread'].is_alive():
                    st.session_state['emotion_thread'].join(timeout=3)
                
                logger.info("Game stopped by user")
                st.rerun()
        
        if not st.session_state.get('emotion_thread') or not st.session_state['emotion_thread'].is_alive():
            st.session_state['emotion_thread'] = threading.Thread(target=detect_emotion_stream, daemon=True)
            st.session_state['emotion_thread'].start()
        
        
        if st.session_state['show_game']:
            st.markdown("---")
            play_game_ui()


if __name__ == '__main__':
    main()
import pygame
import pygame.camera
import numpy as np
from pygame.locals import *
import cv2
from insightface.app import FaceAnalysis
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
import torch
from threading import Thread
from queue import Queue
from enum import Enum
import time
import random

LOADING = "Loading..."
FALLBACK_PREDICTIONS = [
    "Сегодня ваш день! Вселенная приготовила для вас приятный сюрприз. Будьте открыты новому!",
    "Ваша внутренняя сила поразительна. Сегодня она приведет вас к важному insight — просто прислушайтесь к себе.",
    "Сегодняшний день — чистый лист. Нарисуйте на нем самое смелое свое желание, и оно начнет сбываться.",
    "Вас ждет встреча, которая перевернет ваше представление о возможном. Не упустите шанс завязать новый разговор.",
    "Ваша энергия сегодня притягивает успех. Смело беритесь за самые сложные задачи — вы справитесь.",
    "Вселенная шепчет: сегодня день, чтобы сделать тот самый шаг, который вы долго откладывали. Доверьтесь интуиции!",
    "Готовьтесь к неожиданной удаче! Она придет оттуда, откуда вы совсем не ждете. Главное — распознать ее.",
    "Сегодня звезды советуют вам действовать, а не размышлять. Самый верный путь откроется именно в движении.",
    "Ваше обаяние сегодня на максимуме. Используйте его, чтобы найти общий язык с кем-то очень важным для вашего будущего.",
    "Сегодня вы сможете найти ответ на вопрос, который давно вас мучает. Он придет во время отдыха — позвольте себе расслабиться."
]

prompt_queue = Queue()
response_queue = Queue()

class Status(Enum):
    camera = 0
    capturing = 1
    loading = 2
    ready = 3
    error = 4

def init_insightface():
    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 для CPU, укажите ctx_id=0 для GPU если он есть
    return app

def init_pygame():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    return cap

def clear_cv2(cap):
    cap.release()
    cv2.destroyAllWindows()

def detect_face(app, frame):
    faces = app.get(frame)

    if len(faces) > 0:
        face = faces[0]
        bbox = face.bbox.astype(int)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if (bbox[2] - bbox[0]) > 200:
            print("face detected")
            return face

    return None

def face_prompt(face):
    age = int(face.age)
    gender = "Мужчина" if face.gender == 1 else "Женщина"
    emotion = "Радость" if face.emotion == 0 else "Нейтрально"
    user_prompt = f"""
    Ты — духовный предсказатель на выставке. Напиши очень краткое (не более 2-3 предложений) вдохновляющее предсказание на день для этого человека.
    Пол: {gender}.
    Возраст: примерно {age} лет.
    Настроение: {emotion}.

    Предсказание:
    """
    return user_prompt
def make_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class StopOnTokens(StoppingCriteria):
        def __init__(self, stop_tokens):
            super().__init__()
            self.stop_tokens = stop_tokens
    
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop in self.stop_tokens:
                if stop in tokenizer.decode(input_ids[0][-1].item()):
                    return True

            return False
    model_name = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"  # or other LLaMA variants
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Generate text
    stop_tokens = ["\n", "###"]

    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

    while True:
        prompt = prompt_queue.get()
        # Generate with stop criteria
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model.generate(
            **inputs,
            max_length=200,
            stopping_criteria=stopping_criteria,
            do_sample=True,
            temperature=0.7
        )
        
        response_queue.put(trim_both_ends(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]))

def trim_until_first_letter(s):
    for i, char in enumerate(s):
        if char.isalpha():
            return s[i:]
    return ""

def trim_tail_until_last_letter(s):
    for i, char in enumerate(s[::-1]):
        if char.isalpha():
            return s[:-i]
    return ""

def trim_both_ends(s):
    return trim_tail_until_last_letter(trim_until_first_letter(s))

def display_text(text, img):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2

    text_width, text_height = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

    CenterCoordinates = (int(img.shape[1] / 2) - int(text_width / 2), int(img.shape[0] / 2) + int(text_height / 2))

    cv2.putText(img, text, CenterCoordinates, fontFace, fontScale, fontColor, thickness)
    return img

def get_frame():


def main():
    cap = init_pygame()
    if not cap:
        return 1
    app = init_insightface()
    ai_thread = Thread(target=make_prediction)
    ai_thread.start()
    status = Status.camera
    saved_frame = None
    response = None
    start_time = None
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        if status == Status.camera:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read camera")
                break

            frame = cv2.flip(frame, 1)
            saved_frame = frame
            face = detect_face(app, saved_frame)
            
            if face:
                start_time = time.time()
                print("Capturing")
                status = Status.capturing
        elif status == Status.capturing:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read camera")
                break

            frame = cv2.flip(frame, 1)
            saved_frame = frame
            face = detect_face(app, saved_frame)
            if not face:  
            if face:
                saved_frame = frame
                prompt = face_prompt(face)
                prompt_queue.put(prompt)
                start_time = time.time()
                print("Loading")
                status = Status.capturing           
        elif status == Status.loading:
            frame = display_text(LOADING, saved_frame)
            if not response_queue.empty():
                response = response_queue.get_nowait()
                start_time = time.time()
                print("Ready")
                status = Status.ready
            elif time.time() - start_time > 30:
                start_time = time.time()
                print("Skip ready")
                status = Status.error
        elif status == Status.ready:
            frame = display_text(response, saved_frame)
            if time.time() - start_time > 10:
                print("Camera")
                status = Status.camera
        elif status == Status.error:
            text = random.choice(FALLBACK_PREDICTIONS)
            frame = display_text(text, saved_frame)
            if time.time() - start_time > 10:
                print("Camera")
                status = Status.camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('Face Detection', frame)
    ai_thread.join()
    clear_cv2(cap)
main() 
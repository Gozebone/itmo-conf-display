import pygame
import pygame.camera
import numpy as np
from pygame.locals import *
from insightface.app import FaceAnalysis
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
import torch
from threading import Thread, Event
from queue import Queue
from enum import Enum
import time
import random

WIDTH, HEIGHT = 1920, 1080
LOADING = "Loading..."
FALLBACK_PREDICTIONS = [
    "Сегодня ваш день! Вселенная приготовила для вас приятный сюрприз. Будьте открыты новому!",
    "Ваша внутренняя сила поразительна. Сегодня она приведет вас к важному инсайт — просто прислушайтесь к себе.",
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
    app.prepare(ctx_id=0, det_size=(320, 320)) # ctx_id=0 для CPU, укажите ctx_id=0 для GPU если он есть
    return app

def init_pygame():
    pygame.init()

    font = pygame.font.SysFont("Arial", 36)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Webcam with Cyrillic Text - InsightFace Ready")
    return screen, font

def init_pygame_camera():
    pygame.camera.init()
    cameras = pygame.camera.list_cameras()
    if not cameras:
        print("No cameras found!")
        pygame.quit()
        exit()

    cam = pygame.camera.Camera(cameras[0], (640, 480))
    cam.start()
    return cam

def clear_pygame(cap):
    cap.stop()
    pygame.quit()

def pygame_surface_to_numpy_bgr(surface):
    # Convert PyGame surface to a 3D numpy array (RGB)
    rgb_array = pygame.surfarray.array3d(surface)
    # Transpose to get proper dimensions (height, width, channels)
    rgb_array = np.transpose(rgb_array, (1, 0, 2))
    # Convert RGB to BGR (which is what InsightFace expects)
    bgr_array = rgb_array[:, :, [2, 1, 0]]
    return bgr_array

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        test_width, _ = font.size(test_line)
        
        if test_width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def detect_face(app, frame):
    faces = app.get(frame)

    if len(faces) > 0:
        face = faces[0]
        bbox = face.bbox.astype(int)

        # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if (bbox[2] - bbox[0]) > 200:
            return face

    return None

def face_prompt(face):
    age = int(face.age)
    gender = "Мужчина" if face.gender == 1 else "Женщина"
    print(f"Detected Emotion: {face.emotion}")
    emotion = "Радость" if face.emotion == 0 else "Нейтрально"
    user_prompt = f"""
    Ты — духовный предсказатель на выставке. Напиши очень краткое (не более 2-3 предложений) вдохновляющее предсказание на день для этого человека.
    Пол: {gender}.
    Возраст: примерно {age} лет.
    Настроение: {emotion}.

    Предсказание:
    """
    return user_prompt

def make_prediction(stop_event):
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

    while not stop_event.is_set():
        prompt = prompt_queue.get()
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

def get_frame(screen, cam):
    if cam.query_image():
        img = cam.get_image()
        
        # img_width, img_height = img.get_size()
        # crop_width = img_height * 9 // 16  # 9:16 aspect ratio
        # crop_x = (img_width - crop_width) // 2
        
        # cropped_img = img.subsurface(pygame.Rect(crop_x, 0, crop_width, img_height))
        scaled_img = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
        
        screen.blit(scaled_img, (0, 0))
        return scaled_img

    return None

def display_text(text, screen, font, overlay=True):
    wrapped_lines = wrap_text(text, font, HEIGHT - 40) 
    lines_height = len(wrapped_lines) * 40
    first_line_location = (WIDTH - lines_height) // 2

    if overlay:
        overlay = pygame.Surface((lines_height + 40, HEIGHT), pygame.SRCALPHA, 32)
        overlay.fill((0, 0, 0, 160))  # Black with alpha
        screen.blit(overlay, (first_line_location - 20, 0))
    for i, line in enumerate(wrapped_lines):
        line_surface = font.render(line, True, (255, 255, 255))
        rotated_line_surface = pygame.transform.rotate(line_surface, 90)
        line_width, _ = font.size(line)
        center_position = (HEIGHT - line_width) // 2
        screen.blit(rotated_line_surface, (first_line_location + i * 40, center_position))
        
def main():
    screen, font = init_pygame()
    cap = init_pygame_camera()
    if not cap:
        return 1
    app = init_insightface()
    stop_event = Event()
    # ai_thread = Thread(target=make_prediction, args=(stop_event,))
    # ai_thread.start()
    status = Status.camera
    response = None
    start_time = None
    running = True
    random_fallback_text = None
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        if status == Status.camera:
            frame = get_frame(screen, cap)
            numpy_frame = pygame_surface_to_numpy_bgr(frame)
            face = detect_face(app, numpy_frame)
        
            if face:
                bbox = face.bbox
                print(bbox)
                pygame.draw.rect(screen, color=(255,255,255), rect=(0,0,100,100), width=5, border_radius=1)
                # pygame.draw.rect(screen, color=(255,255,255), rect=(bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1])), width=5, border_radius=1)
                # print(face.keys())
                # prompt = face_prompt(face)
                # print(prompt)
                # prompt_queue.put(prompt)
                start_time = time.time()
                print("Loading")
                status = Status.loading
        elif status == Status.loading:
            frame = get_frame(screen, cap)
            display_text(LOADING, screen, font, overlay=False)
            numpy_frame = pygame_surface_to_numpy_bgr(frame)
            face = detect_face(app, numpy_frame)
        
            if face:
                bbox = face.bbox
                rect = tuple(map(int, (bbox[0], bbox[1], abs(bbox[2]-bbox[0]), abs(bbox[3]-bbox[1]))))
                
                pygame.draw.rect(screen, color=(255,255,255), rect=rect, width=5, border_radius=1)
            if not response_queue.empty():
                response = response_queue.get_nowait()
                start_time = time.time()
                print("Ready")
                status = Status.ready
            elif time.time() - start_time > 5:
                start_time = time.time()
                random_fallback_text = random.choice(FALLBACK_PREDICTIONS)
                print("Skip ready")
                status = Status.error
        elif status == Status.ready:
            frame = get_frame(screen, cap)
            display_text(response, screen, font, overlay=True)
            if time.time() - start_time > 10:
                print("Camera")
                status = Status.camera
        elif status == Status.error:
            frame = get_frame(screen, cap)
            display_text(random_fallback_text, screen, font, overlay=True)
            if time.time() - start_time > 10:
                print("Camera")
                status = Status.camera
        pygame.display.flip()
        clock.tick(24)
    stop_event.set()
    # ai_thread.join()
    clear_pygame(cap)
main() 
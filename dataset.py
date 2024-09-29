import pandas as pd
import torch
import cv2
import librosa
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer, XCLIPModel, BertTokenizer, BertModel, ASTForAudioClassification
from PIL import Image
from torchvision import transforms
from moviepy.editor import VideoFileClip
import numpy as np
from transformers import ASTFeatureExtractor
import concurrent.futures
import os


class MultimodalVideoDataset(Dataset):
    def __init__(self, csv_file, video_folder, label_list,
                 bert_model_name="bert-base-uncased",
                 xclip_model_name="microsoft/xclip-base-patch16-zero-shot",
                 ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 num_frames=8, max_workers=4, device='cpu'):
        if not csv_file is None:
            self.data = pd.read_csv(csv_file)
        self.video_folder = video_folder
        self.label_list = label_list
        self.num_frames = num_frames
        self.max_workers = max_workers

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)

        self.xclip_feature_extractor = AutoFeatureExtractor.from_pretrained(xclip_model_name)
        self.xclip_tokenizer = AutoTokenizer.from_pretrained(xclip_model_name)
        self.xclip_model = XCLIPModel.from_pretrained(xclip_model_name).to(device)

        self.ast_feature_extractor = ASTFeatureExtractor.from_pretrained(ast_model_name)
        self.ast_model = ASTForAudioClassification.from_pretrained(ast_model_name).to(device)

        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.device = device

    def extract_audio_from_video(self, video_path, audio_save_path):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_save_path, codec='pcm_s16le')
        audio_clip.close()
        video_clip.close()

    def extract_video_frames(self, video_path, num_frames=8):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // num_frames, 1)

        for frame_idx in range(0, total_frames, frame_interval):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.video_transform(frame_pil)
            frames.append(frame_tensor)

            if len(frames) >= num_frames:
                break

        video_capture.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames).to(self.device)

    def process_text(self, text_data):
        bert_text_tokens = self.bert_tokenizer(
            text_data, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)
        bert_text_embedding = self.bert_model(**bert_text_tokens).pooler_output.squeeze()
        return bert_text_embedding

    def process_audio(self, audio_path):
        output_dim = self.ast_model.config.num_labels
        if not os.path.exists(audio_path):
            return torch.zeros((1, output_dim)).to(self.device)

        waveform, sample_rate = librosa.load(audio_path, sr=None)

        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        max_length = 32000
        if len(waveform) > max_length:
            waveform = waveform[:max_length]

        if len(waveform) < 400:
            waveform = np.pad(waveform, (0, 400 - len(waveform)), mode='constant')

        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        audio_input = self.ast_feature_extractor(
            waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            audio_embedding = self.ast_model(**audio_input).logits.squeeze()

        return audio_embedding

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row["video_id"]
        title = str(row["title"])
        description = str(row["description"])
        tags = [tag.strip().lower() for tag in str(row["tags"]).split(",")]
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
        audio_path = os.path.join(self.video_folder, f"{video_id}.wav")

        text_data = title + " " + description

        # Путь к сохраненному комбинированному эмбеддингу
        embedding_path = os.path.join(self.video_folder, f"{video_id}_embedding_video_text.pt")

        # Проверяем, существует ли уже комбинированный эмбеддинг
        if os.path.exists(embedding_path):
            # Загружаем комбинированный эмбеддинг
            combined_embedding = torch.load(embedding_path).to(self.device)
        else:
            # Вычисляем комбинированный эмбеддинг
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_video = executor.submit(self.extract_video_frames, video_path, self.num_frames)
                future_text = executor.submit(self.process_text, text_data)
                # future_audio = executor.submit(self.process_audio, audio_path)

                video_frames_tensor = future_video.result().unsqueeze(0).to(self.device)
                text_embedding = future_text.result().to(self.device)
                # audio_embedding = future_audio.result().to(self.device)

                xclip_text_inputs = self.xclip_tokenizer(
                    text_data, return_tensors='pt', padding=True, truncation=True
                ).to(self.device)

                with torch.no_grad():
                    xclip_output = self.xclip_model(
                        pixel_values=video_frames_tensor,
                        input_ids=xclip_text_inputs['input_ids'],
                        attention_mask=xclip_text_inputs['attention_mask']
                    )
                    video_text_embedding = torch.cat(
                        (
                            xclip_output.text_embeds.squeeze(0).squeeze(0),
                            xclip_output.video_embeds.squeeze(0)
                        ),
                        dim=-1
                    )

            # combined_embedding = torch.cat(
            #     (video_text_embedding, audio_embedding.squeeze(0), text_embedding), dim=-1
            # )
            combined_embedding = torch.cat(
                (video_text_embedding, text_embedding), dim=-1
            )

            # Сохраняем комбинированный эмбеддинг
            torch.save(combined_embedding.cpu(), embedding_path)

        labels = torch.zeros(len(self.label_list)).to(self.device)
        lower_label_list = [label.lower() for label in self.label_list]
        for tag in tags:
            if tag in lower_label_list:
                labels[lower_label_list.index(tag)] = 1
        return {
            "combined_embedding": combined_embedding,
            "labels": labels
        }

    def __len__(self):
        return len(self.data)
    
# ======== Специальный класс SingleVideoDataset ======== #
class SingleVideoDataset(MultimodalVideoDataset):
    def __init__(self, video_path, title, description, label_list,
                 bert_model_name="bert-base-uncased",
                 xclip_model_name="microsoft/xclip-base-patch16-zero-shot",
                 ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 num_frames=8, device='cpu'):
        # Конструктор инициализирует только необходимые компоненты
        super().__init__(None, None, label_list,
                         bert_model_name, xclip_model_name, ast_model_name,
                         num_frames=num_frames, device=device)
        self.video_path = video_path
        self.title = title
        self.description = description

    def __getitem__(self, idx):
        """Обработка одного видео и текстовых данных."""
        text_data = self.title + " " + self.description
        video_tensor = self.extract_video_frames(self.video_path, self.num_frames).unsqueeze(0).to(self.device)
        text_embedding = self.process_text(text_data).to(self.device)
        #audio_tensor = self.extract_audio_from_video(self.video_path, "temp_audio.wav").to(self.device)

        xclip_text_inputs = self.xclip_tokenizer(text_data, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            xclip_output = self.xclip_model(
                pixel_values=video_tensor,
                input_ids=xclip_text_inputs['input_ids'],
                attention_mask=xclip_text_inputs['attention_mask']
            )
        video_text_embedding = torch.cat((xclip_output.text_embeds.squeeze(0).squeeze(0), xclip_output.video_embeds.squeeze(0)), dim=-1)

        # Создание комбинированного эмбеддинга
        #combined_embedding = torch.cat((video_text_embedding, audio_tensor.squeeze(0), text_embedding), dim=-1).unsqueeze(0).to(self.device)
        combined_embedding = torch.cat((video_text_embedding, text_embedding), dim=-1).unsqueeze(0).to(self.device)

        return combined_embedding

    def __len__(self):
        """Возвращает размер датасета (всегда 1)."""
        return 1

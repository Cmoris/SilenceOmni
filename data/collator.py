import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import transformers

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # === 1. Text/Labels ===
        input_ids = [b["input_ids"].squeeze(0) for b in batch]
        labels = [b["labels"].squeeze(0) for b in batch]
        attention_mask = [torch.ones_like(x) for x in input_ids]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_type_id).to(torch.long)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(torch.long)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # === 2. Video/Audio batch ===
        video_lists = [b["video"] for b in batch]  # [list of chunks], [1,T,C,H,W]
        audio_lists = [b["audio"] for b in batch]  # [list of chunks], [n_mels, T]
        # Calculate chunks
        chunk_counts = [len(v) for v in video_lists]
        max_chunks = max(chunk_counts)

        # Get T_video, T_audio to align with each other
        max_video_frames = max(v.size(0) for video_chunks in video_lists for v in video_chunks)
        max_audio_frames = max(a.size(0) for audio_chunks in audio_lists for a in audio_chunks)

        # === 3. pad video and audio ===
        padded_videos, padded_audios = [], []
        for video_chunks, audio_chunks in zip(video_lists, audio_lists):

            padded_videos.append(torch.nn.utils.rnn.pad_sequence(video_chunks, batch_first=True))  # [N_chunks, T, C, H, W]
            padded_audios.append(torch.stack(audio_chunks, batch_first=True))  # [N_chunks, n_mels, T_audio]

        # === 4. stack batch ===
        video_tensor = torch.stack(padded_videos, dim=0).permute(1,0,2,3,4,5)  # [B, N_chunks, T, C, H, W]
        audio_tensor = torch.stack(padded_audios, dim=0).permute(1,0,2,3)  # [B, N_chunks, n_mels, T_audio]

        # # === 5. construct chunk mask ===
        # chunk_mask = torch.zeros((len(batch), max_chunks), dtype=torch.bool)
        # for i, n in enumerate(chunk_counts):
        #     chunk_mask[i, :n] = 1  # [B, N_chunks], 1=valid, 0=padding
            
        sources = dict(
                        labels=labels,        # [B, L]
                        videos=video_tensor,  # [B, N_chunks, C, T, H, W]
                        audios=audio_tensor,  # [B, N_chunks, n_mels, T_audio]
                        chunk_counts=chunk_counts 
                    )

        batch = {
            "input_ids": input_ids,           # [B, L]
            "labels": labels,                 # [B, L]
            "attention_mask": attention_mask, # [B, L]
            "sources": sources        
        }
        return batch

import functools, torch
from transformers import(
    AutoTokenizer, 
    AutoConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    LogitsProcessor, 
    logging
)
from torchcodec.decoders import AudioDecoder, VideoDecoder
import numpy as np
import os

# from ..constants import MODAL_INDEX_MAP
MODAL_INDEX_MAP = {
    "<video>": -201,
    "<audio>": -202,
    "<video><audio>": -203
}

logger = logging.get_logger(__name__)

FPS = 30
SR = 16000

def get_mm_reader(video_path, audio_path):
    video_decoder = VideoDecoder(source=video_path)
    audio_decoder = AudioDecoder(source=audio_path, sample_rate=SR)
    audio_clip = audio_decoder.get_all_samples().data.squeeze(0)
    video_clip = video_decoder[:]
    video_duration = video_decoder.metadata.duration_seconds
    video_num_frames = video_decoder.metadata.num_frames
    audio_num_frames = len(audio_clip)
    video_clip_pts = np.linspace(1/FPS, video_duration, video_num_frames)
    audio_clip_pts = np.linspace(1/SR, video_duration, audio_num_frames)
    mm_reader = dict(
        video_clip=video_clip,
        video_clip_pts=video_clip_pts,
        audio_clip=audio_clip,
        audio_clip_pts=audio_clip_pts
    )
    return mm_reader

def get_clip(
    clip: torch.Tensor,
    timestamps: torch.Tensor, 
    pts: np.ndarray, 
    pts_index_from: int = 0,
    factor: int = 0
):
    while len(timestamps) % factor != 0:
        timestamps = torch.cat([timestamps, timestamps[:-1] + 1/FPS], dim=0)
    clip_idxs = []
    for timestamp in timestamps:
        while pts_index_from < len(pts) and pts[pts_index_from] < timestamp:
            pts_index_from += 1
        if pts_index_from >= len(pts):
            break
        clip_idxs.append(pts_index_from)
        
    while len(clip_idxs) % factor != 0:
        clip_idxs = clip_idxs[:-1]
        timestamps = timestamps[:-1]
        
    clip = clip[clip_idxs]
    
    return clip, timestamps, clip_idxs


def load_pretrained_model(model_path, model_base, model_args, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    config = AutoConfig.from_pretrained(model_base)

    processor = Qwen2_5OmniProcessor.from_pretrained(model_base)
    print('Loading SilenceQwen3 from base model...')
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')
    
    return model, config, processor


def model_init(model_path=None, **kwargs):
    model_base = kwargs.get("model_base", None)
    model_args = kwargs.get("model_args", None)
    model, config, processor= load_pretrained_model(model_path, model_base, model_args)
    return model, config, processor


class ThresholdLogitsProcessor(LogitsProcessor):
    def __init__(self, token_id: int, base_threshold: float, step: float):
        self.token_id = token_id
        self.base_threshold = base_threshold
        self.step = step
        self.count = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        threshold = self.base_threshold + self.step * self.count 
        low_confidence = torch.softmax(scores, dim=-1)[:, self.token_id] <= threshold
        if low_confidence.any():
            scores[low_confidence, self.token_id] = -float("inf")
        self.count += 1
        return scores
    
class SilenceDemoInfer:
    VIDEO_PLAY_END = object()
    VIDEO_PLAY_CONTINUE = object()
    initial_FPS_frames = int(FPS)
    streaming_FPS_frames = int(FPS)
    initial_SR_frames: int = int(SR)
    streaming_SR_frames: int = int(SR)
    initial_video_time_interval = initial_FPS_frames / FPS
    streaming_video_time_interval = streaming_FPS_frames / FPS
    initial_audio_time_interval = initial_SR_frames / SR
    streaming_audio_time_interval = streaming_SR_frames / SR
    frame_time_interval = 1 / FPS
    audio_time_interval = 1 / SR

    def __init__(self,
                 model_args, 
                 device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
                
        self.model, config, self.processor = model_init(
                model_path=model_args.model_path,
                model_base = model_args.model_base,
                model_args=model_args
            )
        print(self.model)

        self.streaming_eos_token_id = self.processor.tokenizer(' ...').input_ids[-1]
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": 'livecc'},
            ]
        }
        texts = self.processor.apply_chat_template([message], tokenize=False)
        self.system_prompt_offset = texts.index('<|im_start|>user')
        self._cached_video_readers_with_hw = {}
        self._cached_audio_readers_with_hw = {}

    @torch.inference_mode()
    def live_cc(
        self,
        message: str,
        state: dict,
        default_query: str = 'Please describe the video.',
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        streaming_eos_base_threshold: float = None, 
        streaming_eos_threshold_step: float = None, 
        hf_spaces: bool = False,
        **kwargs,
    ): 
        """
        state: dict, (maybe) with keys:
            video_path: str, video path
            video_timestamp: float, current video timestamp
            last_timestamp: float, last processed video timestamp
            last_video_pts_index: int, last processed video frame index
            video_pts: np.ndarray, video pts
            last_history: list, last processed history
        """
        # 1. preparation: video_reader, and last processing info
        video_timestamp, last_video_timestamp = state.get('video_timestamp', 0), state.get('last_video_timestamp', -1 / FPS)
        audio_timestamp, last_audio_timestamp = state.get('audio_timestamp', 0), state.get('last_audio_timestamp', -1 / SR)
        video_path = state.get('video_path', None)
        audio_path = state.get('audio_path', None)
        if not video_path:
            return
        if not audio_path:
            return
        mm_reader = get_mm_reader(video_path, audio_path)
        video_clip = mm_reader["video_clip"]
        audio_clip = mm_reader["audio_clip"]
        video_clip_pts = mm_reader["video_clip_pts"]
        audio_clip_pts = mm_reader["audio_clip_pts"]
        state['video_pts'] = torch.from_numpy(video_clip_pts)
        state['audio_pts'] = torch.from_numpy(audio_clip_pts)
        state['last_video_pts_index'] = -1
        state['last_audio_pts_index'] = -1
        
        video_pts = state.get('video_pts', None)
        audio_pts = state.get('audio_pts', None)
        if video_pts is None:
            return
        video_timestamp = min(video_timestamp, video_pts[-1])
        audio_timestamp = min(audio_timestamp, audio_pts[-1])
        if last_video_timestamp + self.frame_time_interval > video_pts[-1]:
            state['video_end'] = True
            return 

        last_video_pts_index = state['last_video_pts_index']
        last_audio_pts_index = state['last_audio_pts_index']

        # 2. which frames will be processed
        initialized = last_video_timestamp >= 0
        if not initialized:
            video_timestamp = max(video_timestamp, self.initial_video_time_interval)
            audio_timestamp = max(audio_timestamp, self.initial_audio_time_interval)
        if video_timestamp <= last_video_timestamp + self.frame_time_interval:
            return
        video_timestamps = torch.arange(last_video_timestamp + self.frame_time_interval, video_timestamp, self.frame_time_interval) # add compensation
        audio_timestamps = torch.arange(last_audio_timestamp + self.audio_time_interval, audio_timestamp, self.audio_time_interval)
        # 3. fetch frames in required timestamps
        video_clip, video_timestamps, video_clip_idxs = get_clip(
            clip=video_clip,
            timestamps=video_timestamps,
            pts=video_clip_pts,
            pts_index_from=last_video_pts_index+1,
            factor=FPS
        )
        audio_clip, audio_timestamps, audio_clip_idxs = get_clip(
            clip=audio_clip,
            timestamps=audio_timestamps,
            pts=audio_clip_pts,
            pts_index_from=last_audio_pts_index+1,
            factor=SR
        )

        if len(audio_clip_idxs) == 0:
            return
        if len(video_clip_idxs) == 0:
            return
        state['last_video_pts_index'] = video_clip_idxs[-1]
        state['last_video_timestamp'] = video_timestamps[-1]
        
        state['last_audio_pts_index'] = audio_clip_idxs[-1]
        state['last_audio_timestamp'] = audio_timestamps[-1]
        
        # 4. organize to interleave frames
        interleave_video_clips, interleave_video_timestamps = [], []
        interleave_audio_clips, interleave_audio_timestamps = [], []
        if not initialized:
            interleave_video_clips.append(video_clip[:self.initial_FPS_frames])
            interleave_video_timestamps.append(video_timestamps[:self.initial_FPS_frames])
            video_clip = video_clip[self.initial_FPS_frames:]
            video_timestamps = video_timestamps[self.initial_FPS_frames:]
            
            interleave_audio_clips.append(audio_clip[:self.initial_SR_frames])
            interleave_audio_timestamps.append(audio_timestamps[:self.initial_SR_frames])
            audio_clip = audio_clip[self.initial_SR_frames:]
            audio_timestamps = audio_timestamps[self.initial_SR_frames:]
        if len(video_clip) > 0 and len(audio_clip) > 0:
            interleave_video_clips.extend(list(video_clip.split(self.streaming_FPS_frames)))
            interleave_video_timestamps.extend(list(video_timestamps.split(self.streaming_FPS_frames)))

            interleave_audio_clips.extend(list(audio_clip.split(self.streaming_SR_frames)))
            interleave_audio_timestamps.extend(list(audio_timestamps.split(self.streaming_SR_frames)))

        # 5. make conversation and send to model
        for i in range(len(interleave_video_timestamps)):
            
            start_timestamp, stop_timestamp = interleave_video_timestamps[i][0].item(), interleave_video_timestamps[i][-1].item() + self.frame_time_interval

            interleave_video_clip = interleave_video_clips[i]
            interleave_audio_clip = interleave_audio_clips[i]
            
            conversation = [{
                    "role": "user",
                    "content": [
                        {"text": f" Time={start_timestamp:.1f}-{stop_timestamp:.1f}s"},
                        {"video": interleave_video_clip},
                        {"audio": interleave_audio_clip},
                    ]
                }]
            
            if not message and not state.get('message', None):
                message = default_query
                logger.warning(f'No query provided, use default_query={default_query}')
            if message and state.get('message', None) != message:
                conversation[0]["content"].append({"text":message})
                state['message'] = message
            
            texts = self.processor.apply_chat_template(conversation, tokenize=False)
            past_ids = state.get('past_ids', None)
            
            if past_ids is not None:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
            
            inputs = self.processor(text=texts, audio=interleave_audio_clip, images=None, videos=interleave_video_clip, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs.to(self.model.device)
            if past_ids is not None:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1)
            if streaming_eos_base_threshold is not None:
                logits_processor = [ThresholdLogitsProcessor(self.streaming_eos_token_id, streaming_eos_base_threshold, streaming_eos_threshold_step)]
            else:
                logits_processor = None
                
            outputs = self.model.generate(
                **inputs, past_key_values=state.get('past_key_values', None), 
                return_dict_in_generate=True, do_sample=do_sample, 
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                max_new_tokens=16,
                pad_token_id=self.model.config.eos_token_id,
            )
            state['past_key_values'] = outputs.past_key_values
            state['past_ids'] = outputs.sequences[:, :-1]

            response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            if hf_spaces:
                light_state = {k: v for k, v in state.items() if k not in ['past_ids', 'past_key_values']}
                yield (start_timestamp, stop_timestamp), response, light_state
            else:
                yield (start_timestamp, stop_timestamp), response, state

    @torch.inference_mode()
    def video_qa(
        self,
        message: str,
        history: list,
        state: dict,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        hf_spaces: bool = False,
        **kwargs,
    ): 
        """
        state: dict, (maybe) with keys:
            video_path: str, video path
            video_timestamp: float, current video timestamp
            last_timestamp: float, last processed video timestamp
            last_video_pts_index: int, last processed video frame index
            video_pts: np.ndarray, video pts
            last_history: list, last processed history
        """
        video_path = state.get('video_path', None)
        conversation = []
        if hf_spaces:
            for past_message in history:
                content = [{"type": "text", "text": past_message['content']}]
                if video_path: # only use once
                    content.insert(0, {"type": "video", "video": video_path})
                    video_path = None
                conversation.append({"role": past_message["role"], "content": content})
        else:
            pass # use past_key_values
        past_ids = state.get('past_ids', None)
        content = [{"type": "text", "text": message}]
        if past_ids is None and video_path: # only use once
            content.insert(0, {"type": "video", "video": video_path})
        conversation.append({"role": "user", "content": content})
        image_inputs, video_inputs = process_vision_info(conversation)
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, return_tensors='pt')
        if past_ids is not None:
            texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            return_attention_mask=False
        )
        inputs.to(self.model.device)
        if past_ids is not None:
            inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
        outputs = self.model.generate(
            **inputs, past_key_values=state.get('past_key_values', None), 
            return_dict_in_generate=True, do_sample=do_sample, 
            repetition_penalty=repetition_penalty,
            max_new_tokens=512,
            pad_token_id=self.model.config.eos_token_id,
        )
        state['past_key_values'] = outputs.past_key_values if not hf_spaces else None
        state['past_ids'] = outputs.sequences[:, :-1] if not hf_spaces else None
        response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        return response, state

    @torch.inference_mode()
    def live_cc_once_for_evaluation(
        self,
        query: str,
        video: str,
        video_start: float = 0,
        video_end: float = None,
        remote_loader: callable = None,
        max_new_tokens: int = 32,
        repetition_penalty: float = 1.05,
    ): 
        # 1. read video clip
        clip, _ = _read_video_decord_plus({'video': video, 'video_start': video_start, 'video_end': video_end, 'remote_loader': remote_loader})
        clip = _spatial_resize_video(clip)

        # 2. organize to interleave frames
        interleave_clips = []
        ## 2.1 initial_FPS_frames
        interleave_clips.append(clip[:self.initial_FPS_frames])
        clip = clip[self.initial_FPS_frames:]
        ## 2.2 streaming_FPS_frames
        if len(clip) > 0:
            interleave_clips.extend(list(clip.split(self.streaming_FPS_frames)))
        
        # 3. make conversation and send to model
        past_key_values = None
        responses = []
        for i, clip in enumerate(interleave_clips):
            if i == 0:
                start_timestamp, stop_timestamp = 0, self.initial_time_interval
            else:
                start_timestamp, stop_timestamp = stop_timestamp, stop_timestamp + self.streaming_time_interval
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": clip}
                ]
            }
            if not past_key_values:
                message['content'].append({"type": "text", "text": query})
            texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt')
            if past_key_values:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
            inputs = self.processor(
                text=texts,
                images=None,
                videos=[clip],
                return_tensors="pt",
            )
            inputs.to(self.model.device)
            if past_key_values:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
            outputs = self.model.generate(
                **inputs, past_key_values=past_key_values, 
                return_dict_in_generate=True, 
                max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, 
                pad_token_id=self.model.config.eos_token_id,
            )
            past_key_values = outputs.past_key_values
            past_ids = outputs.sequences[:, :-1]
            responses.append([
                video_start + start_timestamp, 
                video_start + stop_timestamp, 
                self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            ])
        return responses

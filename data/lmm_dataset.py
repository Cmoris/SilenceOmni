from dataclasses import dataclass, field
import json, torch, random, tqdm, io, functools
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import AudioDecoder, VideoDecoder
from transformers import logging, AutoProcessor, AutoTokenizer, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# from ..constants import MODAL_INDEX_MAP
MODAL_INDEX_MAP = {
    "<video>": -201,
    "<audio>": -202,
    "<video><audio>": -203
}

logger = logging.get_logger(__name__)

FPS = 30
SR = 16000

@dataclass
class DataArguments:
    annotation_paths: list[str] = field(default_factory=list) 
    initial_fps_frames: int = int(FPS)
    streaming_fps_frames: int = int(FPS)
    initial_audio_frames: int = int(SR)
    streaming_audio_frames: int = int(SR)
    with_context: bool = False
    #Processor

# --- some utils ---
def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()

def bytes_to_pil(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')

def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
    phrase = ''
    for i, (ws, we, word) in enumerate(text_stream[start_from:]):
        if timestamp >= we:
            phrase += ' ' + word.strip()
        else:
            break
    return phrase.strip(), i + start_from

# --- some utils ---
def tokenizer_multimodal_token(texts, tokenizer, multimodal_token=None, return_tensors=None):
    """Tokenize text and multimodal tag to input_ids.

    Args:
        prompt (str): Text prompt (w/ multimodal tag), e.g., '<video>\nDescribe the video.'
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        multimodal_token (int): Token index corresponding to the multimodal tag.
    """
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(texts, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(texts.split(multimodal_token))]
        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def split_video_sides(frame:torch.Tensor):
    width = frame.size(1)
    
    half_width = width // 2
    
    left_half = frame[:, :half_width]
    right_half = frame[:, half_width:half_width*2]
    
    return left_half, right_half


class LMMDatasetForQwen(Dataset):
    def __init__(
        self, 
        annotation_paths: list[str], 
        data_args, 
        processor: AutoProcessor,
        with_context:bool=False,
        initial_fps_frames: int=DataArguments.initial_fps_frames, 
        streaming_fps_frames: int=DataArguments.streaming_fps_frames, 
        initial_audio_frames: int=DataArguments.initial_audio_frames,
        streaming_audio_frames: int=DataArguments.streaming_audio_frames,
        **kwargs
    ):
        super().__init__()
        self.handles = []
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), f"Please organize the annotations in JSONL format, with each data sample on a separate line, and the last line stores the seek indices"
            logger.warning(f'Load {annotation_path}. Please ensure its last line stores the seek indices...')
            seeks = json.loads(readlastline(annotation_path))
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f'Successfully loaded {annotation_path}')

        self.data_args = data_args

        if 'Qwen2_5Omni' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer(
                '<|im_start|>assistant\n<|im_end|>').input_ids
        else:
            raise NotImplementedError(f"Preprocess not implemented for {processor.__class__.__name__}")

        self.processor = processor
        self.with_context = with_context
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
        self.initial_audio_frames = initial_audio_frames
        self.streaming_audio_frames = streaming_audio_frames
    
    def load_conversation(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line
    
    def preprocess_text(self, element: str):
        if self.with_context and ('title' in element or 'previous' in element):
            previous = element.get('previous', '')
            if previous:
                title = ''
            else:
                title = element.get('title', '')
            return (element['text'] + f"\n{title}\n{previous}").strip()
        return element['text']

    def preprocess_conversation_stream(self, conversation: list):
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']
        user_video_dict, user_audio_dict, user_query_dict = user_content
        assert 'video' in user_video_dict, 'please check your data, ensure the video info in the first user content'
        assert 'audio' in user_audio_dict, 'please check your data, ensure the audio info in the first user content'
        assistant_text_stream = assistant_message['content'][0]['text_stream']
        # get clip duration
        clip_start_timestamp, clip_end_timestamp = assistant_text_stream[0]["start"], assistant_text_stream[-1]["end"]
        video_start_frame, video_end_frame = int(clip_start_timestamp*FPS), int(clip_end_timestamp*FPS)
        audio_start_frame, audio_end_frame = int(clip_start_timestamp*SR), int(clip_end_timestamp*SR)
        # load video in strict fps
        video_decoder = VideoDecoder(source=user_video_dict['video'])
        audio_decoder = AudioDecoder(source=user_audio_dict['audio'], sample_rate=SR)
        audio_clip = audio_decoder.get_all_samples().data.squeeze(0)[audio_start_frame:audio_end_frame]
        video_clip = video_decoder.get_frames_in_range(start=video_start_frame, stop=video_end_frame).data
        video_duration = video_decoder.metadata.duration_seconds
        video_num_frames = video_decoder.metadata.num_frames
        video_clip_pts = np.linspace(0, video_duration, video_num_frames)[video_start_frame:video_end_frame]

        # make conversation
        start_timestamp, end_timestamp = clip_start_timestamp, clip_start_timestamp+self.streaming_fps_frames/FPS
        phrase, next_start_from = get_phrase_before_timestamp(assistant_text_stream, video_clip_pts[self.initial_fps_frames-1])
        audio_frames = audio_clip[:self.initial_audio_frames].numpy()
        video_frames = video_clip[:self.initial_fps_frames].numpy()
        conversation = [
            {
                'role': 'user', 'content':[
                    {"type":"text", "text": f" Time={start_timestamp:.1f}-{end_timestamp:.1f}s"},
                    {"type":"video", "video": video_frames},
                    {"type":"audio", "audio": audio_frames},
                    user_query_dict
                ]
                
            },
            {'role': 'assistant', 'content': phrase + ' ...'}
        ]

        frames_list = [video_frames]
        audio_frames_list = [audio_frames]
        for i, j in zip(range(self.initial_fps_frames, len(video_clip), self.streaming_fps_frames),
                        range(self.initial_audio_frames, len(audio_clip), self.streaming_audio_frames)):
            start_timestamp, end_timestamp = (i+video_start_frame) / FPS, (i+video_start_frame + self.streaming_fps_frames) / FPS
            if (i + self.streaming_fps_frames-1) < len(video_clip_pts):
                pts = video_clip_pts[i + self.streaming_fps_frames-1]  
            else: 
                pts = video_clip_pts[-1]
                next_start_from += 1
            phrase, next_start_from = get_phrase_before_timestamp(assistant_text_stream, pts, start_from=next_start_from)
            video_frames = video_clip[i:i + self.streaming_fps_frames].numpy()
            audio_frames = audio_clip[j:j + self.streaming_audio_frames].numpy()

            conversation.extend([
                {
                    'role': 'user', 'content': [
                        {"type":"text", "text": f" Time={start_timestamp:.1f}-{end_timestamp:.1f}s"},
                        {"type":"video", "video": video_frames},
                        {"type":"audio", "audio": audio_frames},
                    ]
                },
                {'role': 'assistant', 'content': phrase + ' ...'}
            ])
            frames_list.append(video_frames)
            audio_frames_list.append(audio_frames)

        # remove the last with no phrase
        while conversation[-1]['content'] == ' ...':
            conversation = conversation[:-2]
            frames_list = frames_list[:-1]
            audio_frames_list = audio_frames_list[:-1]
        return conversation, frames_list, audio_frames_list

    def getitem(self, index):
        conversation = self.load_conversation(index)
        conversation, video_inputs, audio_inputs = self.preprocess_conversation_stream(conversation)
        import ipdb; ipdb.set_trace()
        texts = self.processor.apply_chat_template(conversation, tokenize=False)
        inputs = self.processor(text=texts, audio=audio_inputs, images=None, videos=video_inputs, return_tensors="pt", padding=True, use_audio_in_video=False)
        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]
        batch = {
            'input_ids':input_ids.squeeze(0),
            'labels':labels.squeeze(0),
        }

        return batch

    def __getitem__(self, index):
        max_tries = 1
        # for _ in range(max_tries):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         logger.warning(f"Failed {_}-th try to get item {index}: {e}")
        #         index = random.randint(0, self.__len__() - 1)
        #         logger.warning(f"Retrying to get item {index}")
        # raise Exception(f"Failed to get item after {max_tries} retries")
        
        return self.getitem(index)

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1
        return batched_inputs[0]

    def __len__(self):
        return len(self.handles)

if __name__ == "__main__":
    dataset = LMMDatasetForQwen(
        data_args = DataArguments,
        annotation_paths=[
            '/n/work1/muyun/Dataset/zoom2025/labels/test.jsonl', 
        ], 
        processor=Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B"),
        with_context=False,
    )
    print(len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for i, data in enumerate(dataloader):
        print(data["input_ids"].size())
    # for i in tqdm.tqdm(range(len(dataset))):
    #     conversation = dataset.__getitem__(i)
        # inputs.to('cuda')
        # with torch.inference_mode():
        #     model(**inputs)
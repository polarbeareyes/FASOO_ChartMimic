from openai import OpenAI
import os
import time
from common.registry import registry
import base64
import torch
import transformers
from transformers import GenerationConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

@registry.register_llm("qwenvl")
class QwenVL:
    def __init__(
        self,
        engine="Qwen/Qwen2-VL-2B-Instruct",
        temperature=0.1,
        max_tokens=4096,
        top_p=0.95,
        stop=[""],
        retry_delays=1,
        max_retry_iters=100,
        context_length=4096,
        system_message="",
    ):
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        self.system_message = system_message
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.engine, trust_remote_code=True)
        self.generation_config = self.model.generation_config
        self.generation_config.max_new_tokens = self.max_tokens
        self.generation_config.temperature=self.temperature
        self.generation_config.top_p=self.top_p
        self.generation_config.do_sample=(self.temperature != 0)


    def generate(self, conversation):
        conversation = self._convert_conversation(conversation)
        response = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    def _convert_conversation(self, conversation):
        converted_conversation = []
        if len (conversation) !=1 :
            print("QwenVL only supports 1-turn conversation")
            raise NotImplementedError
        for message in conversation:
            for content in message["content"]:
                if content["type"] == "text":
                    converted_conversation.append({
                        'text': content["text"]
                    })
                elif content["type"] == "image":
                    converted_conversation.append({
                        'image': content["image_url"]
                    })              
                else:
                    raise NotImplementedError
        return converted_conversation


    def _encode_base_image(self, file):
        with open(file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @classmethod
    def from_config(cls, config):
        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        system_message = config.get("system_message", "")
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        retry_delays = config.get("retry_delays", 10)
        context_length = config.get("context_length", 4096)
        return cls(
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            retry_delays=retry_delays,
            system_message=system_message,
            context_length=context_length,
            stop=stop,
        )

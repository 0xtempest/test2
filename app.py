import copy
import os, gc, torch
import time
from huggingface_hub import hf_hub_download
from pynvml import *
from torch.nn import functional as F
import numpy as np
# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)

os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)

from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.7, max_tokens=256)
model = LLM(model="egirlsai/egirls-chat")

import asyncio

async def getModel():
    return model

# def sample_logits_typical(logits, temperature=1.0, top_p=0.95, **kwargs):
#         probs = F.softmax(logits.float(), dim=-1)
#         logits = -torch.log(probs)
#         ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
#         shifted_logits = torch.abs(logits - ent)
#         sorted_ids = torch.argsort(shifted_logits)
#         sorted_logits = shifted_logits[sorted_ids]
#         sorted_probs = probs[sorted_ids]
#         cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
#         cutoff = np.sum(cumulative_probs < top_p)
#         probs[shifted_logits > sorted_logits[cutoff]] = 0
#         if temperature != 1.0:
#             probs = probs ** (1.0 / temperature)
#         out = torch.multinomial(probs, num_samples=1)[0]
#         return int(out)

async def evaluate(
    prompt,
    model
):
    return model.generate(prompt, sampling_params)

def removeTokens(text):
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")

async def buildPrompt(system_prompt, message, conversation):
    fullprompt = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
    
    for m in conversation:
        if m['role'] == 'user':
            fullprompt += f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            fullprompt += "<|im_start|>user\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
        elif m['role'] == 'assistant':
            fullprompt += "<|im_start|>assistant\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
            
    
    # trim message
    message = message.strip()
            
    fullprompt += "<|im_start|>user\n" + removeTokens(message) + "<|im_end|>\n<|im_start|>assistant\n"
    print ("## Prompt ##")
    print (fullprompt)
    
    return fullprompt
    

async def handleLLAMA(system_prompt, message, conversation, model):
    
    system_prompt = system_prompt.strip() if system_prompt != None else ''
    
    fullprompt = await buildPrompt(system_prompt, message, conversation, model)
    
    full_response = fullprompt
    for token, statee in evaluate(fullprompt, model):
        full_response += token
        yield token
        
    gc.collect()
        

from aiohttp import web
import logging

async def handle(request):
    model = await getModel()
    try:
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={'Content-Type': 'text/plain'},
        )
        await response.prepare(request)
        # get the request data (json)
        data = await request.json()    
        
        start_time = time.perf_counter()
        total_tokens = 0
        
        # run handleRwkv generator and output the result
        async for token in handleLLAMA(data['system_prompt'] if 'system_prompt' in data else None, data['message'], data['conversation'], model):
            await response.write(token.encode())
            totalTokens += 1

        end_time = time.perf_counter()
        time_taken = end_time - start_time
        print(f"## Time taken (seconds): {time_taken} ##")
        print(f"## Tokens generated: {total_tokens} ##")
        print(f"## Tokens per second: {total_tokens / time_taken} ##")

        await response.write_eof()
        return response
    except OSError:
        print("## Client disconnected ##")

app = web.Application()
logging.basicConfig(level=logging.DEBUG)
app.add_routes([
    web.post('/api/conversation', handle)
])

web.run_app(app, port=9998)

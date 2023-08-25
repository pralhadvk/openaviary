from fastapi import FastAPI, HTTPException, Request, status, Response as FastAPIResponse, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import aiohttp
import async_timeout
import OpenAIHTTPException
from models import (
    Message, 
    Prompt, 
    DeltaChoices, 
    ChatCompletion, 
    DeltaRole, 
    DeltaEOS, 
    DeltaContent, 
    QueuePriority, 
    MessageChoices, 
    Usage
)
import uuid
import os
import time
import logging
from starlette.responses import StreamingResponse
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,GPTNeoForCausalLM,GPT2Tokenizer
import openai

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

TIMEOUT = os.environ.get("AVIARY_ROUTER_HTTP_TIMEOUT", 175)

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_exception_handler(OpenAIHTTPException,open)

async def generate_text(prompt,model_choice):
    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        
        if not model_choice:
            raise HTTPException(status_code=400, detail="Model choice cannot be empty.")
        
        user_message = next((message["content"] for message in prompt.messages if message["role"] == "user"), None)

        if model_choice == "gpt2":
            tok = AutoTokenizer.from_pretrained(model_choice)
            model = AutoModelForCausalLM.from_pretrained(model_choice)
        elif model_choice == "gptneo":
            tok = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        
        # tok = AutoTokenizer.from_pretrained("gpt2")
        # model = AutoModelForCausalLM.from_pretrained("gpt2")
        inputs = tok([prompt], return_tensors="pt")
        streamer = TextIteratorStreamer(tok)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1000,no_repeat_ngram_size=2)
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        # model.generate(generation_kwargs)

        # for text in streamer:
        #     print(text)
        
        return StreamingResponse(streamer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Text generation failed. Please check the server logs for more information.")

@app.post("/v1/chat/completions")                           
async def chat(
        self,
        model: Annotated[str, Body()],
        messages: List[Message],
        request: Request,
        response: FastAPIResponse,
        max_tokens: Annotated[Optional[int], Body()] = None,
        temperature: Annotated[Optional[float], Body()] = None,
        top_p: Annotated[Optional[float], Body()] = None,
        n: Annotated[int, Body()] = 1,
        stream: Annotated[bool, Body()] = False,
        logprobs: Annotated[Optional[int], Body()] = None,
        echo: Annotated[bool, Body()] = False,
        stop: Annotated[Optional[List[str]], Body()] = None,
        presence_penalty: Annotated[float, Body()] = None,
        frequency_penalty: Annotated[float, Body()] = None,
        logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
        user: Annotated[Optional[str], Body()] = None,
        top_k: Annotated[Optional[int], Body()] = None,
        typical_p: Annotated[Optional[float], Body()] = None,
        watermark: Annotated[Optional[bool], Body()] = False,
        seed: Annotated[Optional[int], Body()] = None,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Args:
            model: The model to query.
            messages: A list of messages describing the conversation so far.
                Contains a required "role", which is the role of the author of this
                message. One of "system", "user", or "assistant".
                Also contains required "content", the contents of the message, and
                an optional "name", the name of the author of this message.
            max_tokens: The maximum number of tokens to generate. Defaults to inf.
            temperature: What sampling temperature to use.
            top_p: An alternative to sampling with temperature, called nucleus sampling.
            n: How many completions to generate for each prompt.
            stream: Whether to stream back partial progress.
            logprobs: Include the log probabilities on the `logprobs` most likely
                tokens, as well the chosen tokens.
            echo: Echo back the prompt in addition to the completion.
            stop: Up to 4 sequences where the API will stop generating further tokens.
                The returned text will not contain the stop sequence.
            presence_penalty: Number between -2.0 and 2.0.
                Positive values penalize new tokens based on whether they appear in
                the text so far, increasing the model's likelihood to talk about
                new topics.
            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
                new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.
            best_of: Generates `best_of` completions server-side and returns the "best".
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.
            top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            typical_p: Typical Decoding mass. See Typical Decoding for Natural Language Generation
                (https://arxiv.org/abs/2202.00666) for more information.
            watermark: Watermarking with A Watermark for Large Language Models
                (https://arxiv.org/abs/2301.10226).
            seed: Random sampling seed.

        Returns:
            A response object with completions.
        """
        
        prompt = Prompt(                                            #validate prompt defined in models.py
            prompt=messages,
            parameters={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "top_k": top_k,
                "typical_p": typical_p,
                "watermark": watermark,
                "seed": seed,
            },
            stopping_sequences=stop,
        )

        print(prompt.prompt)

        # def replace_prefix(model: str) -> str:                      
        #     return model.replace("--", "/")

        # def get_model_path(self, model: str):
        #     model = replace_prefix(model)
        #     route = self._routes.get(model)
        #     if route is None:
        #         raise OpenAIHTTPException(
        #             message=f"Invalid model '{model}'",
        #             status_code=status.HTTP_400_BAD_REQUEST,
        #             type="InvalidModel",
        #         )
        #     return route

        # route = get_model_path(model)           #use api for testing

        # route = "gpt2"

        # if stream:

        #     async def completions_wrapper():
        #         with async_timeout.timeout(TIMEOUT):
        #             finish_reason = None
        #             choices: List[DeltaChoices] = [
        #                 DeltaChoices(                               #DeltaChoices contains Union[DeltaRole, DeltaContent, DeltaEOS]
        #                                                             # DeltaRole: ["system", "assistant", "user"], DeltaContent: Content
        #                     delta=DeltaRole(role="assistant"),
        #                     index=0,
        #                     finish_reason=None,
        #                 )
        #             ]
        #             yield "data: " + ChatCompletion(
        #                 id=model + "-" + str(uuid.uuid4()),
        #                 object="text_completion",
        #                 created=int(time.time()),   
        #                 model=model,
        #                 choices=choices,
        #                 usage=None,
        #             ).model_dump_json() + "\n"

        #             all_results = []

        #     async def get_response_stream(
        #         self,
        #         route: str,
        #         model: str,
        #         prompt: Prompt,
        #         request: Request,
        #         priority: QueuePriority,
        #     ):
        #         is_first_token = True
        #         model_tags = {"model_id": model}        
        #         # self.metrics.requests_started.inc(tags=model_tags)

        #         req_id = uuid.uuid4().hex
                # start_time = time.monotonic()
                # logger.info(f"Starting request handling. Id: {req_id}.")

                # async for chunk in generate_text():
                #     print(chunk)
        #             responses = [
        #                 AviaryModelResponse.parse_raw(p)
        #                 for p in chunk.split(b"\n")
        #                 if p
        #             ]       
        #             combined_response = AviaryModelResponse.merge_stream(*responses)

        #             await self.hooks.trigger_post_execution_hook(
        #                 request, model, prompt.prompt, is_first_token, combined_response
        #             )
        #             # Track metrics
        #             for res in responses:
        #                 self.metrics.track(res, is_first_token, model)
        #                 is_first_token = False
        #                 yield res
        # end_time = time.monotonic()
        # logger.info(
        #     f"Completed request handling. Id: {req_id}. Took {end_time - start_time}s"
        # )

        #         async for results in get_response_stream(
        #             route,
        #             model,
        #             prompt,
        #             request,
        #             priority=QueuePriority.GENERATE_TEXT,
        #         ):
        #             results = results.dict()
        #             if results.get("error"):
        #                 response.status_code = results["error"]["code"]
        #                 logger.error(f"{results['error']}")
        #                 yield "data: " + AviaryModelResponse(
        #                     **results
        #                 ).json() + "\n"
        #             else:
        #                 all_results.append(AviaryModelResponse(**results))
        #                 finish_reason = results["finish_reason"]
        #                 if finish_reason:
        #                     continue
        #                 choices: List[DeltaChoices] = [
        #                     DeltaChoices(
        #                         delta=DeltaContent(
        #                             content=results["generated_text"] or ""
        #                         ),
        #                         index=0,
        #                         finish_reason=None,
        #                     )
        #                 ]
        #                 yield "data: " + ChatCompletion(
        #                     id=model + "-" + str(uuid.uuid4()),
        #                     object="text_completion",
        #                     created=int(time.time()),
        #                     model=model,
        #                     choices=choices,
        #                     usage=None,
        #                 ).model_dump_json() + "\n"

        #             choices: List[DeltaChoices] = [
        #                 DeltaChoices(
        #                     delta=DeltaEOS(),
        #                     index=0,
        #                     finish_reason=finish_reason,
        #                 )
        #             ]
        #             usage = (
        #                 Usage.from_response(
        #                     AviaryModelResponse.merge_stream(*all_results)
        #                 )
        #                 if all_results
        #                 else None
        #             )
        #             yield "data: " + ChatCompletion(
        #                 id=model + "-" + str(uuid.uuid4()),
        #                 object="text_completion",
        #                 created=int(time.time()),
        #                 model=model,
        #                 choices=choices,
        #                 usage=usage,
        #             ).model_dump_json() + "\n"
        #             yield "data: [DONE]\n"

        #     return StreamingResponse(
        #         completions_wrapper(),
        #         media_type="text/event-stream",
        #     )
        # else:
        #     with async_timeout.timeout(TIMEOUT):
        #         results = await self._query(model, prompt, request)
        #         if results.error:
        #             raise OpenAIHTTPException(
        #                 message=results.error.message,
        #                 status_code=results.error.code,
        #                 type=results.error.type,
        #             )
        #         results = results.dict()

        #         # TODO: pick up parameters that make sense, remove the rest

        #         choices: List[MessageChoices] = [
        #             MessageChoices(
        #                 message=Message(
        #                     role="assistant", content=results["generated_text"] or ""
        #                 ),
        #                 index=0,
        #                 finish_reason=results["finish_reason"],
        #             )
        #         ]
        #         usage = Usage.from_response(results)

        #         return ChatCompletion(
        #             id=model + "-" + str(uuid.uuid4()),
        #             object="text_completion",
        #             created=int(time.time()),
        #             model=model,
        #             choices=choices,
        #             usage=usage,
        #         )
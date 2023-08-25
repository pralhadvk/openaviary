from pydantic import BaseModel, validator
from typing import Union,List,Optional,Dict, Any, TypeVar,Literal
from enum import IntEnum
import yaml

TModel = TypeVar("TModel", bound="Model")
TCompletion = TypeVar("TCompletion", bound="Completion")
TChatCompletion = TypeVar("TChatCompletion", bound="ChatCompletion")

class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content

class DeltaRole(BaseModel):
    role: Literal["system", "assistant", "user"]

    def __str__(self):
        return self.role


class DeltaContent(BaseModel):
    content: str

    def __str__(self):
        return self.content


class DeltaEOS(BaseModel):
    class Config:
        extra = "forbid"

class DeltaChoices(BaseModel):
    delta: Union[DeltaRole, DeltaContent, DeltaEOS]
    index: int
    finish_reason: Optional[str]


class ModelData(BaseModel):
    id: str
    object: str
    owned_by: str
    permission: List[str]
    aviary_metadata: Dict[str, Any]

class Model(BaseModel):
    data: List[ModelData]
    object: str = "list"

    @classmethod
    def list(cls) -> TModel:
        pass

class MessageChoices(BaseModel):
    message: Message
    index: int
    finish_reason: str


class TextChoice(BaseModel):
    text: str
    index: int
    logprobs: dict
    finish_reason: Optional[str]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_response(
        cls, response: Union["AviaryModelResponse", Dict[str, Any]]
    ) -> "Usage":
        if isinstance(response, BaseModel):
            response = response.dict()
        return cls(
            prompt_tokens=response["num_input_tokens"] or 0,
            completion_tokens=response["num_generated_tokens"] or 0,
            total_tokens=(response["num_input_tokens"] or 0)
            + (response["num_generated_tokens"] or 0),
        )

class Completion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[TextChoice]
    usage: Optional[Usage]

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        use_prompt_format: bool = True,
        max_tokens: Optional[int] = 16,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> TCompletion:
        pass

class TextChoice(BaseModel):
    text: str
    index: int
    logprobs: dict
    finish_reason: Optional[str]

class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content

class Prompt(BaseModel):
    prompt: Union[str, List[Message]]
    use_prompt_format: bool = True
    parameters: Optional[Dict[str, Any]] = None
    stopping_sequences: Optional[List[str]] = None

class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Union[MessageChoices, DeltaChoices]]
    usage: Optional[Usage]

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> TChatCompletion:
        pass

class ErrorResponse(BaseModel):
    message: str
    internal_message: str
    code: int
    type: str
    param: Dict[str, Any] = {}

    
class MessageChoices(BaseModel):
    message: Message
    index: int
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_response(
        cls, response: Union["AviaryModelResponse", Dict[str, Any]]
    ) -> "Usage":
        if isinstance(response, BaseModel):
            response = response.dict()
        return cls(
            prompt_tokens=response["num_input_tokens"] or 0,
            completion_tokens=response["num_generated_tokens"] or 0,
            total_tokens=(response["num_input_tokens"] or 0)
            + (response["num_generated_tokens"] or 0),
        )

class QueuePriority(IntEnum):
    """Lower value = higher priority"""

    GENERATE_TEXT = 0
    BATCH_GENERATE_TEXT = 1

class ComputedPropertyMixin:
    """
    Include properties in the dict and json representations of the model.
    """

    # Replace with pydantic.computed_field once it's available
    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if isinstance(getattr(cls, prop), property)]

    def dict(self, *args, **kwargs):
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )
        return super().dict(*args, **kwargs)

    def json(
        self,
        *args,
        **kwargs,
    ) -> str:
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )

        return super().json(*args, **kwargs)
    
class BaseModelExtended(BaseModel):
    @classmethod
    def parse_yaml(cls, file, **kwargs) -> "BaseModelExtended":
        kwargs.setdefault("Loader", yaml.SafeLoader)
        dict_args = yaml.load(file, **kwargs)
        return cls.parse_obj(dict_args)

    def yaml(
        self,
        *,
        stream=None,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: Union[bool, None] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        **kwargs,
    ):
        """
        Generate a YAML representation of the model, `include` and `exclude` arguments as per `dict()`.
        """
        return yaml.dump(
            self.model_dump(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                skip_defaults=skip_defaults,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            ),
            stream=stream,
            **kwargs,
        )


class AviaryModelResponse(ComputedPropertyMixin, BaseModelExtended):
    generated_text: Optional[str] = None
    num_input_tokens: Optional[int] = None
    num_input_tokens_batch: Optional[int] = None
    num_generated_tokens: Optional[int] = None
    num_generated_tokens_batch: Optional[int] = None
    preprocessing_time: Optional[float] = None
    generation_time: Optional[float] = None
    timestamp: Optional[float] = None
    finish_reason: Optional[str] = None
    error: Optional[ErrorResponse] = None

    @validator("timestamp", always=True)
    def set_timestamp(cls, v):
        return v or time.time()

    @root_validator
    def text_or_error_or_finish_reason(cls, values):
        if (
            values.get("generated_text") is None
            and values.get("error") is None
            and values.get("finish_reason") is None
        ):
            raise ValueError(
                "Either 'generated_text' or 'error' or 'finish_reason' must be set"
            )
        return values

    @classmethod
    def merge_stream(cls, *responses: "AviaryModelResponse") -> "AviaryModelResponse":
        """
        Merge a stream of responses into a single response.

        The generated text is concatenated. Fields are maxed, except for
        num_generated_tokens and generation_time, which are summed.
        """
        if len(responses) == 1:
            return responses[0]

        generated_text = "".join(
            [response.generated_text or "" for response in responses]
        )
        num_input_tokens = [
            response.num_input_tokens
            for response in responses
            if response.num_input_tokens is not None
        ]
        num_input_tokens = max(num_input_tokens) if num_input_tokens else None
        num_input_tokens_batch = [
            response.num_input_tokens_batch
            for response in responses
            if response.num_input_tokens_batch is not None
        ]
        num_input_tokens_batch = (
            max(num_input_tokens_batch) if num_input_tokens_batch else None
        )
        num_generated_tokens = [
            response.num_generated_tokens
            for response in responses
            if response.num_generated_tokens is not None
        ]
        num_generated_tokens = (
            sum(num_generated_tokens) if num_generated_tokens else None
        )
        num_generated_tokens_batch = [
            response.num_generated_tokens_batch
            for response in responses
            if response.num_generated_tokens_batch is not None
        ]
        num_generated_tokens_batch = (
            sum(num_generated_tokens_batch) if num_generated_tokens_batch else None
        )
        preprocessing_time = [
            response.preprocessing_time
            for response in responses
            if response.preprocessing_time is not None
        ]
        preprocessing_time = max(preprocessing_time) if preprocessing_time else None
        generation_time = [
            response.generation_time
            for response in responses
            if response.generation_time is not None
        ]
        generation_time = sum(generation_time) if generation_time else None
        error = next(
            (response.error for response in reversed(responses) if response.error), None
        )

        return cls(
            generated_text=generated_text,
            num_input_tokens=num_input_tokens,
            num_input_tokens_batch=num_input_tokens_batch,
            num_generated_tokens=num_generated_tokens,
            num_generated_tokens_batch=num_generated_tokens_batch,
            preprocessing_time=preprocessing_time,
            generation_time=generation_time,
            timestamp=responses[-1].timestamp,
            finish_reason=responses[-1].finish_reason,
            error=error,
        )

    @property
    def total_time(self) -> Optional[float]:
        if self.generation_time is None and self.preprocessing_time is None:
            return None
        return (self.preprocessing_time or 0) + (self.generation_time or 0)

    @property
    def num_total_tokens(self) -> Optional[float]:
        try:
            return self.num_input_tokens + self.num_generated_tokens
        except Exception:
            return None

    @property
    def num_total_tokens_batch(self) -> Optional[float]:
        try:
            return self.num_input_tokens_batch + self.num_generated_tokens_batch
        except Exception:
            return None

    @property
    def total_time_per_token(self) -> Optional[float]:
        try:
            return self.total_time / self.num_total_tokens
        except Exception:
            return None

    @property
    def generation_time_per_token(self) -> Optional[float]:
        try:
            return self.generation_time / self.num_total_tokens
        except Exception:
            return None

    @property
    def total_time_per_token_batch(self) -> Optional[float]:
        try:
            return self.total_time / self.num_total_tokens_batch
        except Exception:
            return None

    @property
    def generation_time_per_token_batch(self) -> Optional[float]:
        try:
            return self.generation_time / self.num_total_tokens_batch
        except Exception:
            return None

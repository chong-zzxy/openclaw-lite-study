from providers.base import LLMProvider, LLMResponse, ToolCall

# OpenAIProvider 延迟导入，避免未安装 openai 时 import 就报错
def get_openai_provider():
    from providers.openai_provider import OpenAIProvider
    return OpenAIProvider

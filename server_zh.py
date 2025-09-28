# -*- coding: utf-8 -*-
"""
Anthropic API 代理服务器 - 支持 Gemini 和 OpenAI 模型
让 Anthropic 客户端（如 Claude Code）使用 Gemini、OpenAI 或直接使用 Anthropic 后端
"""

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# 从 .env 文件加载环境变量
load_dotenv()

# 配置日志系统
logging.basicConfig(
    level=logging.WARN,  # 改为 INFO 级别以显示更多详细信息
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 配置 uvicorn 使其更安静
import uvicorn
# 告诉 uvicorn 的日志记录器保持安静
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# 创建过滤器以阻止包含特定字符串的日志消息
class MessageFilter(logging.Filter):
    """日志消息过滤器，用于过滤掉不需要的日志信息"""
    def filter(self, record):
        # 阻止包含这些字符串的消息
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# 将过滤器应用到根日志记录器以捕获所有消息
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# 模型映射日志的自定义格式化器
class ColorizedFormatter(logging.Formatter):
    """自定义格式化器以突出显示模型映射"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # 对模型映射日志应用颜色和格式
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# 将自定义格式化器应用到控制台处理器
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# 创建 FastAPI 应用实例
app = FastAPI()

# 从环境变量获取 API 密钥
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 从环境变量获取 OpenAI 基础 URL（如果设置了）
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# 获取首选提供商（默认为 openai）
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# 从环境变量获取模型映射配置
# 如果未设置，默认为最新的 OpenAI 模型
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# OpenAI 模型列表
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # 添加默认大模型
    "gpt-4.1-mini" # 添加默认小模型
]

# Gemini 模型列表
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

def clean_gemini_schema(schema: Any) -> Any:
    """递归地从 JSON 模式中移除 Gemini 不支持的字段"""
    if isinstance(schema, dict):
        # 移除 Gemini 工具参数不支持的特定键
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # 检查字符串类型中不支持的 'format'
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"为 Gemini 模式中的字符串类型移除不支持的格式 '{schema['format']}'")
                schema.pop("format")

        # 递归清理嵌套模式（属性、项目等）
        for key, value in list(schema.items()): # 使用 list() 允许在迭代期间修改
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # 递归清理列表中的项目
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Anthropic API 请求的模型类
class ContentBlockText(BaseModel):
    """文本内容块"""
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    """图像内容块"""
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    """工具使用内容块"""
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    """工具结果内容块"""
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    """系统内容"""
    type: Literal["text"]
    text: str

class Message(BaseModel):
    """消息模型"""
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    """工具模型"""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    """思考配置"""
    enabled: bool = True

class MessagesRequest(BaseModel):
    """消息请求模型"""
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # 将存储原始模型名称
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # 重命名以避免冲突
        """模型字段验证器 - 处理模型映射逻辑"""
        original_model = v
        new_model = v # 默认为原始值

        logger.debug(f"📋 模型验证: 原始='{original_model}', 首选='{PREFERRED_PROVIDER}', 大模型='{BIG_MODEL}', 小模型='{SMALL_MODEL}'")

        # 移除提供商前缀以便于匹配
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- 映射逻辑 --- 开始 ---
        mapped = False
        if PREFERRED_PROVIDER == "anthropic":
            # 不重新映射到大/小模型，只添加前缀
            new_model = f"anthropic/{clean_v}"
            mapped = True

        # 根据提供商偏好将 Haiku 映射到 SMALL_MODEL
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # 根据提供商偏好将 Sonnet 映射到 BIG_MODEL
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # 如果未映射，为匹配已知列表的模型添加前缀
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # 技术上映射以添加前缀
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # 技术上映射以添加前缀
        # --- 映射逻辑 --- 结束 ---

        if mapped:
            logger.debug(f"📌 模型映射: '{original_model}' ➡️ '{new_model}'")
        else:
             # 如果没有发生映射且没有前缀，记录警告或决定默认值
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ 模型没有前缀或映射规则: '{original_model}'. 按原样使用。")
             new_model = v # 确保如果没有应用规则则返回原始值

        # 在值字典中存储原始模型
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    """令牌计数请求模型"""
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # 将存储原始模型名称
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info): # 重命名以避免冲突
        """令牌计数模型验证器 - 使用与 MessagesRequest 验证器相同的逻辑"""
        # 注意：Pydantic 验证器如果不是类方法可能不容易共享状态
        # 为了清晰起见在这里重新实现逻辑，可以重构
        original_model = v
        new_model = v # 默认为原始值

        logger.debug(f"📋 令牌计数验证: 原始='{original_model}', 首选='{PREFERRED_PROVIDER}', 大模型='{BIG_MODEL}', 小模型='{SMALL_MODEL}'")

        # 移除提供商前缀以便于匹配
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- 映射逻辑 --- 开始 ---
        mapped = False
        # 根据提供商偏好将 Haiku 映射到 SMALL_MODEL
        if 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # 根据提供商偏好将 Sonnet 映射到 BIG_MODEL
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # 如果未映射，为匹配已知列表的模型添加前缀
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # 技术上映射以添加前缀
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # 技术上映射以添加前缀
        # --- 映射逻辑 --- 结束 ---

        if mapped:
            logger.debug(f"📌 令牌计数映射: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ 令牌计数模型没有前缀或映射规则: '{original_model}'. 按原样使用。")
             new_model = v # 确保如果没有应用规则则返回原始值

        # 在值字典中存储原始模型
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    """令牌计数响应模型"""
    input_tokens: int

class Usage(BaseModel):
    """使用情况模型"""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    """消息响应模型"""
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """HTTP 中间件 - 记录请求详情"""
    # 获取请求详情
    method = request.method
    path = request.url.path
    
    # 仅在调试级别记录基本请求详情
    logger.debug(f"请求: {method} {path}")
    
    # 处理请求并获取响应
    response = await call_next(request)
    
    return response

# 不使用验证函数，因为我们使用环境 API 密钥

def parse_tool_result_content(content):
    """辅助函数：正确解析和规范化工具结果内容"""
    if content is None:
        return "未提供内容"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "无法解析的内容\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # 任何其他类型的回退
    try:
        return str(content)
    except:
        return "无法解析的内容"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """将 Anthropic API 请求格式转换为 LiteLLM 格式（遵循 OpenAI）"""
    # LiteLLM 在使用格式 model="anthropic/claude-3-opus-20240229" 时已经处理 Anthropic 模型
    # 所以我们只需要将我们的 Pydantic 模型转换为预期格式的字典
    
    messages = []
    
    # 如果存在系统消息则添加
    if anthropic_request.system:
        # 处理不同格式的系统消息
        if isinstance(anthropic_request.system, str):
            # 简单字符串格式
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # 内容块列表
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # 添加对话消息
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # 对用户消息中的 tool_result 进行特殊处理
            # OpenAI/LiteLLM 格式期望助手调用工具，
            # 用户的下一条消息将结果作为纯文本包含
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # 对于带有 tool_result 的用户消息，拆分为单独的消息
                text_content = ""
                
                # 提取所有文本部分并连接它们
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # 将工具结果作为单独的消息添加 - 模拟正常流程
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # 处理不同格式的工具结果内容
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # 如果内容是块列表，从每个块中提取文本
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # 通过尝试提取文本或转换为 JSON 来处理任何字典
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # 处理字典内容
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # 通过转换为字符串来处理任何其他类型
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "无法解析的内容"
                            
                            # 在 OpenAI 格式中，工具结果来自用户（而不是内容块）
                            text_content += f"工具结果 {tool_id}:\n{result_content}\n"
                
                # 作为包含所有内容的单个用户消息添加
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # 对其他消息类型的常规处理
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # 如果需要，处理工具使用块
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # 处理不同格式的工具结果内容
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # 正确处理内容字段
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # 如果是简单字符串，为其创建文本块
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # 如果已经是块列表，保持原样
                                    processed_content_block["content"] = block.content
                                else:
                                    # 默认回退
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # 默认空内容
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # 为 OpenAI 模型限制 max_tokens 到 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"为 OpenAI/Gemini 模型限制 max_tokens 到 16384（原始值: {anthropic_request.max_tokens}）")
    
    # 创建 LiteLLM 请求字典
    litellm_request = {
        "model": anthropic_request.model,  # 它理解 "anthropic/claude-x" 格式
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # 仅对 Anthropic 模型包含 thinking 字段
    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        litellm_request["thinking"] = anthropic_request.thinking

    # 如果存在可选参数则添加
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # 将工具转换为 OpenAI 格式
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # 如果是 pydantic 模型则转换为字典
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # 确保 tool_dict 是字典，如果 'tool' 不是类字典则处理潜在错误
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"无法将工具转换为字典: {tool}")
                     continue # 如果转换失败则跳过此工具

            # 如果目标是 Gemini 模型则清理模式
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"为 Gemini 工具清理模式: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # 创建 OpenAI 兼容的函数工具
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # 使用可能清理过的模式
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # 如果存在则将 tool_choice 转换为 OpenAI 格式
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # 处理 Anthropic 的 tool_choice 格式
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # 如果无法确定则默认为 auto
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """将 LiteLLM（OpenAI 格式）响应转换为 Anthropic API 响应格式"""
    
    # 增强的响应提取，具有更好的错误处理
    try:
        # 获取干净的模型名称以检查功能
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # 检查这是否是 Claude 模型（支持内容块）
        is_claude_model = clean_model.startswith("claude-")
        
        # 处理来自 LiteLLM 的 ModelResponse 对象
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # 直接从 ModelResponse 对象提取数据
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # 向后兼容 - 处理字典响应
            # 如果响应是字典则使用它，否则尝试转换为字典
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # 如果 .dict() 失败，尝试使用 model_dump 或 __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # 回退 - 手动提取属性
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # 从响应字典中提取内容
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # 为 Anthropic 格式创建内容列表
        content = []
        
        # 如果存在则添加文本内容块（文本可能为 None 或空，用于纯工具调用响应）
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # 如果存在则添加工具调用（Anthropic 格式中的 tool_use）- 仅适用于 Claude 模型
        if tool_calls and is_claude_model:
            logger.debug(f"处理工具调用: {tool_calls}")
            
            # 如果不是列表则转换为列表
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"处理工具调用 {idx}: {tool_call}")
                
                # 根据是字典还是对象提取函数数据
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # 如果需要则将字符串参数转换为字典
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"无法将工具参数解析为 JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"添加 tool_use 块: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_claude_model:
            # 对于非 Claude 模型，将工具调用转换为文本格式
            logger.debug(f"为非 Claude 模型将工具调用转换为文本: {clean_model}")
            
            # 我们将工具信息附加到文本内容
            tool_text = "\n\n工具使用:\n"
            
            # 如果不是列表则转换为列表
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # 根据是字典还是对象提取函数数据
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # 如果需要则将字符串参数转换为字典
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)
                
                tool_text += f"工具: {name}\n参数: {arguments_str}\n\n"
            
            # 将工具文本添加或附加到内容
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # 获取使用信息 - 从对象或字典中安全提取值
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # 将 OpenAI finish_reason 映射到 Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # 默认
        
        # 确保内容永不为空
        if not content:
            content.append({"type": "text", "text": ""})
        
        # 创建 Anthropic 风格的响应
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"转换响应时出错: {str(e)}\n\n完整回溯:\n{error_traceback}"
        logger.error(error_message)
        
        # 如果出现任何错误，创建回退响应
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"转换响应时出错: {str(e)}. 请检查服务器日志。"}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """处理来自 LiteLLM 的流式响应并转换为 Anthropic 格式"""
    try:
        # 发送 message_start 事件
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # 格式类似于 Anthropic 的 ID
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # 第一个文本块的内容块索引
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # 发送 ping 以保持连接活跃（Anthropic 这样做）
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # 跟踪累积的文本内容
        text_sent = False  # 跟踪是否已发送任何文本内容
        text_block_closed = False  # 跟踪文本块是否已关闭
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # 处理每个块
        async for chunk in response_generator:
            try:
                # 检查这是否是带有使用数据的响应结束
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # 处理文本内容
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # 从选择中获取增量
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # 如果没有增量，尝试获取消息
                        delta = getattr(choice, 'message', {})
                    
                    # 检查 finish_reason 以知道何时完成
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # 处理文本内容
                    delta_content = None
                    
                    # 处理不同格式的增量内容
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # 累积文本内容
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # 如果没有工具调用开始，总是发出文本增量
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # 处理工具调用
                    delta_tool_calls = None
                    
                    # 处理不同格式的工具调用
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # 如果有工具调用则处理
                    if delta_tool_calls:
                        # 我们看到的第一个工具调用 - 需要正确处理文本
                        if tool_index is None:
                            # 如果我们一直在流式传输文本，关闭该文本块
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # 如果我们累积了文本但没有发送，现在需要发出它
                            # 这处理第一个增量既有文本又有工具调用的情况
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # 发送累积的文本
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # 关闭文本块
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # 即使我们没有发送任何内容也关闭文本块 - 模型有时发出空文本块
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # 如果不是列表则转换为列表
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # 获取此工具调用的索引（用于多个工具）
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # 检查这是新工具还是继续
                            if tool_index is None or current_index != tool_index:
                                # 新工具调用 - 创建新的 tool_use 块
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # 提取函数信息
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # 开始新的 tool_use 块
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # 提取函数参数
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # 如果有参数，将它们作为增量发送
                            if arguments:
                                # 尝试检测参数是有效的 JSON 还是只是片段
                                try:
                                    # 如果已经是字典，使用它
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # 否则，尝试解析它
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # 如果是片段，将其视为字符串
                                    args_json = arguments
                                
                                # 添加到累积的工具内容
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # 发送更新
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # 处理 finish_reason - 结束流式响应
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # 关闭任何打开的工具调用块
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # 如果我们累积了文本但从未发送或关闭文本块，现在执行
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # 发送累积的文本
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # 关闭文本块
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # 将 OpenAI finish_reason 映射到 Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # 发送带有停止原因和使用的 message_delta
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # 发送 message_stop 事件
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # 发送最终的 [DONE] 标记以匹配 Anthropic 的行为
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # 记录错误但继续处理其他块
                logger.error(f"处理块时出错: {str(e)}")
                continue
        
        # 如果我们没有获得完成原因，关闭任何打开的块
        if not has_sent_stop_reason:
            # 关闭任何打开的工具调用块
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # 关闭文本内容块
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # 发送带有使用的最终 message_delta
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # 发送 message_stop 事件
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # 发送最终的 [DONE] 标记以匹配 Anthropic 的行为
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"流式处理中出错: {str(e)}\n\n完整回溯:\n{error_traceback}"
        logger.error(error_message)
        
        # 发送错误 message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # 发送 message_stop 事件
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # 发送最终的 [DONE] 标记
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    """主要的消息创建端点 - 处理 Anthropic API 请求并转换为其他提供商"""
    try:
        # 在这里打印请求体
        body = await raw_request.body()
    
        # 将原始请求体解析为 JSON，因为它是字节
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # 获取用于日志记录的显示名称，只是没有提供商前缀的模型名称
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # 清理模型名称以进行功能检查
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        logger.debug(f"📊 处理请求: 模型={request.model}, 流式={request.stream}")
        
        # 将 Anthropic 请求转换为 LiteLLM 格式
        litellm_request = convert_anthropic_to_litellm(request)
        
        # 根据模型确定使用哪个 API 密钥
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            # 如果配置了自定义 OpenAI 基础 URL 则使用
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL
                logger.debug(f"为模型使用 OpenAI API 密钥和自定义基础 URL {OPENAI_BASE_URL}: {request.model}")
            else:
                logger.debug(f"为模型使用 OpenAI API 密钥: {request.model}")
        elif request.model.startswith("gemini/"):
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug(f"为模型使用 Gemini API 密钥: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"为模型使用 Anthropic API 密钥: {request.model}")
        
        # 对于 OpenAI 模型 - 修改请求格式以处理限制
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"处理 OpenAI 模型请求: {litellm_request['model']}")
            
            # 对于 OpenAI 模型，我们需要将内容块转换为简单字符串
            # 并处理其他要求
            for i, msg in enumerate(litellm_request["messages"]):
                # 特殊情况 - 当内容是 tool_result 列表时直接处理消息内容
                # 这是我们在错误中看到的特定情况
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"发现只有 tool_result 内容的消息 - 需要特殊处理")
                        # 从所有 tool_result 块中提取内容
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "工具结果:\n"
                            result_content = block.get("content", [])
                            
                            # 处理不同格式的内容
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # 回退到任何字典的字符串表示
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # 用提取的文本替换列表
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"将 tool_result 转换为纯文本: {all_text.strip()[:200]}...")
                        continue  # 跳过此消息的正常处理
                
                # 1. 处理内容字段 - 正常情况
                if "content" in msg:
                    # 检查内容是否是列表（内容块）
                    if isinstance(msg["content"], list):
                        # 将复杂的内容块转换为简单字符串
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # 处理不同的内容块类型
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # 处理 tool_result 内容块 - 提取嵌套文本
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[工具结果 ID: {tool_id}]\n"
                                    
                                    # 从 tool_result 内容中提取文本
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # 通过尝试提取文本或转换为 JSON 来处理任何字典
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # 处理字典内容
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # 处理 tool_use 内容块
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[工具: {tool_name} (ID: {tool_id})]\n输入: {tool_input}\n\n"
                                
                                # 处理图像内容块
                                elif block.get("type") == "image":
                                    text_content += "[图像内容 - 在文本格式中不显示]\n"
                        
                        # 确保 OpenAI 模型的内容永不为空
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # 也检查 None 或空字符串内容
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # 不允许空内容
                
                # 2. 移除 OpenAI 在消息中不支持的字段
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"从消息中移除不支持的字段: {key}")
                        del msg[key]
            
            # 3. 最终验证 - 检查任何剩余的无效值并转储完整的消息详情
            for i, msg in enumerate(litellm_request["messages"]):
                # 记录消息格式以进行调试
                logger.debug(f"消息 {i} 格式检查 - 角色: {msg.get('role')}, 内容类型: {type(msg.get('content'))}")
                
                # 如果内容仍然是列表或 None，用占位符替换
                if isinstance(msg.get("content"), list):
                    logger.warning(f"关键: 消息 {i} 在处理后仍有列表内容: {json.dumps(msg.get('content'))}")
                    # 最后手段 - 将整个内容字符串化为 JSON
                    litellm_request["messages"][i]["content"] = f"内容作为 JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"消息 {i} 有 None 内容 - 用占位符替换")
                    litellm_request["messages"][i]["content"] = "..." # 回退占位符
        
        # 仅记录关于请求的基本信息，不是完整详情
        logger.debug(f"模型请求: {litellm_request.get('model')}, 流式: {litellm_request.get('stream', False)}")
        
        # 处理流式模式
        if request.stream:
            # 使用 LiteLLM 进行流式处理
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # 假设此时成功
            )
            # 确保我们使用异步版本进行流式处理
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # 使用 LiteLLM 进行常规完成
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # 假设此时成功
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ 收到响应: 模型={litellm_request.get('model')}, 时间={time.time() - start_time:.2f}s")
            
            # 将 LiteLLM 响应转换为 Anthropic 格式
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # 尽可能多地捕获关于错误的信息
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # 检查 LiteLLM 特定属性
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # 检查字典中的其他异常详情
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # 辅助函数安全地将对象序列化为 JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # 使用安全序列化记录所有错误详情
        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"处理请求时出错: {json.dumps(sanitized_details, indent=2)}")
        
        # 格式化响应错误
        error_message = f"错误: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\n消息: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\n响应: {error_details['response']}"
        
        # 返回详细错误
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    """令牌计数端点 - 计算请求的令牌数量"""
    try:
        # 记录传入的令牌计数请求
        original_model = request.original_model or request.model
        
        # 获取用于日志记录的显示名称，只是没有提供商前缀的模型名称
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # 清理模型名称以进行功能检查
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # 将消息转换为 LiteLLM 可以理解的格式
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # 令牌计数不使用的任意值
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # 使用 LiteLLM 的 token_counter 函数
        try:
            # 导入 token_counter 函数
            from litellm import token_counter
            
            # 美观地记录请求
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # 假设此时成功
            )
            
            # 准备令牌计数器参数
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }
            
            # 如果配置了 OpenAI 模型的自定义基础 URL 则添加
            if request.model.startswith("openai/") and OPENAI_BASE_URL:
                token_counter_args["api_base"] = OPENAI_BASE_URL
            
            # 计算令牌
            token_count = token_counter(**token_counter_args)
            
            # 返回 Anthropic 风格的响应
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("无法从 litellm 导入 token_counter")
            # 回退到简单近似
            return TokenCountResponse(input_tokens=1000)  # 默认回退
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"计算令牌时出错: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"计算令牌时出错: {str(e)}")

@app.get("/")
async def root():
    """根端点 - 返回服务信息"""
    return {"message": "Anthropic Proxy for LiteLLM"}

# 定义终端输出的 ANSI 颜色代码
class Colors:
    """终端颜色常量"""
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """以美观的、类似 Twitter 的格式记录请求，显示 Claude 到 OpenAI 的映射"""
    # 格式化 Claude 模型名称
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # 提取端点名称
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # 提取没有提供商前缀的 OpenAI 模型名称
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # 格式化工具和消息
    tools_str = f"{Colors.MAGENTA}{num_tools} 工具{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} 消息{Colors.RESET}"
    
    # 格式化状态代码
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # 将所有内容组合成清晰、美观的格式
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # 打印到控制台
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("运行方式: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # 配置 uvicorn 以最少的日志运行
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")

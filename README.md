# DeepThinking
在OpwenWebUI中支持类似DeepClaude的思维链和回复模型分离 - 仅支持0.5.6及以上版本 (双模型版本 - Think Model &amp; Base Model)

https://openwebui.com/f/timwhite/deepthinking/

```
"""
title: Deep Thinking
author: TimWhite
description: 在OpwenWebUI中支持类似DeepClaude的思维链和回复模型分离 - 仅支持0.5.6及以上版本 (双模型版本 - Think Model & Base Model)
version: 1.0.0
licence: MIT
"""
import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import asyncio

class Pipe:
    class Valves(BaseModel):
        """
        插件配置参数类 (Valves).
        包含 Think Model 和 Base Model 两组 API 参数配置。
        """
        think_model_DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1",
            description="Think Model - DeepSeek API的基础请求地址",
        )
        think_model_DEEPSEEK_API_KEY: str = Field(
            default="", description="Think Model - 用于身份验证的DeepSeek API密钥，可从控制台获取"
        )
        think_model_DEEPSEEK_API_MODEL: str = Field(
            default="deepseek-reasoner",
            description="Think Model - API请求的模型名称，默认为 deepseek-reasoner (用于生成思维链)",
        )
        base_model_DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Base Model - OpenAI API的基础请求地址, 也可以是其他OpenAI格式api",
        )
        base_model_DEEPSEEK_API_KEY: str = Field(
            default="", description="Base Model - 用于身份验证的API密钥，可从控制台获取"
        )
        base_model_DEEPSEEK_API_MODEL: str = Field(
            default="gpt-4o-mini",
            description="Base Model - API请求的模型名称，默认为 gpt-4o-mini (用于生成最终答案)",
        )

    def __init__(self):
        """
        Pipe 类的初始化方法.
        初始化配置参数, 数据前缀, 思考状态, 以及事件发射器.
        """
        self.valves = self.Valves()  # 初始化配置参数
        self.data_prefix = "data: "  # SSE 数据流前缀
        self.thinking = -1    # -1:未开始思考 0:思考中 1:已回答 (思考状态机)
        self.emitter = None     # 事件发射器 (用于 OpwenWebUI 插件框架)

    def pipes(self):
        """
        定义插件管道信息.
        返回插件支持的管道列表，这里使用 Base Model 的模型 ID 作为管道 ID 和名称.
        """
        return [
            {
                "id": self.valves.base_model_DEEPSEEK_API_MODEL,
                "name": self.valves.base_model_DEEPSEEK_API_MODEL,
            }
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        主处理管道函数 (核心逻辑).
        处理用户请求，依次调用 Think Model 获取思维链，再调用 Base Model 获取最终答案。
        """
        self.thinking = -1          # 重置思考状态
        self.emitter = __event_emitter__ # 设置事件发射器
        user_messages = body["messages"] # 保存用户的原始消息，后续传递给 Base Model

        # 验证 API 密钥配置 (Think Model 和 Base Model 密钥都需要配置)
        if not self.valves.think_model_DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置 Think Model API 密钥"}, ensure_ascii=False)
            return
        if not self.valves.base_model_DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置 Base Model API 密钥"}, ensure_ascii=False)
            return

        # --------------------- 步骤 1: 请求 Think Model 获取思维链 ---------------------
        think_model_headers = {
            "Authorization": f"Bearer {self.valves.think_model_DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        think_model_payload = {
            **body,
            "model": self.valves.think_model_DEEPSEEK_API_MODEL, # 使用 Think Model
        }

        think_content = ""  # 用于保存从 Think Model 获取的思维链内容

        try: # Think Model Request 的 try 代码块开始
            async with httpx.AsyncClient(http2=True) as client: # 使用 http2 优化连接
                async with client.stream( # 使用 stream 方法获取 SSE 数据流
                    "POST",
                    f"{self.valves.think_model_DEEPSEEK_API_BASE_URL}/chat/completions", # Think Model API Endpoint
                    json=think_model_payload,
                    headers=think_model_headers,
                    timeout=300, # 设置超时时间
                ) as response:
                    if response.status_code != 200: # 检查 HTTP 状态码，非 200 表示错误
                        error = await response.aread() # 读取错误响应内容
                        yield self._format_error(response.status_code, error) # 格式化错误信息并 yield
                        return

                    async for line in response.aiter_lines(): # 异步迭代 SSE 数据流的每一行
                        if not line.startswith(self.data_prefix): # 过滤非 data 行
                            continue
                        json_str = line[len(self.data_prefix) :] # 截取 JSON 字符串
                        try:
                            data = json.loads(json_str) # 解析 JSON 数据
                        except json.JSONDecodeError as e: # JSON 解析错误处理
                            error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        choice = data.get("choices", [{}])[0] # 获取 choices 列表中的第一个 choice
                        if choice.get("finish_reason"): # 检查是否完成 (Think Model 思考完成)
                            break # 结束 Think Model 的数据接收循环

                        state_output = await self._update_thinking_state( # 更新思考状态机
                            choice.get("delta", {}) # 提取 delta 信息用于状态更新
                        )
                        if state_output: # 如果状态发生变化，则 yield 状态标记
                            yield state_output
                            if state_output == "<think>": # 如果是开始思考状态，yield 换行符
                                yield "\n"

                        content = self._process_content(choice["delta"]) # 处理内容 (提取 reasoning_content 或 content)
                        if content: # 如果提取到内容
                            if content.startswith("<think>"): # 处理 thinking 开始标记
                                match = re.match(r"^<think>", content)
                                if match:
                                    content = re.sub(r"^<think>", "", content) # 移除标记
                                    yield "<think>" # yield 标记
                                    await asyncio.sleep(0.1) # 适当延时
                                    yield "\n" # yield 换行
                            elif content.startswith("</think>"): # 处理 thinking 结束标记
                                match = re.match(r"^</think>", content)
                                if match:
                                    content = re.sub(r"^</think>", "", content) # 移除标记
                                    yield "</think>" # yield 标记
                                    await asyncio.sleep(0.1) # 适当延时
                                    yield "\n" # yield 换行
                            think_content += content # 累加思维链内容
                            yield content #  <- 重要修改：这里仍然需要 yield 思维链内容，以便在 UI 上显示

        except Exception as e: # Think Model Request 的 try 代码块异常处理
            yield self._format_exception(e) # 格式化异常信息并 yield
            return
        finally:
            pass #  Think Model 请求完成后的清理代码 (当前为空)


        # --------------------- 步骤 2: 请求 Base Model 获取最终答案 ---------------------
        base_model_headers = {
            "Authorization": f"Bearer {self.valves.base_model_DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        # 将思维链内容添加到发送给 Base Model 的消息列表中
        base_model_messages = user_messages + [{"role": "assistant", "content": f"<think>\n{think_content}\n</think>"}] # 使用包含思维链的消息列表, 并添加标签

        base_model_payload = {
            **body,
            "model": self.valves.base_model_DEEPSEEK_API_MODEL, # 使用 Base Model
            "messages": base_model_messages # 使用包含思维链的消息列表
        }

        try: # Base Model Request 的 try 代码块开始
            async with httpx.AsyncClient(http2=True) as client: # 使用 http2 优化连接
                async with client.stream( # 使用 stream 方法获取 SSE 数据流
                    "POST",
                    f"{self.valves.base_model_DEEPSEEK_API_BASE_URL}/chat/completions", # Base Model API Endpoint
                    json=base_model_payload,
                    headers=base_model_headers,
                    timeout=300, # 设置超时时间
                ) as response:
                    if response.status_code != 200: # 检查 HTTP 状态码
                        error = await response.aread() # 读取错误响应
                        yield self._format_error(response.status_code, error) # 格式化错误并 yield
                        return

                    async for line in response.aiter_lines(): # 异步迭代 SSE 数据流
                        if not line.startswith(self.data_prefix): # 过滤非 data 行
                            continue
                        json_str = line[len(self.data_prefix) :] # 截取 JSON 字符串
                        try:
                            data = json.loads(json_str) # 解析 JSON 数据
                        except json.JSONDecodeError as e: # JSON 解析错误处理
                            error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        choice = data.get("choices", [{}])[0] # 获取 choices 列表中的第一个 choice
                        if choice.get("finish_reason"): # 检查是否完成 (Base Model 回答完成)
                            return # 结束 Base Model 的数据接收，整个 pipe 方法也结束

                        #  Base Model 响应中不再有 thinking 状态，因此这里只需要处理 content
                        content = self._process_content(choice["delta"]) # 提取 content
                        if content: # 如果提取到内容，则 yield 内容
                            yield content

        except Exception as e: # Base Model Request 的 try 代码块异常处理
            yield self._format_exception(e) # 格式化异常信息并 yield
            return
        finally:
            pass # Base Model 请求完成后的清理代码 (当前为空)


    async def _update_thinking_state(self, delta: dict) -> str:
        """
        更新思考状态机 (简化版).
        根据 delta 数据判断思考状态是否发生变化，并返回状态标记 (如 "<think>", "</think>").
        """
        state_output = ""
        # 状态转换：未开始 -> 思考中
        if self.thinking == -1 and delta.get("reasoning_content"):
            self.thinking = 0 # 设置为思考中状态
            state_output = "<think>" # 返回开始思考标记
        # 状态转换：思考中 -> 已回答
        elif (
            self.thinking == 0
            and not delta.get("reasoning_content") # reasoning_content 为空，表示思维链结束
            and delta.get("content") # content 不为空，表示开始返回最终答案
        ):
            self.thinking = 1 # 设置为已回答状态
            state_output = "\n</think>\n\n" # 返回结束思考标记和换行
        return state_output # 返回状态标记 (可能为空字符串，表示状态未变化)

    def _process_content(self, delta: dict) -> str:
        """
        处理内容.
        优先返回 reasoning_content (思维链内容)，如果 reasoning_content 为空，则返回 content (最终答案内容).
        """
        return delta.get("reasoning_content", "") or delta.get("content", "")

    def _format_error(self, status_code: int, error: bytes) -> str:
        """
        格式化错误信息.
        将 HTTP 状态码和错误内容格式化为 JSON 字符串返回.
        """
        try:
            err_msg = json.loads(error).get("message", error.decode(errors="ignore"))[
                :200 # 截取错误信息前 200 字符
            ]
        except: # 兼容 JSON 解析失败的情况
            err_msg = error.decode(errors="ignore")[:200] # 截取原始错误信息前 200 字符
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False # 返回 JSON 格式错误信息
        )

    def _format_exception(self, e: Exception) -> str:
        """
        格式化异常信息.
        将异常类型和异常信息格式化为 JSON 字符串返回.
        """
        err_type = type(e).__name__ # 获取异常类型名
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False) # 返回 JSON 格式异常信息
```

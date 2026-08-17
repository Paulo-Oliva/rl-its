from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    text: str
    data: dict = field(default_factory=dict)
    ok: bool = True


@dataclass
class ToolContext:
    problem: str = ""
    answer: str = ""
    transcript: list[dict] = field(default_factory=list)
    student_level: str = ""
    difficulty: str = ""


class Tool(ABC):
    name: str
    description: str
    parameters: dict = {"type": "object", "properties": {}}

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    @abstractmethod
    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        ...


# Test tool
class EchoTool(Tool):
    name = "call_echo"
    description = "echo a message back (test tool)"
    parameters = {
        "type": "object",
        "properties": {
            "msg": {
                "type": "string",
                "description": "text to echo"
            }
        },
        "required": ["msg"]
    }

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        msg = args.get("msg", "")
        return ToolResult(text=f"echo: {msg}", data={"echo": msg}, ok=bool(msg))

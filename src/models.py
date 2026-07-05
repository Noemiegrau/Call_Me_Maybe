from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """A single parameter in a function definition.

    Attributes:
        type: The parameter type (number, integer, string, boolean).
    """

    type: str


class FunctionReturn(BaseModel):
    """The return type of a function.

    Attributes:
        type: The return type (number, integer, string, boolean).
    """

    type: str


class FunctionDefinition(BaseModel):
    """A function that the LLM can call.

    Attributes:
        name: The function identifier (e.g. fn_add_numbers).
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their type definitions.
        returns: The return type of the function.
    """

    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionReturn


class Prompt(BaseModel):
    """A single natural language prompt from the input file.

    Attributes:
        prompt: The natural language request to process.
    """

    prompt: str


class FunctionCall(BaseModel):
    """A single result entry written to the output file.

    Attributes:
        prompt: The original natural language request.
        name: The name of the function to call.
        parameters: The arguments to pass to the function.
    """

    prompt: str
    name: str
    parameters: dict[str, float | int | str | bool]

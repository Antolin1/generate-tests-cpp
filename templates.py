import re

WIZARD_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nWrite {num} google tests for this function:\n{func}\n\n### Response:\nTEST({func_name},"
)

DS_TEMPLATE = """
{func}
// {num} tests
TEST({func_name},"""

DS_TEMPLATE_DOC = """
// {doc}
{func}
// {num} tests
TEST({func_name},"""

WIZARD_TEMPLATE_HUMANEVAL = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nCreate a Python script for this problem:\n{question}\n\n### Response:\n"
)


def postprocess_markdown(response):
    pattern = r'```python\s+(.*?)```'
    code_snippet = re.search(pattern, response, re.DOTALL).group(1)
    return code_snippet


def execute_code(code, test, entrypoint):
    check_program = code + "\n" + test + "\n" + f"check({entrypoint})"
    # print(check_program)
    try:
        exec(check_program, {})
        return "ok"
    except:
        return "exception"

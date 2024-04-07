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

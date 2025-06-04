import ast
import pathlib
import re


def get_split_into_sentences():
    path = pathlib.Path(__file__).resolve().parents[1] / "app.py"
    source = path.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "split_into_sentences":
            func_mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(func_mod)
            namespace = {"re": re}
            exec(compile(func_mod, filename=str(path), mode="exec"), namespace)
            return namespace["split_into_sentences"]
    raise ValueError("split_into_sentences not found")


def test_split_newlines():
    split_into_sentences = get_split_into_sentences()
    text = 'Hello world!\nHow are you?\n\n"I am fine." Yes!'
    expected = ['Hello world!', 'How are you?', 'I am fine.', 'Yes!']
    assert split_into_sentences(text) == expected


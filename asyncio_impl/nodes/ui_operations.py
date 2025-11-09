from pathlib import Path
import gradio as gr

def return_file(input_file):
    if input_file:
        print(Path(input_file).read_text())
        return (Path(__file__).parents[2] / "pyproject.toml").as_posix()
    else:
        return (Path(__file__).parents[2] / "LICENSE").as_posix()


def ui_processor():
    interface = gr.Interface(
        fn=return_file,
        inputs=gr.File(label="Upload Text file"),
        outputs=gr.File(label="Download file"),
    )

    interface.launch(allowed_paths=[str(Path(__file__).parents[2])])

ui_processor()
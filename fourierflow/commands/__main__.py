from typer import Typer

from fourierflow.commands import train, sample
from fourierflow.utils import setup_logger

setup_logger()

app = Typer()
app.add_typer(train.app, name="train")
app.add_typer(sample.app, name="sample")

if __name__ == "__main__":
    app()

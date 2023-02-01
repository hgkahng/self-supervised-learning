
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn
from rich.progress import TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn


def configure_progress_bar(console: Console = None,
                           transient: bool = True,
                           auto_refresh: bool = False,
                           disable: bool = False,
                           **kwargs) -> Progress:
    if console is None:
        console = Console(color_system='256', width=240)

    columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

    return Progress(
        *columns,
        console=console,
        auto_refresh=auto_refresh,
        transient=transient,
        disable=disable
    )

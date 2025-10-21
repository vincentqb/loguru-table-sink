import sys
from io import StringIO
from typing import Any, Dict, List, Optional


class TableSink:
    """A loguru sink that displays log records as an updating table."""

    LEVEL_COLORS = {
        "TRACE": "\033[36m",  # Cyan
        "DEBUG": "\033[34m",  # Blue
        "INFO": "\033[37m",  # White
        "SUCCESS": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31;1m",  # Red + Bold
    }
    RESET = "\033[0m"

    def __init__(
        self,
        key_column: str = "epoch",
        float_precision: int = 4,
        max_rows: Optional[int] = None,
        colorize: bool = True,
    ):
        """
        Initialize the table sink.

        Args:
            key_column: The column name to use as the primary key/identifier
            float_precision: Number of decimal places for float values
            max_rows: Maximum number of rows to display (None = unlimited)
            colorize: Whether to colorize rows by log level
        """
        self.key_column = key_column
        self.float_precision = float_precision
        self.max_rows = max_rows
        self.colorize = colorize

        self.columns: List[str] = []
        self.rows: Dict[Any, Dict[str, Any]] = {}
        self.row_levels: Dict[Any, str] = {}  # Track level for each row
        self.row_order: List[Any] = []
        self.header_printed = False
        self.last_line_count = 0
        self.finished = False

    def _format_value(self, value: Any) -> str:
        """Format a value for display in the table."""
        if value is None or value == "":
            return ""
        elif isinstance(value, float):
            return f"{value:.{self.float_precision}f}"
        else:
            return str(value)

    def _update_columns(self, data: Dict[str, Any]):
        """Update the column list based on new data."""
        # Ensure key column is first
        if self.key_column not in self.columns and self.key_column in data:
            self.columns.insert(0, self.key_column)

        # Add any new columns
        for key in data.keys():
            if key not in self.columns:
                self.columns.append(key)

    def _calculate_column_widths(self) -> Dict[str, int]:
        """Calculate the width needed for each column."""
        widths = {}

        for col in self.columns:
            # Start with header width
            width = len(col)

            # Check all row values
            for row_data in self.rows.values():
                if col in row_data:
                    value_width = len(self._format_value(row_data[col]))
                    width = max(width, value_width)

            # Add padding
            widths[col] = width + 2

        return widths

    def _build_separator(self, widths: Dict[str, int]) -> str:
        """Build a separator line."""
        parts = []
        for col in self.columns:
            parts.append("─" * widths[col])
        return "─".join(parts)

    def _build_row(self, data: Dict[str, Any], widths: Dict[str, int], color_code: str = "") -> str:
        """Build a single row of the table."""
        parts = []
        for col in self.columns:
            value = self._format_value(data.get(col, ""))
            parts.append(value.rjust(widths[col] - 1) + " ")

        row_content = " " + " ".join(parts) + " "

        if color_code and self.colorize:
            return color_code + row_content + self.RESET
        return row_content

    def _clear_lines(self, n: int):
        """Clear n lines from the terminal."""
        if n > 0:
            # Move cursor up n lines and clear them
            sys.stdout.write(f"\033[{n}A")
            sys.stdout.write("\033[J")

    def _render_table(self, final: bool = False):
        """Render the entire table."""
        output = StringIO()
        widths = self._calculate_column_widths()

        # Top border
        output.write(self._build_separator(widths) + "\n")

        # Header
        header_data = {col: col for col in self.columns}
        output.write(self._build_row(header_data, widths) + "\n")

        # Header separator
        output.write(self._build_separator(widths) + "\n")

        # Data rows with colors
        for idx, key in enumerate(self.row_order):
            row_data = self.rows[key]
            level = self.row_levels.get(key, "INFO")
            color_code = self.LEVEL_COLORS.get(level, "")
            output.write(self._build_row(row_data, widths, color_code) + "\n")

        # Bottom border if final
        if final:
            output.write(self._build_separator(widths) + "\n")

        table_str = output.getvalue()

        # Clear previous table
        self._clear_lines(self.last_line_count)

        # Write new table
        sys.stdout.write(table_str)
        sys.stdout.flush()

        # Update line count
        self.last_line_count = table_str.count("\n")

    def __call__(self, message):
        """
        Loguru sink function.

        Expected usage:
            logger.bind(epoch=0, train_loss=1.5541, train_accuracy=0.3393).info("Training")
        """
        if self.finished:
            return

        # Extract extra fields from the record
        record = message.record
        extra_data = record["extra"].copy()

        # Skip if no extra data
        if not extra_data:
            return

        # Get the key value
        if self.key_column not in extra_data:
            return

        key_value = extra_data[self.key_column]
        level_name = record["level"].name

        # Update columns
        self._update_columns(extra_data)

        # Update or add row
        if key_value not in self.rows:
            self.row_order.append(key_value)

            # Enforce max_rows limit
            if self.max_rows and len(self.row_order) > self.max_rows:
                oldest_key = self.row_order.pop(0)
                del self.rows[oldest_key]
                del self.row_levels[oldest_key]

        self.rows[key_value] = extra_data
        self.row_levels[key_value] = level_name

        self._render_table()

    def finish(self):
        """Finish the table by adding a bottom border."""
        if not self.finished and self.rows:
            self.finished = True
            self._render_table(final=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finish the table."""
        self.finish()
        return False


if __name__ == "__main__":
    import time

    from loguru import logger

    # Remove default handler
    logger.remove()

    # Create table sink as context manager
    table_sink = TableSink(key_column="epoch", float_precision=4, max_rows=6)
    logger.add(table_sink)

    with table_sink:

        print("Training a simple linear model on the Iris dataset.")

        for epoch in range(5):
            train_loss = 1.5 / (epoch + 1) + 0.05
            train_acc = 0.3 + epoch * 0.08

            logger.bind(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc).info("training")

            time.sleep(0.3)

        logger.bind(
            epoch=4, train_loss=0.6019, train_accuracy=0.6964, valid_loss=0.8493, valid_accuracy=0.6842
        ).success("training")
        time.sleep(0.3)

        for epoch in range(5, 10):
            train_loss = 1.5 / (epoch + 1)
            train_acc = 0.3 + epoch * 0.08

            if epoch == 6:
                logger.bind(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc).warning("training")
            elif epoch == 8:
                logger.bind(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc).error("training")
            else:
                logger.bind(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc).info("training")

            if epoch == 7:
                for i in range(10):
                    logger.bind(epoch=epoch, train_loss=train_loss + i, train_accuracy=train_acc).debug("training")
                    time.sleep(0.3)

            time.sleep(0.3)

    print("Training complete!")

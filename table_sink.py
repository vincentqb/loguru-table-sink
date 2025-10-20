import sys
from io import StringIO
from typing import Any, Dict, List, Optional


class TableSink:
    """A loguru sink that displays log records as an updating table."""

    def __init__(
        self,
        key_column: str = "epoch",
        divider_interval: Optional[int] = None,
        float_precision: int = 4,
        max_rows: Optional[int] = None,
    ):
        """
        Initialize the table sink.

        Args:
            key_column: The column name to use as the primary key/identifier
            divider_interval: Add a divider row every N rows (None = no dividers)
            float_precision: Number of decimal places for float values
            max_rows: Maximum number of rows to display (None = unlimited)
        """
        self.key_column = key_column
        self.divider_interval = divider_interval
        self.float_precision = float_precision
        self.max_rows = max_rows

        self.columns: List[str] = []
        self.rows: Dict[Any, Dict[str, Any]] = {}
        self.row_order: List[Any] = []
        self.header_printed = False
        self.last_line_count = 0

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

    def _build_separator(self, widths: Dict[str, int], top: bool = False) -> str:
        """Build a separator line."""
        parts = []
        for col in self.columns:
            parts.append("─" * widths[col])

        if top:
            return "╭" + "┬".join(parts) + "╮"
        else:
            return "├" + "┼".join(parts) + "┤"

    def _build_row(self, data: Dict[str, Any], widths: Dict[str, int]) -> str:
        """Build a single row of the table."""
        parts = []
        for col in self.columns:
            value = self._format_value(data.get(col, ""))
            parts.append(value.center(widths[col]))

        return "│" + "│".join(parts) + "│"

    def _clear_lines(self, n: int):
        """Clear n lines from the terminal."""
        if n > 0:
            # Move cursor up n lines and clear them
            sys.stdout.write(f"\033[{n}A")
            sys.stdout.write("\033[J")

    def _render_table(self):
        """Render the entire table."""
        output = StringIO()
        widths = self._calculate_column_widths()

        # Top border
        output.write(self._build_separator(widths, top=True) + "\n")

        # Header
        header_data = {col: col for col in self.columns}
        output.write(self._build_row(header_data, widths) + "\n")

        # Header separator
        output.write(self._build_separator(widths) + "\n")

        # Data rows
        for idx, key in enumerate(self.row_order):
            row_data = self.rows[key]
            output.write(self._build_row(row_data, widths) + "\n")

            # Add divider if needed
            if self.divider_interval and (idx + 1) % self.divider_interval == 0 and idx < len(self.row_order) - 1:
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

        # Update columns
        self._update_columns(extra_data)

        # Update or add row
        if key_value not in self.rows:
            self.row_order.append(key_value)

            # Enforce max_rows limit
            if self.max_rows and len(self.row_order) > self.max_rows:
                # Remove the oldest row
                oldest_key = self.row_order.pop(0)
                del self.rows[oldest_key]

        self.rows[key_value] = extra_data

        # Render the table
        self._render_table()


# Example usage
if __name__ == "__main__":
    import time

    from loguru import logger

    # Remove default handler
    logger.remove()

    # Add table sink
    table_sink = TableSink(key_column="epoch", divider_interval=5, float_precision=4, max_rows=4)
    logger.add(table_sink)

    # Simulate training
    print("Training a simple linear model on the Iris dataset.")

    # First few epochs without validation
    for epoch in range(5):
        train_loss = 1.5 / (epoch + 1) + 0.05
        train_acc = 0.3 + epoch * 0.08

        logger.bind(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc).info("training")

        time.sleep(0.3)

    # Add validation metrics
    logger.bind(epoch=4, train_loss=0.6019, train_accuracy=0.6964, valid_loss=0.8493, valid_accuracy=0.6842).info(
        "training"
    )
    time.sleep(0.3)

    # Continue with more epochs
    for epoch in range(5, 10):
        train_loss = 1.5 / (epoch + 1)
        train_acc = 0.3 + epoch * 0.08

        logger.bind(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc).info("training")
        if epoch == 7:
            for i in range(10):
                logger.bind(epoch=epoch, train_loss=train_loss + i, train_accuracy=train_acc).info("training")
                time.sleep(0.3)

        time.sleep(0.3)

    print("\n")  # Final newline after table

import collections
import numpy as np
from pathlib import Path
import typing


_EPS = 1e-9


class TypstTable:
    def __init__(self, columns: list[tuple[str, str]]):
        self.columns = columns
        self.rows = []

    def add_row(self, row: collections.namedtuple):
        self.rows.append(row)
        return self

    def get_markup(self, align: str = 'center'):
        col_cnt = len(self.columns)
        lines = [
            f'#table(columns: {col_cnt},',
            f'  align: ({", ".join([align] * col_cnt)}),',
            f'  table.header({", ".join(map(lambda x: TypstTable._format_value(x[1]), self.columns))}),',
        ]        
        for row in self.rows:
            lines.extend(self._get_markup_row(row))
        lines.append(')')
        return '\n'.join(lines)

    def dump(self, filename: Path, align: str = 'center'):
        filename.parents[0].mkdir(parents=True, exist_ok=True)
        filename.write_text(self.get_markup(align=align))

    def _get_markup_row(self, row: collections.namedtuple) -> list[str]:
        lines = []
        for key, _ in self.columns:
            value = getattr(row, key, f'<Unknown key: {key}>')
            lines.append(f'  {TypstTable._format_value(value)},')
        return lines

    @staticmethod
    def _format_value(value: object) -> str:
        if TypstTable._is_integer(value):
            return f'[${int(value)}$]'
 
        if TypstTable._is_float(value):
            return f'[{TypstTable._get_formatted_float(float(value))}]'
        
        return f'[{value.strip()}]'

    @staticmethod
    def _get_formatted_float(value: float) -> str:
        if not np.isfinite(value):
            return str(value)

        if abs(value) < _EPS:
            return '0'

        exp = int(np.floor(np.log10(abs(value))))
        coef = value / (10.0 ** exp)
        return f'${coef:.2f} times 10^({exp})$'

    @staticmethod
    def _is_integer(value: object) -> bool:
        return (
            isinstance(value, (int, np.integer)) or
            (TypstTable._is_float(value) and np.isfinite(value) and float(value).is_integer())
        )

    @staticmethod
    def _is_float(value: object) -> bool:
        return isinstance(value, (float, np.floating))


def get_params_from_generator(generator: typing.Callable) -> list[str]:
    return list(map(lambda x: x[0], generator()))

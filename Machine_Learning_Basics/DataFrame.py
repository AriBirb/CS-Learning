


class Matrix:
    def __init__(self, data: list[list[float]]):
        if any({len(data[i]) != len(data[i+1]) for i in range(len(data)-1)}):
            raise ValueError("Each row must have the same length.")
        if len(data) == 0:
            raise ValueError("There must be at least one row.")
        if len(data[0]) == 0:
            raise ValueError("There must be at least one column.")
        self.num_rows = len(data)
        self.num_columns = len(data[0])
        self.data = data

    def __str__(self):
        longest = 0
        for row in self.data:
            for value in row:
                if len(str(value)) > longest:
                    longest = len(str(value))

        if self.num_rows == 1:
            row = '['
            for i in range(self.num_columns-1):
                row += str(self.data[0][i]).center(longest) + ' '
            row += str(self.data[0][self.num_columns-1]).center(longest) + ']'
            return(row)
        else:
            row = "⎡"
            for i in range(self.num_columns - 1):
                row += str(self.data[0][i]).center(longest) + ' '
            row += str(self.data[0][self.num_columns - 1]).center(longest) + '⎤'

            row += '\n'

            for i in range(1, self.num_rows-1):
                row += "⎢"
                for j in range(self.num_columns - 1):
                    row += str(self.data[i][j]).center(longest) + ' '
                row += str(self.data[i][self.num_columns - 1]).center(longest) + '⎥'
                row += '\n'

            row += "⎣"
            for i in range(self.num_columns - 1):
                row += str(self.data[self.num_rows-1][i]).center(longest) + ' '
            row += str(self.data[self.num_rows-1][self.num_columns - 1]).center(longest) + '⎦'
            return row

    def __mul__(self, other):
        if self.num_columns != other.num_rows:
            raise ValueError("Invalid Dimensions for Multiplication: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ") * (" +
                             str(other.num_rows) + " x " + str(other.num_columns) + ")")

        result = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.num_columns))
                for j in range(other.num_columns)
            ]
            for i in range(self.num_rows)
        ]

        return Matrix(result)

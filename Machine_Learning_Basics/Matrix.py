import math

class Matrix:
    '''Class for all the basic functions of matrices'''
    def __init__(self, data: list[list[float]]):
        if any({len(data[i]) != len(data[i+1]) for i in range(len(data)-1)}):
            raise ValueError("Each row must have the same length.")
        if len(data) == 0:
            raise ValueError("There must be at least one row.")
        if len(data[0]) == 0:
            raise ValueError("There must be at least one column.")
        self.num_rows = len(data)
        self.num_columns = len(data[0])
        self.data = [row[:] for row in data]

    def __str__(self):
        '''Return string representation of matrix data.'''
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

    def __eq__(self, other):
        '''Return True if Matricies have equal data and False otherwise'''
        if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
            return False
        return all(all(self.data[i][j] == other.data[i][j]
                       for j in range(self.num_columns))
                   for i in range(self.num_rows))

    def __mul__(self, other):
        '''Return product of two matricies if their dimensions are appropriate.'''
        if self.num_columns != other.num_rows:
            raise ValueError("Invalid Dimensions for Multiplication: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ") * (" +
                             str(other.num_rows) + " x " + str(other.num_columns) + ")")

        result = [[
            sum(self.data[i][k] * other.data[k][j] for k in range(self.num_columns))
            for j in range(other.num_columns)]
            for i in range(self.num_rows)]

        return Matrix(result)

    def __add__(self, other):
        '''Return sum of two matricies if their dimensions are the same.'''
        if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
            raise ValueError("Invalid Dimensions for Addition: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ") + (" +
                             str(other.num_rows) + " x " + str(other.num_columns) + ")")
        result = [[
            self.data[i][j] + other.data[i][j]
            for j in range(self.num_columns)]
            for i in range(self.num_rows)]

        return Matrix(result)

    def __sub__(self, other):
        '''Return difference of two matricies if their dimensions are the same.'''
        if self.num_rows != other.num_rows or self.num_columns != other.num_columns:
            raise ValueError("Invalid Dimensions for Addition: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ") + (" +
                             str(other.num_rows) + " x " + str(other.num_columns) + ")")
        result = [[
            self.data[i][j] - other.data[i][j]
            for j in range(self.num_columns)]
            for i in range(self.num_rows)]

        return Matrix(result)

    def identity(size: int):
        '''Creates an identity matrix of size (nxn).'''
        data = [[1 if j == i else 0 for j in range(size)] for i in range(size)]
        return Matrix(data)

    def zeros(rows: int, columns: int):
        '''Creates a zeros matrix of size (nxm).'''
        data = [[0 for _ in range(columns)] for _ in range(rows)]
        return Matrix(data)

    def transpose(self):
        '''Returns the transpose of the matrix'''
        result = [[self.data[j][i]
                   for j in range(self.num_rows)]
                  for i in range(self.num_columns)]
        return Matrix(result)

    def transposed(self):
        '''Mutates the matrix to be its transpose'''
        self.data = self.transpose().data
        self.num_rows = self.transpose().num_rows
        self.num_columns = self.transpose().num_columns

    def LU_decomposition(self):
        '''Returns the LU decompositoon in the form [L, U, perm_vector, sign].
        NOTE: decomposition will continue even if the matrix is singular, U will have a 0 in the diagonal'''
        if self.num_rows != self.num_columns:
            raise ValueError("Invalid Dimensions for LU decomposition: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ")")
        U = Matrix(self.data)
        L = Matrix.identity(self.num_rows)
        permutations = list(range(self.num_rows))
        sign = 1
        for k in range(U.num_columns):
            index_max = k
            value_max = abs(U.data[k][k])
            for i in range(k+1, U.num_rows):
                if abs(U.data[i][k]) > value_max:
                    value_max = abs(U.data[i][k])
                    index_max = i

            if value_max == 0:
                continue
            if index_max != k:
                U.data[k], U.data[index_max] = U.data[index_max], U.data[k]
                for j in range(k):
                    L.data[k][j], L.data[index_max][j] = L.data[index_max][j], L.data[k][j]
                permutations[k], permutations[index_max] = permutations[index_max], permutations[k]
                sign *= -1

            for i in range(k+1, U.num_rows):
                L.data[i][k] = U.data[i][k] / U.data[k][k]

            for i in range(k+1, U.num_rows):
                U.data[i] = [U.data[i][j] - U.data[k][j] * L.data[i][k]
                                  for j in range(U.num_columns)]

        return [L, U, permutations, sign]

    def determinant(self):
        '''Returns the determinant of the matrix.'''
        if self.num_rows != self.num_columns:
            raise ValueError("Invalid Dimensions for LU decomposition: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ")")
        L, U, permutations, sign = self.LU_decomposition()
        if any(U.data[i][i] == 0 for i in range(U.num_rows)):
            return 0
        det = sign * math.prod(U.data[i][i] for i in range(U.num_rows))
        return det

    def inverse(self):
        if self.num_rows != self.num_columns:
            raise ValueError("Invalid Dimensions for Inversion: (" +
                             str(self.num_rows) + " x " + str(self.num_columns) + ")")
        L, U, permutations, sign = self.LU_decomposition()
        if any(U.data[i][i] == 0 for i in range(U.num_rows)):
            raise ValueError("Matrix is singular, inverse does not exist")
        b = []

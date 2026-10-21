"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros((matrices.shape[1], 1), dtype=float)
    for i in range(matrices.shape[0]):
        result += matrices[i] @ vectors[i]
    return result


def test_sum_prod():
    mats = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    vecs = np.array([[[1], [1]], [[1], [1]]])               
    res = sum_prod(MatrixVectorBatchInput(matrices=mats, vectors=vecs))
    assert np.allclose(res, [[3], [3]])
    assert res.shape == (2, 1)

    mats1 = np.array([[[1, 2], [3, 4]]])                    
    vecs1 = np.array([[[1], [0]]])                          
    res1 = sum_prod(MatrixVectorBatchInput(matrices=mats1, vectors=vecs1))
    assert np.allclose(res1, [[1], [3]])

    mats3 = np.array([[[1, 0], [0, 1]]] * 3)                
    vecs3 = np.array([[[1], [2]]] * 3)                      
    res3 = sum_prod(MatrixVectorBatchInput(matrices=mats3, vectors=vecs3))
    assert np.allclose(res3, [[3], [6]])
    print("1 accept")


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(float)


def test_binarize():
    m = np.array([[1, 2], [3, 4]])
    out = binarize(BinarizeInput(matrix=m, threshold=2))
    assert np.array_equal(out, [[0, 0], [1, 1]])
    assert np.array_equal(m, [[1, 2], [3, 4]]) 

    m2 = np.array([[5, 6], [7, 8]])
    out2 = binarize(BinarizeInput(matrix=m2, threshold=0))
    assert np.array_equal(out2, np.ones_like(m2, dtype=float))

    out3 = binarize(BinarizeInput(matrix=m2, threshold=100))
    assert np.array_equal(out3, np.zeros_like(m2, dtype=float))

    m4 = np.array([[2, 2], [2, 3]])
    out4 = binarize(BinarizeInput(matrix=m4, threshold=2))
    assert np.array_equal(out4, [[0, 0], [0, 1]])
    print("2 accept")


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix]


def test_unique_rows():
    m = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_rows(MatrixInput(matrix=m)) == [[1, 2], [3]]

    m2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert unique_rows(MatrixInput(matrix=m2)) == [[1, 2, 3, 4], [5, 6, 7, 8]]

    m3 = np.array([[2, 1, 2, 1]])
    assert unique_rows(MatrixInput(matrix=m3)) == [[1, 2]]
    print("3 accept")


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(matrix[:, j])) for j in range(matrix.shape[1])]


def test_unique_columns():
    m = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_columns(MatrixInput(matrix=m)) == [[1, 3], [1, 3], [2, 3]]

    m2 = np.array([[1, 2], [1, 2], [1, 2]])
    assert unique_columns(MatrixInput(matrix=m2)) == [[1], [2]]

    m3 = np.array([[5, 3], [5, 3], [5, 3]])
    assert unique_columns(MatrixInput(matrix=m3)) == [[5], [3]]
    print("4 accept")


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def test_matrix_statistics():
    out = matrix_statistics(RandomMatrixInput(
        rows=4, columns=5, mean=0.0, std=1.0, seed=42,
    ))
    assert out.matrix.shape == (4, 5)
    assert out.row_means.shape == (4,)
    assert out.column_means.shape == (5,)
    assert out.row_variances.shape == (4,)
    assert out.column_variances.shape == (5,)

    assert abs(np.mean(out.matrix) - 0.0) < 1.0

    out2 = matrix_statistics(RandomMatrixInput(
        rows=4, columns=5, mean=0.0, std=1.0, seed=42,
    ))
    assert np.allclose(out.matrix, out2.matrix)

    out3 = matrix_statistics(RandomMatrixInput(
        rows=4, columns=5, mean=0.0, std=1.0, seed=43,
    ))
    assert not np.allclose(out.matrix, out3.matrix)

    assert np.allclose(out.row_means, np.mean(out.matrix, axis=1))
    assert np.allclose(out.column_variances, np.var(out.matrix, axis=0))
    print("5 accept")
   

def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    grid = np.zeros((rows, columns), dtype=type(first))
    for i in range(rows):
        for j in range(columns):
            grid[i, j] = first if (i + j) % 2 == 0 else second
    return grid


def test_chess():
    out = chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(out, expected)

    out1 = chess(ChessInput(rows=1, columns=1, first=5, second=9))
    assert np.array_equal(out1, [[5]])

    out2 = chess(ChessInput(rows=1, columns=3, first=1, second=2))
    assert np.array_equal(out2, [[1, 2, 1]])

    out3 = chess(ChessInput(rows=3, columns=1, first=1, second=2))
    assert np.array_equal(out3, [[1], [2], [1]])

    out4 = chess(ChessInput(rows=4, columns=4, first=7, second=8))
    assert out4[0, 0] == 7
    assert out4[0, 1] == 8
    assert out4[1, 0] == 8
    print("6 accept")


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    cx, cy = image_width // 2, image_height // 2
    x0 = cx - width // 2
    y0 = cy - height // 2
    x1 = x0 + width
    y1 = y0 + height

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_width)
    y1 = min(y1, image_height)

    image[y0:y1, x0:x1] = shape_color
    return image


def test_draw_rectangle():
    out = draw_rectangle(RectangleInput(
        width=2, height=2,
        image_height=5, image_width=5,
        shape_color=(255, 0, 0), background_color=(0, 0, 0),
    ))
    assert out.shape == (5, 5, 3)
    assert np.array_equal(out[0, 0], [0, 0, 0])
    assert np.array_equal(out[2, 2], [255, 0, 0])
    assert np.array_equal(out[0, :], np.zeros((5, 3), dtype=np.uint8))

    out2 = draw_rectangle(RectangleInput(
        width=10, height=10,
        image_height=3, image_width=3,
        shape_color=(1, 2, 3), background_color=(9, 9, 9),
    ))
    assert np.array_equal(out2, np.full((3, 3, 3), [1, 2, 3], dtype=np.uint8))
    print("7 accept")

   

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = background_color

    x0, y0 = image_width / 2.0, image_height / 2.0

    yy, xx = np.mgrid[0:image_height, 0:image_width]
    mask = ((xx - x0) ** 2) / (semi_axis_x ** 2) + ((yy - y0) ** 2) / (semi_axis_y ** 2) <= 1.0

    image[mask] = shape_color
    return image


def test_draw_ellipse():
    out = draw_ellipse(EllipseInput(
        semi_axis_x=2, semi_axis_y=2,
        image_height=7, image_width=7,
        shape_color=(255, 0, 0), background_color=(0, 0, 0),
    ))
    assert out.shape == (7, 7, 3)
    assert np.array_equal(out[3, 3], [255, 0, 0])
    assert np.array_equal(out[0, 0], [0, 0, 0])

    out2 = draw_ellipse(EllipseInput(
        semi_axis_x=1, semi_axis_y=1,
        image_height=5, image_width=5,
        shape_color=(1, 1, 1), background_color=(2, 2, 2),
    ))
    assert np.array_equal(out2[2, 2], [1, 1, 1])
    assert np.array_equal(out2[0, 0], [2, 2, 2])

    out3 = draw_ellipse(EllipseInput(
        semi_axis_x=100, semi_axis_y=100,
        image_height=3, image_width=3,
        shape_color=(5, 5, 5), background_color=(0, 0, 0),
    ))
    assert np.all(out3 == [5, 5, 5])
    print("8 accept")


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    
    values = np.asarray(values, dtype=float)
    window = int(window)

    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))

    maxima = []
    minima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append(i)

    n = len(values)
    moving = np.array([
        np.mean(values[i:i + window]) for i in range(n - window + 1)
    ])

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving,     
    )


def test_analyze_time_series():
    vals = [1, 3, 2, 5, 4]
    out = analyze_time_series(TimeSeriesInput(values=vals, window=2))
    assert abs(out.mean - np.mean(vals)) < 1e-9
    assert abs(out.variance - np.var(vals)) < 1e-9
    assert abs(out.std - np.std(vals)) < 1e-9
    assert out.local_maxima_indices == [1, 3]
    assert out.local_minima_indices == [2]
    assert np.allclose(out.moving_average, [2.0, 2.5, 3.5, 4.5])

    out2 = analyze_time_series(TimeSeriesInput(values=[1, 2, 3], window=3))
    assert np.allclose(out2.moving_average, [2.0])
    assert out2.local_maxima_indices == []
    assert out2.local_minima_indices == []

    out3 = analyze_time_series(TimeSeriesInput(values=[2, 2, 2, 2], window=2))
    assert out3.local_maxima_indices == []
    assert out3.local_minima_indices == []

    out4 = analyze_time_series(TimeSeriesInput(values=[1, 2, 3, 4], window=2))
    assert out4.local_maxima_indices == []
    assert out4.local_minima_indices == []

    out5 = analyze_time_series(TimeSeriesInput(values=[1, 5, 3], window=1))
    assert np.allclose(out5.moving_average, [1, 5, 3])
    print("9 accept")


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)

    if not class_count:
        class_count = int(labels.max()) + 1

    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result


def test_one_hot():
    out = one_hot(OneHotInput(labels=[0, 2, 3, 0], class_count=None))
    expected = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    assert np.array_equal(out, expected)

    out2 = one_hot(OneHotInput(labels=[0, 1], class_count=4))
    assert out2.shape == (2, 4)
    assert np.array_equal(out2, [[1, 0, 0, 0], [0, 1, 0, 0]])

    out3 = one_hot(OneHotInput(labels=[0, 1, 2], class_count=3))
    assert out3.shape == (3, 3)

    out4 = one_hot(OneHotInput(labels=[0], class_count=None))
    assert np.array_equal(out4, [[1]])

    out5 = one_hot(OneHotInput(labels=[2, 2, 2], class_count=None))
    assert out5.shape == (3, 3)
    assert np.array_equal(out5, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    print("10 accept")


if __name__ == "__main__":
    test_sum_prod()
    test_binarize()
    test_unique_rows()
    test_unique_columns()
    test_matrix_statistics()
    test_chess()
    test_draw_rectangle()
    test_draw_ellipse()
    test_analyze_time_series()
    test_one_hot()

"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics
)

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum([np.dot(m, v) for m, v in zip(matrices, vectors)], axis=0)

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)

def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [sorted(list(set(row))) for row in matrix]

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [sorted(list(set(matrix[:, i]))) for i in range(matrix.shape[1])]

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    m, n, mean, std, seed = data.m, data.n, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(m, n))
    
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)
    row_vars = np.var(matrix, axis=1)
    col_vars = np.var(matrix, axis=0)
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means.tolist(),
        col_means=col_means.tolist(),
        row_vars=row_vars.tolist(),
        col_vars=col_vars.tolist()
    )

def chess(data: ChessInput) -> np.ndarray:
    m, n, a, b = data.m, data.n, data.a, data.b
    matrix = np.zeros((m, n))
    coords = np.indices((m, n))
    matrix[(coords[0] + coords[1]) % 2 == 0] = a
    matrix[(coords[0] + coords[1]) % 2 != 0] = b
    return matrix

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    m, n, width, height = data.m, data.n, data.width, data.height
    bg_color, rect_color = data.bg_color, data.rect_color
    
    image = np.full((m, n, 3), bg_color, dtype=np.uint8)
    
    center_y, center_x = m // 2, n // 2
    top = max(0, center_y - height // 2)
    bottom = min(m, center_y + (height + 1) // 2)
    left = max(0, center_x - width // 2)
    right = min(n, center_x + (width + 1) // 2)
    
    image[top:bottom, left:right] = rect_color
    return image

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    m, n, a, b = data.m, data.n, data.a, data.b
    bg_color, ellipse_color = data.bg_color, data.ellipse_color
    
    image = np.full((m, n, 3), bg_color, dtype=np.uint8)
    center_y, center_x = m // 2, n // 2
    
    y, x = np.ogrid[:m, :n]
    
    mask = ((x - center_x) ** 2) / (a ** 2) + ((y - center_y) ** 2) / (b ** 2) <= 1
    image[mask] = ellipse_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    series, p = np.array(data.series), data.p
    
    mean_val = float(np.mean(series))
    var_val = float(np.var(series))
    std_val = float(np.std(series))
    
    local_maxima = []
    local_minima = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            local_maxima.append(i)
        elif series[i] < series[i - 1] and series[i] < series[i + 1]:
            local_minima.append(i)
            
    moving_average = [
        float(np.mean(series[i:i + p])) for i in range(len(series) - p + 1)
    ]
    
    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std=std_val,
        local_maxima=local_maxima,
        local_minima=local_minima,
        moving_average=moving_average
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    vector, class_count = data.vector, data.class_count
    if class_count is None:
        class_count = int(np.max(vector)) + 1
        
    return np.eye(class_count)[vector]

use std::cmp::min;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    /// Creates an `m` x `n` matrix with all its elements set to `0`.
    #[inline]
    pub fn new(m: usize, n: usize) -> Matrix {
        Matrix {
            data: vec![vec![0.0; n]; m],
            rows: m,
            cols: n,
        }
    }

    /// Creates an `m` x `n` identity matrix.
    #[inline]
    pub fn identity(m: usize, n: usize) -> Matrix {
        let mut identity_matrix = Matrix::new(m, n);
        for i in 0..std::cmp::min(m, n) {
            identity_matrix.data[i][i] = 1.0;
        }
        identity_matrix
    }

    /// Returns the a_ij element of a matrix.
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        return self.data[i][j];
    }

    /// Returns the transpose of a matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j][i] = self.data[i][j];
            }
        }
        transposed
    }
}

/// Returns the matrix product of A and B.
/// 
/// Panics if `a.cols` is not equal to `b.rows`.
#[inline]
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "A must have the same number of columns as the number of rows in B.");
    let b_prime = b.transpose();
    let mut product = Matrix::new(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b_prime.rows {
            let mut sum: f64 = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, j) * b_prime.get(j, k);
            }
            product.data[i][j] = sum;
        }
    }
    product
}
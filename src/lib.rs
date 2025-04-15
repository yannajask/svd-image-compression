#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    /// Creates an `m` x `n` matrix with all its elements set to `0`.
    #[inline]
    pub fn new(m: usize, n: usize) -> Matrix {
        Matrix {
            data: vec![0.0; m * n],
            rows: m,
            cols: n,
        }
    }

    /// Creates an `m` x `n` identity matrix.
    #[inline]
    pub fn identity(n: usize) -> Matrix {
        let mut identity_matrix = Matrix::new(n, n);
        for i in 0..n {
            identity_matrix.data[i * n + i] = 1.0;
        }
        identity_matrix
    }

    /// Returns the `a_ij`-th element of a matrix.
    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        return self.data[i * self.cols + j];
    }

    /// Returns the transpose of a matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        transposed
    }

    pub fn modify_range(&mut self, i: usize, j: usize, other: &Matrix) {
        // this function needs to be fixed later to do the proper checks and be
        // optimized, but this is to get it working
        let m = other.rows;
        let n = other.cols;
        for k in i..m {
            for l in j..n {
                self.data[k * self.cols + l] = other.get(k, l);
            }
        }
    }

    pub fn get_range(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Matrix {
        let m = row_end - row_start;
        let n = col_end - col_start;
        let mut submatrix = Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                submatrix.data[i * n + j] = self.get(row_start + i, col_start + j);
            }
        }
        submatrix
    }

    pub fn scale(&self, s: f64) -> Matrix {
        let mut scaled_matrix = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                scaled_matrix.data[i * self.cols + j] = s * self.get(i, j);
            }
        }
        scaled_matrix
    }
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn norm(x: &[f64]) -> f64 {
    let mut norm = 0.0_f64;
    for i in x {
        norm += i * i;
    }
    norm.sqrt()
}

fn normalize(x: &[f64]) -> Vec<f64> {
    let norm = norm(x);
    match norm {
        0.0 => x.to_vec(),
        _ => x.iter().map(|&x| x / norm).collect(),
    }
}

/// Returns the matrix product of A and B.
/// 
/// Panics if `a.cols` is not equal to `b.rows`.
#[inline]
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "A must have the same number of columns as the number of rows in B.");
    let mut product = Matrix::new(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, j) * b.get(k, j);
            }
            product.data[i + b.cols + j] = sum;
        }
    }
    product
}

fn matrix_addition(a: &Matrix, b: Matrix) -> Matrix {
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.rows, b.rows);
    let mut sum = Matrix::new(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            sum.data[i + a.cols + j] = a.get(i, j) + b.get(i, j);
        }
    }
    sum
}

fn outer_product(u: &[f64], v: &[f64]) -> Matrix {
    let m = u.len();
    let n = v.len();
    let mut product = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            product.data[i * n + j] = u[i] * v[j];
        }
    }
    product
}

//https://www.math.iit.edu/~fass/477577_Chapter_12.pdf
pub fn householder_bidiag(a: &Matrix) -> (Matrix, Matrix, Matrix) {
    let m = a.rows;
    let n = a.cols;
    let mut b = a.clone();
    let mut v = Matrix::identity(m);
    let mut u = Matrix::identity(n);

    for k in 0..n {
        let x: Vec<f64> = (k..m).map(|i| a.data[i * n + k]).collect();
        let mut u_k = x.clone();
        u_k[0] += sign(x[0]) * norm(&x);
        u_k = normalize(&u_k);
        u.data[k*n..(k+1)*n].copy_from_slice(&u_k); // this is wrong
        // B(k: m, k: n) -= 2 * u_k (u_k^t * B[k:m, k:n])
        // 
        // B.modify_range(k, k, hh_transformation);

        if k <= (n - 2) {
            let x: Vec<f64> = (k+1..n).map(|j| a.data[k * n + j]).collect();
            let mut v_k = x.clone();
            v_k[0] += sign(x[0]) * norm(&x);
            v_k = normalize(&v_k);
            // goes to column vector in v
            // B[k:m, (k+1):n] -= 2 * A[k:m, (k+1):n, v_k] & v_k^t
        }
    }
    (v, b, u)
}
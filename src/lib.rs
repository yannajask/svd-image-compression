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
    x.iter().map(|&x| x / norm).collect()
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

//https://www.math.iit.edu/~fass/477577_Chapter_12.pdf
pub fn householder_bidiag(a: &Matrix) -> (Matrix, Matrix, Matrix) {
    let m = a.rows;
    let n = a.cols;
    let mut b = a.clone();
    let mut v = Matrix::identity(m);
    let mut u = Matrix::identity(n);

    for k in 0..n {
        // x = A(k: m,k)
        // u_k = x + sign(x[0]) * x.norm() * e_1
        // u_k = u_k.norm()
        let x: Vec<f64> = (k..m).map(|i| a.data[i * n + k]).collect();
        let mut u_k = x.clone();
        u_k[0] += sign(x[0]) * norm(&x);
        normalize(&mut u_k);
        u.data[k*n..(k+1)*n].copy_from_slice(&u_k); // this is wrong
        // B(k: m, k: n) -= 2 * u_k (u_k^t * B[k:m, k:n])

        if k <= (n - 2) {
            let x: Vec<f64> = (k+1..n).map(|j| a.data[k * n + j]).collect();
            let mut v_k = x.clone();
            v_k[0] += sign(x[0]) * norm(&x);
            normalize(&mut v_k);
            // goes to column vector in v
            // B[k:m, (k+1):n] -= 2 * A[k:m, (k+1):n, v_k] & v_k^t
        }
    }
    (v, b, u)
}
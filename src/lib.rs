use std::ops::{Index, IndexMut};
use std::fmt;

// to do: find a more elegant way to do this
static TOLERANCE: f64 = 1e-12;

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

    pub fn from_vec(m: usize, n: usize, data: &[f64]) -> Matrix {
        Matrix {
            data: data.to_vec(),
            rows: m,
            cols: n,
        }
    }

    /// Creates an `m` x `n` identity matrix.
    #[inline]
    pub fn identity(n: usize) -> Matrix {
        let mut identity_matrix = Matrix::new(n, n);
        for i in 0..n {
            identity_matrix[[i, i]] = 1.0;
        }
        identity_matrix
    }

    /// Returns the transpose of a matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed[[j, i]] = self[[i, j]];
            }
        }
        transposed
    }

    pub fn modify_range(&mut self, i: usize, j: usize, other: &Matrix) {
        // this function needs to be fixed later to do the proper checks and be
        // optimized, but this is to get it working
        let m = other.rows;
        let n = other.cols;
        for k in 0..m {
            for l in 0..n {
                self[[i + k, j + l]] = other[[k, l]];
            }
        }
    }

    pub fn get_range(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Matrix {
        let m = row_end - row_start;
        let n = col_end - col_start;
        let mut submatrix = Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                submatrix[[i, j]] = self[[row_start + i, col_start + j]];
            }
        }
        submatrix
    }

    pub fn scale(&self, s: f64) -> Matrix {
        let scaled_data: Vec<f64> = self.data.iter().map(|&x| s * x).collect();
        Matrix::from_vec(self.rows, self.cols, &scaled_data)
    }
}

impl Index<[usize; 2]> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [i, j] = index;
        unsafe { self.data.get_unchecked(i * self.cols + j) }
    }
}

impl IndexMut<[usize; 2]> for Matrix {
    #[inline]
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let [i, j] = index;
        unsafe { self.data.get_unchecked_mut(i * self.cols + j) }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                write!(f, "{:.3}", self[[i, j]])?;
                if j < self.cols - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i < self.rows - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
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
    let mut norm_sq = 0.0_f64;
    for i in x {
        norm_sq += i * i;
    }
    norm_sq.sqrt()
}

fn normalize(x: &[f64]) -> Vec<f64> {
    let norm = norm(x);
    if norm < TOLERANCE {
        vec![0.0, x.len() as f64]
    } else {
        x.iter().map(|&x| x / norm).collect()
    }
}

/// Returns the matrix product of A and B.
/// 
/// Panics if `a.cols` is not equal to `b.rows`.
// to do: use parallel strassen
#[inline]
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "A must have the same number of columns as the number of rows in B.");
    let mut product = Matrix::new(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a[[i, k]] * b[[k, j]];
            }
            product[[i, j]] = sum;
        }
    }
    product
}

// to do: use maps and from_vec
fn matrix_addition(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.cols);
    assert_eq!(a.rows, b.rows);
    let mut sum = Matrix::new(a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            sum[[i, j]] = a[[i, j]] + b[[i, j]];
        }
    }
    sum
}

// w is a unit vector
fn householder_reflection(w: &Matrix, n: usize) -> Matrix {
    let mut p = Matrix::identity(n);
    for i in 0..n {
        for j in 0..n {
            p[[i, j]] -= 2.0 * w[[i, 0]] * w[[j, 0]];
        }
    }
    p
}

//https://www.math.iit.edu/~fass/477577_Chapter_12.pdf
pub fn householder_bidiag(a: &Matrix) -> (Matrix, Matrix, Matrix) {
    let m = a.rows;
    let n = a.cols;
    let mut b = a.clone();

    let mut u = Matrix::identity(m); // left reflections
    let mut v = Matrix::identity(n); // right reflections

    for k in 0..n {
        if k < m {
            let x: Vec<f64> = (k..m).map(|i| b[[i, k]]).collect();
            let alpha = norm(&x);
            if alpha < TOLERANCE {
                let mut u_k = x.clone();
                u_k[0] += sign(x[0]) * alpha;
                u_k = normalize(&u_k);
                
                // this needs to be cleaned up
                //  B(k: m, k: n) -= 2 * u_k (u_k^t * B[k:m, k:n])
                /*
                let u_k = Matrix::from_vec(u_k.len(), 1, &u_k);
                let b_k = b.get_range(k, m, k, n);
                let tmp = matrix_multiply(&u_k.transpose(), &b_k);
                let scaled = matrix_multiply(&u_k, &tmp).scale(-2.0);
                let hh_transformation = matrix_addition(&b_k, &scaled);
                b.modify_range(k, k, &hh_transformation);
                u = matrix_multiply(&u, &householder_reflection(&u_k, k, n));
                //
                */
                let mut u_k_padded = Matrix::new(m, 1);
                for i in 0..u_k.len() {
                    u_k_padded[[k + i, 0]] = u_k[i];
                }
                let hr = householder_reflection(&u_k_padded, m);
                b = matrix_multiply(&hr, &b);
                u = matrix_multiply(&u, &hr);
            }
        }

        if k < n - 1 {
            let x: Vec<f64> = (k + 1..n).map(|j| b[[k, j]]).collect();
            let mut v_k = x.clone();
            let alpha = norm(&x);
            if alpha < TOLERANCE {
                v_k[0] += sign(x[0]) * alpha;
                v_k = normalize(&v_k);
                // this also needs to be cleaned up
                // B[k:m, (k+1):n] -= 2 * A[k:m, (k+1):n, v_k] & v_k^t
                /*
                let v_k = Matrix::from_vec(v_k.len(), 1, &v_k);
                let b_k = b.get_range(k, m, k + 1, n);
                let tmp = matrix_multiply(&b_k, &v_k);
                let scaled = matrix_multiply(&tmp, &v_k.transpose()).scale(-2.0);
                let hh_transformation = matrix_addition(&b_k, &scaled);
                b.modify_range(k, k + 1, &hh_transformation);
                v = matrix_multiply(&v, &householder_reflection(&v_k, k + 1, m));
                */
                let mut v_k_padded = Matrix::new(n, 1);
                for i in 0..v_k.len() {
                    v_k_padded[[k + 1 + i, 0]] = v_k[i];
                }
                let hr = householder_reflection(&v_k_padded, n);
                b = matrix_multiply(&b, &hr);
                v = matrix_multiply(&hr, &v);
            }
        }
    }
    (u, b, v)
}

fn jacobi_svd(u: &Matrix, b: &Matrix, v: &Matrix) -> Matrix {
    //let mut singular_values: Vec<f64>;
    //let m = b.rows;
    //let n = b.cols;
    unimplemented!();
}

fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tol: f64) {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let diff = (a[[i, j]] - b[[i, j]]).abs();
            assert!(diff <= tol, "Mismatch at ({}, {}), A: {} vs. B: {}", i, j, a[[i, j]], b[[i, j]]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tall_bidiag() {
        let a = Matrix::from_vec(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, b, v) = householder_bidiag(&a);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), TOLERANCE);
    }

    #[test]
    fn test_square_bidiag() {
        let a = Matrix::from_vec(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let (u, b, v) = householder_bidiag(&a);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), TOLERANCE);
    }

    #[test]
    fn test_wide_bidiag() {
        let a = Matrix::from_vec(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, b, v) = householder_bidiag(&a);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), TOLERANCE);
    }
}
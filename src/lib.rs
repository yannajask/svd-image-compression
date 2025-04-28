use std::ops::{Index, IndexMut, Range};
use std::mem::swap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    /// Creates an `m` x `n` matrix with all its elements set to `0`.
    /// 
    /// Panics if one of `m` or `n` is zero.
    #[inline]
    pub fn new(m: usize, n: usize) -> Matrix {
        assert!(m > 0 && n > 0, "Dimensions must be positive! Given: {}x{}", m, n);
        Matrix {
            data: vec![0.0; m * n],
            rows: m,
            cols: n,
        }
    }

    /// Creates an `m` x `n` matrix from a vector with length `mn`.
    #[inline]
    pub fn from_vec(m: usize, n: usize, data: &[f64]) -> Matrix {
        Matrix {
            data: data.to_vec(),
            rows: m,
            cols: n,
        }
    }

    /// Creates an `n` x `n` identity matrix.
    /// 
    /// Panics if `n` is zero.
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

    /// Returns the dimensions (rows, columns) of a matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the submatrix given a range of zero-based indices.
    /// 
    /// Panics if indices are out of range of the matrix's dimensions.
    pub fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Matrix {
        assert!(rows.end <= self.rows && cols.end <= self.cols,
                "Slice [{:#?}, {:#?}] must be within the dimensions {}x{}",
                rows, cols, self.rows, self.cols);
        let mut data = Vec::new();
        let m = rows.end - rows.start;
        let n = cols.end - cols.start;
        for i in rows.start..rows.end {
            for j in cols.start..cols.end {
                data.push(self[[i, j]]);
            }
        }
        Matrix::from_vec(m, n, &data)
    }

    pub fn apply_left_givens(&mut self, c: f64, s: f64, i: usize, j: usize, p: usize, q: usize) {
        for k in p..=q {
            let t = c * self[[i, k]] + s * self[[j, k]];
            self[[j, k]] = -s * self[[i, k]] + c * self[[j, k]];
            self[[i, k]] = t;
        }
    }

    pub fn apply_right_givens(&mut self, c: f64, s: f64, i: usize, j: usize, p: usize, q: usize) {
        for k in p..=q {
            let t = c * self[[k, i]] - s * self[[k, j]];
            self[[k, j]] = s * self[[k, i]] + c * self[[k, j]];
            self[[k, i]] = t;
        }
    }
}

impl Index<[usize; 2]> for Matrix {
    type Output = f64;
    #[inline]
    /// Returns the element of a matrix at the zero-based index (i, j).
    /// 
    /// Panics if `i` or `j` are outside the ranges `0 <= i <= m`, `0 <= j <= n`, respectively.
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [i, j] = index;
        unsafe { self.data.get_unchecked(i * self.cols + j) }
    }
}

impl IndexMut<[usize; 2]> for Matrix {
    #[inline]
    /// Returns a mutable reference to the element of a matrix at the zero-based index (i, j).
    /// 
    /// Panics if `i` or `j` are outside the ranges `0 <= i <= m`, `0 <= j <= n`, respectively.
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        let [i, j] = index;
        unsafe { self.data.get_unchecked_mut(i * self.cols + j) }
    }
}

impl fmt::Display for Matrix {
    #[inline]
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

/// Computes the norm of a given vector.
#[inline]
fn norm(x: &[f64]) -> f64 {
    let mut norm_sq = 0.0_f64;
    for i in x {
        norm_sq += i * i;
    }
    norm_sq.sqrt()
}

/// Normalizes a given vector. Returns the zero vector
/// for norms within `1e-16`.
#[inline]
fn normalize(x: &[f64]) -> Vec<f64> {
    let norm = norm(x);
    if norm < 1e-16 {
        vec![0.0; x.len()]
    } else {
        x.iter().map(|&x| x / norm).collect()
    }
}

/// Returns the matrix product of A and B.
/// 
/// Panics if `a.cols` is not equal to `b.rows`.
#[inline]
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.cols == b.rows, "Mismatch of dimensions! A: {}x{} vs. B: {}x{}", a.rows, a.cols, b.rows, b.cols);
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

/// Returns the bidiagonal decomposition of A = UBV^T.
/// 
/// U is an orthogonal `m` x `m` matrix,
/// V is an orthogonal `n` x `n` matrix,
/// and B is an `m` x `n` upper bidiagonal matrix.
/// 
/// Note that V is returned, not its transpose.
#[inline]
pub fn bidiagonalize(a: &Matrix) -> (Matrix, Matrix, Matrix) {
    let (m, n) = a.shape();
    let mut b = a.clone();           // bidiagonal matrix
    let mut u = Matrix::identity(m); // left reflections
    let mut v = Matrix::identity(n); // right reflections

    for k in 0..m.min(n) {
        let x: Vec<f64> = (k..m).map(|i| b[[i, k]]).collect();
        let mut u_k = x.clone();
        u_k[0] += x[0].signum() * norm(&x);
        u_k = normalize(&u_k);

        // B[k:m, k:n] -= 2 * u_k (u_k^T * B[k:m, k:n])
        for j in k..n {
            let mut product = 0.0;
            for i in 0..(m - k) {
                product += u_k[i] * b[[k + i, j]];
            }
            for i in 0..(m - k) {
                b[[k + i, j]] -= 2.0 * u_k[i] * product;
            }
        }

        // accumulate transformation in U
        for i in 0..m {
            let mut product = 0.0;
            for j in 0..(m - k) {
                product += u[[i, k + j]] * u_k[j];
            }
            for j in 0..(m - k) {
                u[[i, k + j]] -= 2.0 * product * u_k[j];
            }
        }

        if k < n - 1 {
            let x: Vec<f64> = (k + 1..n).map(|j| b[[k, j]]).collect();
            let mut v_k = x.clone();
            v_k[0] += x[0].signum() * norm(&x);
            v_k = normalize(&v_k);

            // B[k:m, (k+1):n] -= 2 * (A[k:m, (k+1):n] * v_k) v_k^T
            for i in k..m {
                let mut product = 0.0;
                for j in 0..(n - k - 1) {
                    product += b[[i, k + 1 + j]] * v_k[j];
                }
                for j in 0..(n - k - 1) {
                    b[[i, k + 1 + j]] -= 2.0 * product * v_k[j];
                }
            }

            // accumulate transformation in V
            for i in 0..n {
                let mut product = 0.0;
                for j in 0..(n - k - 1) {
                    product += v[[i, k + 1 + j]] * v_k[j];
                }
                for j in 0..(n - k - 1) {
                    v[[i, k + 1 + j]] -= 2.0 * product * v_k[j];
                }
            }
        }
    }
    (u, b, v)
}

/// Returns the Givens rotation coefficients `c` and `s`, given `a` and `b`.
/// 
/// https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation
#[inline]
pub fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (a.signum(), 0.0)
    } else if a == 0.0 {
        (0.0, -b.signum())
    } else if a.abs() > b.abs() {
        let t = b / a;
        let u = b.signum() * (1.0 + t * t).sqrt();
        let c = 1.0 / u;
        (c, -c * t)
    } else {
        let t = a / b;
        let u = b.signum() * (1.0 + t * t).sqrt();
        (t / u, -1.0 / u)
    }
}

/// 
///
/// Panics if `b.rows` < `b.cols`.
/// 
/// https://utminers.utep.edu/xzeng/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln15.pdf
#[inline]
pub fn qr_step(u: &mut Matrix, b: &mut Matrix, v: &mut Matrix, p: usize, q: usize) {
    let (m, n) = b.shape();
    assert!(m >= n, "B must have more rows than columns: {}x{}", m, n);
    assert!(q - p > 0, "Indices p and q must make at least a 2x2 submatrix! p: {}, q: {}", p, q);

    // get wilkinson shift
    let a_qm1 = b[[q - 1, q -1]];
    let b_qm1 = b[[q - 1, q]];
    let a_q = b[[q, q]];

    let delta = (a_qm1 - a_q) / 2.0;
    let mu = a_q - (delta.signum() * b_qm1 * b_qm1) / (delta.abs() + (delta * delta + b_qm1 * b_qm1).sqrt());

    let t_11 = b[[p, p]] * b[[p, p]];
    let t_12 = b[[p, p]] * b[[p, p + 1]];

    let mut y = t_11 - mu;
    let mut z = t_12;

    // qr steps
    for k in p..q {
        // right rotation
        let (c, s) = givens_rotation(y, z);
        b.apply_right_givens(c, s, k, k + 1, p, q);
        v.apply_right_givens(c, s, k, k + 1, 0, n - 1);

        // left rotation
        y = b[[k, k]];
        z = b[[k + 1, k]];
        let (c, s) = givens_rotation(y, z);
        b.apply_left_givens(c, s, k + 1, k, p, q);
        u.apply_right_givens(c, -s, k + 1, k, 0, m - 1);

        // update y and z
        if p < q - 1 {
            y = b[[k, k + 1]];
            z = b[[k, k + 2]];
        }
    }
}

/// Returns the bidiagonal decomposition of A = USV^T.
/// 
/// U is an orthogonal `m` x `m` matrix,
/// V is an orthogonal `n` x `n` matrix,
/// and S is an `m` x `n` matrix with the singular values of A on its diagonal.
/// 
/// Note that V is returned, not its transpose.
#[inline]
pub fn svd(a: &Matrix) -> (Matrix, Matrix, Matrix) {
    let (mut m, mut n) = a.shape();
    let (mut u, mut b, mut v);
    let wide = m < n;

    // transpose wide matrices where m < n
    if wide {
        (u, b, v) = bidiagonalize(&a.transpose());
        swap(&mut m, &mut n);
    } else {
        (u, b, v) = bidiagonalize(&a);
    }

    println!("U Matrix after bidiagonalization:\n{}", u);

    let tol = 1e-16;
    let mut q = n - 1;
    
    while q > 0 {
        for i in 0..(n - 1) {
            if b[[i, i + 1]].abs() <= tol * (b[[i, i]].abs() + b[[i + 1, i +1]].abs()) {
                b[[i, i + 1]] = 0.0;
            }
        }

        q = 0;
        for k in (0..(n - 1)).rev() {
            if b[[k, k + 1]].abs() > tol {
                q = k + 1;
                break;
            }
        }
        if q == 0 { break }

        let mut p = 0;
        for k in (0..(q - 1)).rev() {
            if b[[k, k + 1]].abs() < tol {
                p = k + 1;
                break;
            }
        }

        let mut found_zero = false;
        for k in p..q {
            if b[[k, k]].abs() < tol {
                found_zero = true;
                break;
            }
        }

        if found_zero {
            for k in p..q {
                if b[[k, k]].abs() < tol {
                    let (c, s) = givens_rotation(b[[k, k + 1]], b[[k + 1, k + 1]]);
                    b.apply_left_givens(c, s, k + 1, k, k, q);
                    u.apply_right_givens(c, -s, k + 1, k, 0, m - 1);
                }
            }
            continue;
        }
        qr_step(&mut u, &mut b, &mut v, p, q);
    }

    // TO DO: order columns from largest to smallest singular values

    if wide {
        (v, b.transpose(), u)
    } else {
        println!("U Matrix after SVD:\n{}", u);
        (u, b, v)
    }
}

/// Returns the rank of a diagonal matrix.
#[inline]
pub fn rank(sigma: &Matrix) -> usize {
    let (m, n) = sigma.shape();
    let mut rank = 0;
    for i in 0..m.min(n) {
        if sigma[[i, i]] > 0.0 { rank += 1 }
    }
    rank
}

/// Returns the rank `k` truncated matrices of an SVD USV^T.
#[inline]
pub fn rank_k_approximation(u: &Matrix, sigma: &Matrix, v: &Matrix, k: usize) -> Matrix {
    let (m, n) = sigma.shape();
    assert!(m >= k && n >= k, "Cannot truncate a {}x{} matrix to {}x{}", m, n, k, k);
    let u_k = u.slice(0..m, 0..k);
    let sigma_k = sigma.slice(0..k, 0..k);
    let v_k = v.slice(0..n, 0..k);
    matrix_multiply(&u_k, &matrix_multiply(&sigma_k, &v_k.transpose()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-4;

    fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tol: f64) {
        assert_eq!(a.shape(), b.shape());
        for i in 0..a.rows {
            for j in 0..a.cols {
                let diff = (a[[i, j]] - b[[i, j]]).abs();
                assert!(diff <= tol, "Mismatch at ({}, {}), A: {} vs. B: {}", i, j, a[[i, j]], b[[i, j]]);
            }
        }
    }

    fn assert_orthogonal(a: &Matrix, tol: f64) {
        let (m, n) = a.shape();
        assert_eq!(m, n);
        let a_t = a.transpose();

        // create I, AA^T, (A^T)A
        let identity = Matrix::identity(n);
        let aa_t = matrix_multiply(&a, &a_t);
        let a_ta = matrix_multiply(&a_t, &a);

        // check for equality within certain tolerance
        assert_matrix_approx_eq(&aa_t, &identity, tol);
        assert_matrix_approx_eq(&a_ta, &identity, tol);
    }

    #[test]
    fn tall_bidiag() {
        let a = Matrix::from_vec(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, b, v) = bidiagonalize(&a);
        assert_orthogonal(&u, TOLERANCE);
        assert_orthogonal(&v, TOLERANCE);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), TOLERANCE);
    }

    #[test]
    fn square_bidiag() {
        let a = Matrix::from_vec(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let (u, b, v) = bidiagonalize(&a);
        assert_orthogonal(&u, TOLERANCE);
        assert_orthogonal(&v, TOLERANCE);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), TOLERANCE);
    }

    #[test]
    fn wide_bidiag() {
        let a = Matrix::from_vec(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, b, v) = bidiagonalize(&a);
        assert_orthogonal(&u, TOLERANCE);
        assert_orthogonal(&v, TOLERANCE);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), TOLERANCE);
    }

    /// https://en.wikipedia.org/wiki/Givens_rotation#Triangularization
    #[test]
    fn givens_triangularization() {
        let mut a = Matrix::from_vec(3, 3, &[6.0, 5.0, 0.0, 5.0, 1.0, 4.0, 0.0, 4.0, 3.0]);
        let (c, s) = givens_rotation(a[[0, 0]], a[[1, 0]]);
        a.apply_left_givens(c, s, 1, 0, 0, 2);
        let (c, s) = givens_rotation(a[[1, 1]], a[[2, 1]]);
        a.apply_left_givens(c, s, 2, 1, 0, 2);
        let r = Matrix::from_vec(3, 3, &[7.8102, 4.4813, 2.5607, 0.0, 4.6817, 0.9665, 0.0, 0.0, -4.1843]);
        assert_matrix_approx_eq(&a, &r, TOLERANCE);
    }

    #[test]
    fn tall_svd() {
        let a = Matrix::from_vec(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, s, v) = svd(&a);
        assert_orthogonal(&u, TOLERANCE);
        assert_orthogonal(&v, TOLERANCE);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&s, &v.transpose())), TOLERANCE);
    }

    #[test]
    fn square_svd() {
        let a = Matrix::from_vec(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let (u, s, v) = svd(&a);
        assert_orthogonal(&u, TOLERANCE);
        assert_orthogonal(&v, TOLERANCE);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&s, &v.transpose())), TOLERANCE);
    }

    #[test]
    fn wide_svd() {
        let a = Matrix::from_vec(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, s, v) = svd(&a);
        assert_orthogonal(&u, TOLERANCE);
        assert_orthogonal(&v, TOLERANCE);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&s, &v.transpose())), TOLERANCE);
    }
}
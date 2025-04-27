use std::ops::{Index, IndexMut, Range};
use std::fmt;

// to do:
// -add proper asserts to constructors
// -optimize matrix multiplication with strassen algorithm
// -remove repeated for loops for bidiagonalization
// -implement jacobi svd
// -implement rank fn
// -implement rank_k truncation fn
// -implement is_bidiagonal, is_diagonal, is_orthogonal matrix fn's
// -add unit tests for svd
// -put functions in modules:
// ---image_compression
//     |---svd
//     |---image_to_matrix


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

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn slice(&self, row_range: Range<usize>, col_range: Range<usize>) -> Matrix {
        let mut data = Vec::new();
        let m = row_range.end - row_range.start;
        let n = col_range.end - col_range.start;
        for i in row_range.start..row_range.end {
            for j in col_range.start..col_range.end {
                data.push(self[[i, j]]);
            }
        }
        Matrix::from_vec(m, n, &data)
    }

    // https://www.sciencedirect.com/topics/engineering/givens-rotation
    pub fn apply_left_givens(&mut self, c: f64, s: f64, i: usize, j: usize) {
        for k in 0..self.cols {
            let t = c * self[[i, k]] + s * self[[j, k]];
            self[[j, k]] = -s * self[[i, k]] + c * self[[j, k]];
            self[[i, k]] = t;
        }
    }

    pub fn apply_right_givens(&mut self, c: f64, s: f64, i: usize, j: usize) {
        for k in 0..self.rows {
            let t = c * self[[k, i]] + s * self[[k, j]];
            self[[k, j]] = -s * self[[k, i]] + c * self[[k, j]];
            self[[k, i]] = t;
        }
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

fn norm(x: &[f64]) -> f64 {
    let mut norm_sq = 0.0_f64;
    for i in x {
        norm_sq += i * i;
    }
    norm_sq.sqrt()
}

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
// to do: use strassen algorithm
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

//https://www.math.iit.edu/~fass/477577_Chapter_12.pdf
// to do: less repeated iteration when updating matrices or run in parallel
pub fn householder_bidiag(a: &Matrix) -> (Matrix, Matrix, Matrix) {
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

// https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation
fn givens_rotation(a: f64, b: f64) -> (f64, f64) {
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

// https://dspace.mit.edu/bitstream/handle/1721.1/75282/18-335j-fall-2006/contents/lecture-notes/lec16.pdf
// https://faculty.ucmerced.edu/mhyang/course/eecs275/lectures/lecture17.pdf
// assumes m >= n
fn qr_step(u: &mut Matrix, b: &mut Matrix, v: &mut Matrix, p: usize, q: usize) {
    let (m, n) = b.shape();

    // get wilkinson shift
    let delta = (b[[q - 2, q - 2]] - b[[q - 1, q - 1]]) / 2.0;
    let b_m1 = b[[q - 2, q - 1]] * b[[q - 2, q - 1]];
    let mu = b[[q - 1, q - 1]] - (delta.signum() * b_m1) / (delta.abs() + (delta * delta + b_m1).sqrt());

    let mut y = b[[p, p]] - mu;
    let mut z = b[[p, p + 1]];

    // qr steps
    for k in p..(q - 1) {
        // left rotation
        let (c, s) = givens_rotation(y, z);
        b.apply_left_givens(c, s, k, k + 1);
        u.apply_left_givens(c, s, k, k + 1);

        // right rotation
        y = b[[k, k]];
        z = b[[k + 1, k]];
        let (c, s) = givens_rotation(y, z);
        b.apply_right_givens(c, s, k, k + 1);
        v.apply_right_givens(c, s, k, k + 1);

        // update y and z
        if k < q - 2 { 
            y = b[[k, k + 1]];
            z = b[[k, k + 2]];
        }
    }
}

pub fn svd(a: &Matrix) -> (Matrix, Matrix, Matrix) {
    let (mut m, mut n) = a.shape();
    let (mut u, mut b, mut v);
    let wide = m < n;

    // transpose wide matrices where m < n
    if wide {
        (v, b, u) = householder_bidiag(&a.transpose());
        std::mem::swap(&mut m, &mut n);
    } else {
        (u, b, v) = householder_bidiag(&a);
    }
    println!("bidiagonalized matrix");

    let mut q = 0;
    let tol = 1e-12;

    while q < n - 1 {
        println!("q: {}", q);
        // zero small superdiagonal entries
        for i in 0..(n - 1) {
            if b[[i, i + 1]].abs() < tol * (b[[i, i]].abs() + b[[i + 1, i + 1]].abs()) {
                println!("B zeroed out at {}, {}", i, i + 1);
                b[[i, i + 1]] = 0.0;
            }
        }
        
        // find largest q and smallest p such that B = diag(B11, B22, B33)
        // where B33 is diagonal and B22 has nonzero superdiagonal
        q = 0;
        for i in (0..(n - 1)).rev() {
            if b[[i, i + 1]].abs() > 0.0 {
                q = i + 1;
                break;
            }
        }

        if q == 0 {
            println!("Breaking loop because q == 0");
            break;
        }

        let mut p = 0;
        for i in (0..q).rev() {
            if b[[i, i + 1]].abs() == 0.0 {
                p = i + 1;
                break;
            }
        }

        println!("Q: {}, P: {}", q, p);

        if q - p == 1 {
            if b[[p, p]] < 0.0 {
                b[[p, p]] = -b[[p, p]];
                for i in 0..m {
                    u[[i, p]] = -u[[i, p]];
                }
            }
            q -= 1;
        } else if q < n {
            println!("QR step on p = {}, q = {}", p, q);
            qr_step(&mut u, &mut b, &mut v, p, q);
            for i in 0..(n - 1) {
                println!("b[{}, {}]: {}", i, i + 1, b[[i, i + 1]]);
            }
        }
    }

    // make singular values positive
    for i in 0..n {
        if b[[i, i]] < 0.0 {
            b[[i, i]] = -b[[i, i]];
            for j in 0..n {
                v[[j, i]] = -v[[j, i]];
            }
        }
    }

    if wide {
        (v, b.transpose(), u)
    } else {
        (u, b, v)
    }
}

pub fn rank(sigma: &Matrix) -> usize {
    let mut rank = 0;
    for i in 0..sigma.rows.min(sigma.cols) {
        if sigma[[i, i]] > 0.0 { rank += 1 }
    }
    rank
}

pub fn rank_k_approximation(u: &Matrix, sigma: &Matrix, v: &Matrix, k: usize) -> Matrix {
    let (m, n) = sigma.shape();
    let u_k = u.slice(0..m, 0..k);
    let sigma_k = sigma.slice(0..k, 0..k);
    let v_k = v.slice(0..n, 0..k);
    matrix_multiply(&u_k, &matrix_multiply(&sigma_k, &v_k.transpose()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrix_approx_eq(a: &Matrix, b: &Matrix, tol: f64) {
        assert_eq!(a.shape(), b.shape());
        for i in 0..a.rows {
            for j in 0..a.cols {
                let diff = (a[[i, j]] - b[[i, j]]).abs();
                assert!(diff <= tol, "Mismatch at ({}, {}), A: {} vs. B: {}", i, j, a[[i, j]], b[[i, j]]);
            }
        }
    }

    #[test]
    fn tall_bidiag() {
        let a = Matrix::from_vec(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, b, v) = householder_bidiag(&a);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), 1e-12);
    }

    #[test]
    fn square_bidiag() {
        let a = Matrix::from_vec(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let (u, b, v) = householder_bidiag(&a);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), 1e-12);
    }

    #[test]
    fn wide_bidiag() {
        let a = Matrix::from_vec(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (u, b, v) = householder_bidiag(&a);
        assert_matrix_approx_eq(&a, &matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())), 1e-12);
    }
}
use image::{GrayImage, ImageBuffer, ImageReader, Luma};
use svd_image_compression::{householder_bidiag, matrix_multiply, svd, Matrix};


fn main() {
    // eventually want to get path name to a variable
    // also this can just be its own function i think
    /*
    let input_file = "demo.jpeg"
    let img = ImageReader::open(input_file)?.decode()?.to_luma8();
    let a = image_to_matrix(&img);
    let (u, sigma, v) = svd(a);
    let rank = rank(sigma);
    // ...
    let mut a_k = rank_k_approximation(u, sigma, v, k);
    // ...
    let compressed_img = matrix_to_image(&a_k);
    let output_file = strip_extension_and_append(input_file, rank);
    compressed_img.save(output_file)?;
    Ok(())
    */

    let test = Matrix::from_vec(3, 3, &[35.0, 2.0, 3.0, 255.0, 5.0, 63.0, 7.0, 8.0, 9.0]);
    println!("A:\n{}", test);
    let (u, b, v) = householder_bidiag(&test);
    println!("V:\n{}", v);
    println!("B:\n{}", b);
    println!("U:\n{}", u);
    println!("A:\n{}", matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())));

    let test_t = test.transpose();
    println!("A^T:\n{}", test_t);
    let (u, b, v) = householder_bidiag(&test_t);
    println!("A^T:\n{}", matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())));
    println!("A:\n{}", matrix_multiply(&v, &matrix_multiply(&b.transpose(), &u.transpose())));
    /*
    let (u, sigma, v) = svd(&test);
    println!("V:\n{}", v);
    println!("S:\n{}", sigma);
    println!("U:\n{}", b);
    println!("A:\n{}", matrix_multiply(&u, &matrix_multiply(&sigma, &v.transpose())));
    */
}

// want to move these to lib.rs later
fn image_to_matrix(img: &GrayImage) -> Matrix {
    let (width, height) = img.dimensions();
    let mut a = Matrix::new(height as usize, width as usize);
    for i in 0..height {
        for j in 0..width {
            let pixel = img.get_pixel(i, j);
            a[[j as usize, i as usize]] = pixel[0] as f64 / 255.0;
        }
    }
    a
}

fn matrix_to_image(a: &Matrix) -> GrayImage {
    let height = a.rows as u32;
    let width = a.cols as u32;
    let mut img = ImageBuffer::new(width, height);
    for i in 0..height {
        for j in 0..width {
            let pixel = (a[[j as usize, i as usize]] * 255.0).round().clamp(0.0, 255.0) as u8;
            img.put_pixel(i, j, Luma([pixel]));
        }
    }
    img
}

fn strip_extension_and_append(file_name: &str, k: i32) -> String {
    let prefix = match file_name.find('.') {
        Some(pos) => &file_name[..pos],
        _ => file_name,
    };
    format!("{} rank {}.jpeg", prefix, k)
}
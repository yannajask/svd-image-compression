use image::{GrayImage, ImageBuffer, ImageReader, Luma};
use svd_image_compression::{householder_bidiag, matrix_multiply, Matrix};


fn main() {
    // eventually want to get path name to a variable
    // also this can just be its own function i think
    /*
    let img = ImageReader::open("test.png")?.decode()?;
    let gray_img = img.to_luma8();
    let a = image_to_matrices(&gray_img);
    let a_k = a;
    // ...
    let compressed_img = matrix_to_image(&a_k);
    compressed_img.save("rank_k.jpg")?;
    Ok(())
    */

    let test = Matrix::from_vec(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    println!("A:\n{}", test);
    let (u, b, v) = householder_bidiag(&test);
    println!("V:\n{}", v);
    println!("B:\n{}", b);
    println!("U:\n{}", u);
    println!("A:\n{}", matrix_multiply(&u, &matrix_multiply(&b, &v.transpose())));
}

// want to move these to lib.rs later
fn image_to_matrices(img: &GrayImage) -> Matrix {
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
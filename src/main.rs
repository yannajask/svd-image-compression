use image::{GrayImage, ImageBuffer, ImageReader, Luma};
use svd_image_compression::{rank, rank_k_approximation, svd, Matrix};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_file = "demo.jpeg";
    let img = ImageReader::open(input_file)?.decode()?.to_luma8();
    let a = image_to_matrix(&img);
    let (u, s, v) = svd(&a, 500);
    let _rank = rank(&s);
    let k = 100;
    let a_k = rank_k_approximation(&u, &s, &v, k);
    // ...
    let compressed_img = matrix_to_image(&a_k);
    let output_file = strip_extension_and_append(input_file, k as i32);
    compressed_img.save(output_file)?;
    Ok(())
}

// want to move these to lib.rs later
fn image_to_matrix(img: &GrayImage) -> Matrix {
    let (width, height) = img.dimensions();
    let mut a = Matrix::new(height as usize, width as usize);
    for i in 0..height {
        for j in 0..width {
            let pixel = img.get_pixel(j, i);
            a[[i as usize, j as usize]] = pixel[0] as f64 / 255.0;
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
            let pixel = (a[[i as usize, j as usize]] * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            img.put_pixel(j, i, Luma([pixel]));
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
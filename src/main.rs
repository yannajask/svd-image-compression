use std::{env, io};
use image::{GrayImage, ImageBuffer, ImageReader, Luma};
use svd_image_compression::{rank, rank_k_approximation, svd, Matrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_image> [max_iterations]", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let max_iterations = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or(500)
    } else {
        500
    };

    println!("Calculating SVD on {}, with {} max iterations...", &input_file, max_iterations);

    // compute initial svd with max iterations
    let img = ImageReader::open(input_file)?.decode()?.to_luma8();
    let a = image_to_matrix(&img);
    let (u, s, v) = svd(&a, max_iterations);
    let rank = rank(&s);

    println!("Image rank is k = {}.", rank);
    println!("Enter the number of singular values (k) to keep.");

    // get k from input
    let mut k = String::new();

    io::stdin()
        .read_line(&mut k)
        .expect("Failed to read input.");

    let mut k: usize = k.trim().parse().expect("Please give a a valid number!");

    // default to using k = rank
    if k > rank {
        println!("k = {} is greater than rank = {}, defaulting to k = {}.", k, rank, rank);
        k = rank;
    }

    // compute rank k approximation and save compressed image
    let a_k = rank_k_approximation(&u, &s, &v, k);
    let compressed_img = matrix_to_image(&a_k);
    let output_file = strip_extension_and_append(input_file, k as i32);
    compressed_img.save(&output_file)?;

    println!("Saved compressed image as {}", &output_file);
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
extern crate rust_ndarray_test;

use ndarray::{concatenate, prelude::*, stack};
use rust_ndarray_test::common::ndarray_init::*;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const SHAPE: (usize, usize) = (1024, 1024);
const N_MEASURES: usize = 100;
const SAVE_PATH: &str = "./data/process_time.csv";

macro_rules! measure {
    ($x:expr, $y: tt) => {{
        let mut elapsed: Vec<f64> = Vec::new();
        for _ in (0..$y) {
            let start = Instant::now();
            let _1 = $x;
            let end = start.elapsed();
            elapsed.push(
                end.as_secs() as f64
                    + end.subsec_micros() as f64 * 1E-6
                    + end.subsec_nanos() as f64 * 1E-9,
            );
        }
        let elapsed: Array1<f64> = Array::from(elapsed);
        let mean_msec: f64 = elapsed.mean().unwrap() * 1000.;
        let std_msec: f64 = elapsed.std(0.) * 1000.;
        // let elapsed_msec: f64 = 1000. * elapsed / $y as f64;
        println!(
            "averaged process time over {} times: {:.6} +/- {:.6} msec.",
            $y, mean_msec, std_msec
        );
        vec![mean_msec, std_msec]
    }};
}

fn test_create_normal_distributed_ndarray<D, Sh>(mu: f64, sigma: f64, shape: Sh) -> Array<f64, D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    initialize_randomized_ndarray(DistributionEnum::Normal, shape, &[mu, sigma])
}

fn test_create_poisson_distributed_ndarray<D, Sh>(mu: f64, shape: Sh) -> Array<f64, D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    initialize_randomized_ndarray(DistributionEnum::Poisson, shape, &[mu])
}

fn test_create_uniform_distributed_ndarray<D, Sh>(low: f64, high: f64, shape: Sh) -> Array<f64, D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    initialize_randomized_ndarray(DistributionEnum::Normal, shape, &[low, high])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_measures: usize = match args.len() {
        2 => args[1].parse().unwrap(),
        _ => N_MEASURES,
    };
    let mut elapsed_list: Vec<Vec<f64>> = Vec::new();

    println!("Array creation:");
    println!("Array::range");
    let first: f64 = 0.0;
    let end: f64 = 10.0;
    let step: f64 = (end - first) / (SHAPE.0 * SHAPE.1) as f64;
    elapsed_list.push(measure!(
        Array::<f64, Ix1>::range(first, end, step),
        n_measures
    ));

    println!("Array::linspace");
    let first: f64 = 0.0;
    let end: f64 = 10.0;
    elapsed_list.push(measure!(
        Array::<f64, Ix1>::linspace(first, end, SHAPE.0 * SHAPE.1),
        n_measures
    ));

    println!("Array::ones");
    elapsed_list.push(measure!(Array::<f64, Ix2>::ones(SHAPE), n_measures));

    println!("Array::zeros");
    elapsed_list.push(measure!(Array::<f64, Ix2>::zeros(SHAPE), n_measures));

    println!("Array::from_elem");
    let v: f64 = 7.0;
    elapsed_list.push(measure!(Array::from_elem(SHAPE, v), n_measures));

    println!("Array::eye");
    elapsed_list.push(measure!(Array::<f64, Ix2>::eye(SHAPE.0), n_measures));

    println!("Randomize:");
    println!("Normal distribution");
    let mu: f64 = 0.0;
    let sigma: f64 = 1.0;
    elapsed_list.push(measure!(
        test_create_normal_distributed_ndarray(mu, sigma, SHAPE),
        n_measures
    ));

    println!("Poisson distribution");
    let mu: f64 = 10.0;
    elapsed_list.push(measure!(
        test_create_poisson_distributed_ndarray(mu, SHAPE),
        n_measures
    ));

    println!("Uniform distributino");
    let low: f64 = 0.0;
    let high: f64 = 1.0;
    elapsed_list.push(measure!(
        test_create_uniform_distributed_ndarray(low, high, SHAPE),
        n_measures
    ));

    println!("Mathematics:");
    let mu: f64 = 0.0;
    let sigma: f64 = 1.0;
    let vec: Array1<f64> =
        initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE.0 * SHAPE.1, &[mu, sigma]);
    let vec2: Array1<f64> =
        initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE.0 * SHAPE.1, &[mu, sigma]);
    let mat: Array2<f64> =
        initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE, &[mu, sigma]);
    let mat2: Array2<f64> =
        initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE, &[mu, sigma]);
    let mut mat3: Array2<f64> = mat.clone();

    println!("mat.dot(&mat2)");
    elapsed_list.push(measure!(mat.dot(&mat2), n_measures));

    println!("mat.dot(&vec)");
    elapsed_list.push(measure!(mat.dot(&vec.slice(s![0..SHAPE.0])), n_measures));

    println!("vec.dot(&mat)");
    elapsed_list.push(measure!(vec.slice(s![0..SHAPE.0]).dot(&mat), n_measures));

    println!("vec.dot(&vec2)");
    elapsed_list.push(measure!(vec.dot(&vec2), n_measures));

    println!("&mat + &mat2");
    elapsed_list.push(measure!(&mat + &mat2, n_measures));

    println!("mat**3");
    elapsed_list.push(measure!(mat.mapv(|v| v.powi(3)), n_measures));

    println!("sqrt(mat)");
    elapsed_list.push(measure!(mat.mapv(f64::sqrt), n_measures));

    println!("mat > 0.5");
    elapsed_list.push(measure!(mat.mapv(|v| v > 0.5), n_measures));

    println!("mat.sum()");
    elapsed_list.push(measure!(mat.sum(), n_measures));

    println!("mat.sum_axis(Axis(1))");
    elapsed_list.push(measure!(mat.sum_axis(Axis(1)), n_measures));

    println!("mat.mean()");
    elapsed_list.push(measure!(mat.mean(), n_measures));

    println!("mat.mean_axis(Axis(1))");
    elapsed_list.push(measure!(mat.mean_axis(Axis(1)), n_measures));

    println!("mat.abs_diff_eq(&mat2, 1E-8)");
    elapsed_list.push(measure!(mat.abs_diff_eq(&mat2, 1E-8), n_measures));

    println!("mat.diag()");
    elapsed_list.push(measure!(mat.diag(), n_measures));

    println!("Array manipulation:");
    println!("mat.fill(3.0)");
    elapsed_list.push(measure!(mat3.fill(3.), n_measures));

    println!("mat.assign(&mat2)");
    elapsed_list.push(measure!(mat3.assign(&mat2), n_measures));

    println!("concatenate![Axis(1), mat, mat2]");
    elapsed_list.push(measure!(concatenate![Axis(1), mat3, mat2], n_measures));

    println!("stack![Axis(1), mat, mat2]");
    elapsed_list.push(measure!(stack![Axis(1), mat3, mat2], n_measures));

    println!("mat.insert_axis(Axis(2))");
    elapsed_list.push(measure!(mat3.slice(s![.., .., NewAxis]), n_measures));

    println!("mat.reversed_axes");
    elapsed_list.push(measure!(mat.t(), n_measures));

    println!("Array::from_iter(mat.iter().cloned())");
    elapsed_list.push(measure!(Array::from_iter(mat.iter().cloned()), n_measures));

    println!("Type conversion:");
    println!("convert u8 to f32; mat.mapv(|v| f32::from(v))");
    let mat: Array2<u8> =
        initialize_randomized_ndarray(DistributionEnum::Uniform, SHAPE, &[0., 255.])
            .mapv(|v| v as u8);
    elapsed_list.push(measure!(mat.mapv(|v| f32::from(v)), n_measures));

    println!("convert u8 to i32; mat.mapv(|v| i32::from(v))");
    elapsed_list.push(measure!(mat.mapv(|v| i32::from(v)), n_measures));

    println!("try to convert i8 to u8; mat.mapv(|v| u8::try_from(v).unwrap())");
    let mat: Array2<i8> =
        initialize_randomized_ndarray(DistributionEnum::Uniform, SHAPE, &[0., 127.])
            .mapv(|v| v as i8);
    elapsed_list.push(measure!(mat.mapv(|v| u8::try_from(v).unwrap()), n_measures));

    println!("convert f32 to i32; mat.mapv(|v| v as i32)");
    let mat: Array2<f32> =
        initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE, &[0., 1000.])
            .mapv(|v| v as f32);
    elapsed_list.push(measure!(mat.mapv(|v| v as i32), n_measures));

    let mut file: File = File::create(SAVE_PATH).unwrap();
    for v in elapsed_list {
        for v_ in v {
            write!(file, "{},", v_.to_string()).unwrap();
        }
        write!(file, "\n").unwrap();
    }
}

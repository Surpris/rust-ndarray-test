extern crate rust_ndarray_test;

// use csv::Writer;
use ndarray::prelude::*;
use rust_ndarray_test::common::ndarray_init::*;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const SHAPE: (usize, usize) = (1024, 1024);
const N_MEASURES: usize = 10;
const SAVE_PATH: &str = "./data/process_time.csv";

macro_rules! measure {
    ($x:expr, $y: tt) => {{
        let mut elapsed: f64 = 0.0;
        for _ in (0..$y) {
            let start = Instant::now();
            let _1 = $x;
            let end = start.elapsed();
            elapsed += end.as_secs() as f64 + end.subsec_micros() as f64 * 1E-6 + end.subsec_nanos() as f64 * 1E-9;
        }
        let elapsed_msec: f64 = 1000. * elapsed / $y as f64;
        println!(
            "averaged process time over {} times: {:.6} msec.",
            $y, elapsed_msec
        );
        elapsed_msec
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
    let mut elapsed_list: Vec<f64> = Vec::new();

    println!("Array creation:");
    println!("Array::range");
    let first: f64 = 0.0;
    let end: f64 = 10.0;
    let step: f64 = (end - first) / (SHAPE.0 * SHAPE.1) as f64;
    elapsed_list.push(measure!(
        Array::<f64, Ix1>::range(first, end, step),
        N_MEASURES
    ));

    println!("Array::linspace");
    let first: f64 = 0.0;
    let end: f64 = 10.0;
    elapsed_list.push(measure!(
        Array::<f64, Ix1>::linspace(first, end, SHAPE.0 * SHAPE.1),
        N_MEASURES
    ));

    println!("Array::ones");
    elapsed_list.push(measure!(Array::<f64, Ix2>::ones(SHAPE), N_MEASURES));

    println!("Array::zeros");
    elapsed_list.push(measure!(Array::<f64, Ix2>::zeros(SHAPE), N_MEASURES));

    println!("Array::from_elem");
    let v: f64 = 7.0;
    elapsed_list.push(measure!(Array::from_elem(SHAPE, v), N_MEASURES));

    println!("Array::eye");
    elapsed_list.push(measure!(Array::<f64, Ix2>::eye(SHAPE.0), N_MEASURES));

    println!("Randomiz:");
    println!("Normal distribution");
    let mu: f64 = 0.0;
    let sigma: f64 = 1.0;
    elapsed_list.push(measure!(
        test_create_normal_distributed_ndarray(mu, sigma, SHAPE),
        N_MEASURES
    ));

    println!("Poisson distribution");
    let mu: f64 = 10.0;
    elapsed_list.push(measure!(
        test_create_poisson_distributed_ndarray(mu, SHAPE),
        N_MEASURES
    ));

    println!("Uniform distributino");
    let low: f64 = 0.0;
    let high: f64 = 1.0;
    elapsed_list.push(measure!(
        test_create_uniform_distributed_ndarray(low, high, SHAPE),
        N_MEASURES
    ));

    println!("Mathematics:");
    let mu: f64 = 0.0;
    let sigma: f64 = 1.0;
    let vec: Array1<f64> = initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE.0 * SHAPE.1, &[mu, sigma]);
    let vec2: Array1<f64> = initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE.0 * SHAPE.1, &[mu, sigma]);
    let mat: Array2<f64> = initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE, &[mu, sigma]);
    let mat2: Array2<f64> = initialize_randomized_ndarray(DistributionEnum::Normal, SHAPE, &[mu, sigma]);

    println!("mat.reversed_axes");
    elapsed_list.push(measure!(mat.t(), N_MEASURES));

    println!("mat.dot(&mat2)");
    elapsed_list.push(measure!(mat.dot(&mat2), N_MEASURES));

    println!("a.dot(&vec)");
    elapsed_list.push(measure!(mat.dot(&vec.slice(s![0..SHAPE.0])), N_MEASURES));

    println!("vec.dot(&mat)");
    elapsed_list.push(measure!(vec.slice(s![0..SHAPE.0]).dot(&mat), N_MEASURES));

    println!("vec.dot(&vec2)");
    elapsed_list.push(measure!(vec.dot(&vec2), N_MEASURES));

    println!("&mat + &mat2");
    elapsed_list.push(measure!(&mat + &mat2, N_MEASURES));

    println!("mat**3");
    elapsed_list.push(measure!(mat.mapv(|v| v.powi(3)), N_MEASURES));

    println!("sqrt(mat)");
    elapsed_list.push(measure!(mat.mapv(f64::sqrt), N_MEASURES));

    println!("mat > 0.5");
    elapsed_list.push(measure!(mat.mapv(|v| v > 0.5), N_MEASURES));

    println!("mat.sum()");
    elapsed_list.push(measure!(mat.sum(), N_MEASURES));

    println!("mat.sum_axis(Axis(1))");
    elapsed_list.push(measure!(mat.sum_axis(Axis(1)), N_MEASURES));

    println!("mat.mean()");
    elapsed_list.push(measure!(mat.mean(), N_MEASURES));

    println!("mat.mean_axis(Axis(1))");
    elapsed_list.push(measure!(mat.mean_axis(Axis(1)), N_MEASURES));

    println!("mat.abs_diff_eq(&mat2, 1E-8)");
    elapsed_list.push(measure!(mat.abs_diff_eq(&mat2, 1E-8), N_MEASURES));

    println!("mat.diag()");
    elapsed_list.push(measure!(mat.diag(), N_MEASURES));

    let mut file = File::create(SAVE_PATH).unwrap();
    for v in elapsed_list {
        writeln!(file, "{}", v.to_string()).unwrap();
    }
}

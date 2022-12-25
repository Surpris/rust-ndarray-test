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
            elapsed += end.as_secs() as f64 + end.subsec_micros() as f64 * 1E-6;
        }
        let elapsed_msec: f64 = 1000. * elapsed / $y as f64;
        println!(
            "averaged process time over {} times: {:.5} msec.",
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

    println!("Array::range");
    elapsed_list.push(measure!(
        Array::<f64, Ix1>::linspace(0., 10., SHAPE.0 * SHAPE.1),
        N_MEASURES
    ));

    println!("Array::linspace");
    elapsed_list.push(measure!(
        Array::<f64, Ix1>::linspace(0., 10., SHAPE.0 * SHAPE.1),
        N_MEASURES
    ));

    println!("Array::ones");
    elapsed_list.push(measure!(Array::<f64, Ix2>::ones(SHAPE), N_MEASURES));

    println!("Array::zeros");
    elapsed_list.push(measure!(Array::<f64, Ix2>::zeros(SHAPE), N_MEASURES));

    println!("test_create_normal_distributed_ndarray");
    let mu: f64 = 0.0;
    let sigma: f64 = 1.0;
    elapsed_list.push(measure!(
        test_create_normal_distributed_ndarray(mu, sigma, SHAPE),
        N_MEASURES
    ));

    println!("test_create_poisson_distributed_ndarray");
    let mu: f64 = 10.0;
    elapsed_list.push(measure!(
        test_create_poisson_distributed_ndarray(mu, SHAPE),
        N_MEASURES
    ));

    println!("test_create_uniform_distributed_ndarray");
    let low: f64 = 0.0;
    let high: f64 = 1.0;
    elapsed_list.push(measure!(
        test_create_uniform_distributed_ndarray(low, high, SHAPE),
        N_MEASURES
    ));
    let mut file = File::create(SAVE_PATH).unwrap();
    for v in elapsed_list {
        writeln!(file, "{}", v.to_string()).unwrap();
    }
}

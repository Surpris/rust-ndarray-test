extern crate rust_ndarray_test;

use ndarray::prelude::*;
use std::time::Instant;
use rust_ndarray_test::common::ndarray_init::*;

macro_rules! measure {
    ($x:expr) => {
        {
            let start = Instant::now();
            let result = $x;
            let end = start.elapsed();
            let elapsed: f64 = end.as_secs() as f64 + end.subsec_micros() as f64 * 1E-6;
            println!("elapsed time: {} sec.", elapsed);
            result
        }
    };
}

fn test_create_ones_array<D, Sh>(shape: Sh) -> Array<f64, D>
where D: Dimension, Sh : ShapeBuilder<Dim = D> {
    Array::<f64, D>::ones(shape)
}

fn test_create_zeros_array<D, Sh>(shape: Sh) -> Array<f64, D>
where D: Dimension, Sh : ShapeBuilder<Dim = D> {
    Array::<f64, D>::zeros(shape)
}

fn test_create_normal_distributed_ndarray<D, Sh>(shape: Sh) -> Array<f64, D>
where D: Dimension, Sh : ShapeBuilder<Dim = D> {
    let mu: f64 = 1.0;
    let sigma: f64 = 0.5;
    initialize_randomized_ndarray(DistributionEnum::Normal, shape, &[mu, sigma])
}

fn main() {
    let shape = (1000, 1000);

    println!("test_create_ones_array");
    measure!(test_create_ones_array(shape));

    println!("test_create_zeros_array");
    measure!(test_create_zeros_array(shape));

    println!("test_create_normal_distributed_ndarray");
    measure!(test_create_normal_distributed_ndarray(shape));
    println!("Hello, world!");
}
